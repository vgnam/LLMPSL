from __future__ import annotations

import math
import random
from threading import Lock
from typing import List, Sequence

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from ...base import Function

EPS = 1e-12


def is_dominated(obj1, obj2):
    """Return True if obj1 dominates obj2 for minimization."""
    return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))


def fast_non_dominated_sort(population):
    fronts = []
    S = {}
    n = {}
    rank = {}

    for i, p in enumerate(population):
        S[i] = []
        n[i] = 0
        for j, q in enumerate(population):
            if i == j:
                continue
            if is_dominated(p.score, q.score):
                S[i].append(j)
            elif is_dominated(q.score, p.score):
                n[i] += 1
        if n[i] == 0:
            rank[i] = 0
            if len(fronts) == 0:
                fronts.append([])
            fronts[0].append(i)

    i = 0
    while i < len(fronts):
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in S[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    rank[q_idx] = i + 1
                    next_front.append(q_idx)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts


def calculate_crowding_distance(population, indices):
    distances = {i: 0.0 for i in indices}
    if len(indices) <= 2:
        for i in indices:
            distances[i] = float("inf")
        return distances
    num_objectives = len(population[indices[0]].score)
    for m in range(num_objectives):
        indices.sort(key=lambda i: population[i].score[m])
        distances[indices[0]] = distances[indices[-1]] = float("inf")
        min_obj = population[indices[0]].score[m]
        max_obj = population[indices[-1]].score[m]
        if max_obj == min_obj:
            continue
        for k in range(1, len(indices) - 1):
            prev_obj = population[indices[k - 1]].score[m]
            next_obj = population[indices[k + 1]].score[m]
            distances[indices[k]] += (next_obj - prev_obj) / (max_obj - min_obj)
    return distances


def population_management(population, N):
    funcs = [f for f in population if getattr(f, "score", None) is not None]
    if len(funcs) <= N:
        return funcs
    fronts = fast_non_dominated_sort(funcs)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= N:
            selected.extend(front)
        else:
            remaining = N - len(selected)
            distances = calculate_crowding_distance(funcs, front)
            sorted_front = sorted(front, key=lambda i: -distances[i])
            selected.extend(sorted_front[:remaining])
            break
    return [funcs[i] for i in selected]


# ---------------------------------------------------------------------------
# PFGL: Pareto Front Geometry Learning helpers
# ---------------------------------------------------------------------------

def _non_dominated_mask(points: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated points (minimization)."""
    n = len(points)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                mask[i] = False
                break
    return mask


def extract_front_points(population: Sequence[Function]):
    """Extract non-dominated score points and angular parameterize them.

    Returns
    -------
    theta : np.ndarray
        Angular coordinates in [-pi/2, pi/2] (2-D).
    r : np.ndarray
        Normalised radial distances from ideal point.
    nd_scores : np.ndarray
        Original objective vectors of the non-dominated points.
    ideal : np.ndarray
        Ideal point used for normalisation.
    nadir : np.ndarray
        Nadir point used for normalisation.
    """
    funcs = [f for f in population if getattr(f, "score", None) is not None]
    if not funcs:
        return (np.empty(0), np.empty(0), np.empty((0, 2)),
                np.zeros(2), np.ones(2))

    scores = np.array([list(f.score) for f in funcs], dtype=float)
    # 2-D only for now; generalise to M-D spherical if needed later.
    if scores.shape[1] != 2:
        raise ValueError("LLM-PFGL currently supports 2-objective problems only.")

    ideal = np.min(scores, axis=0)
    nadir = np.max(scores, axis=0)
    denom = np.maximum(nadir - ideal, EPS)
    norm = (scores - ideal) / denom

    nd_mask = _non_dominated_mask(norm)
    nd_norm = norm[nd_mask]
    nd_scores = scores[nd_mask]

    theta = np.arctan2(nd_norm[:, 1], nd_norm[:, 0])
    r = np.linalg.norm(nd_norm, axis=1)
    return theta, r, nd_scores, ideal, nadir


def fit_front_gp(theta_obs: np.ndarray, r_obs: np.ndarray) -> GaussianProcessRegressor | None:
    """Fit a scalar GP  theta -> r  with an RBF+noise kernel."""
    if len(theta_obs) < 3:
        return None
    theta_obs = np.asarray(theta_obs).reshape(-1, 1)
    r_obs = np.asarray(r_obs)
    kernel = RBF(length_scale=0.3, length_scale_bounds=(1e-2, 1.0)) + WhiteKernel(
        noise_level=0.01, noise_level_bounds=(1e-5, 1.0)
    )
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=42)
    try:
        gp.fit(theta_obs, r_obs)
        return gp
    except Exception:
        return None


def detect_surgical_gap(
    gp: GaussianProcessRegressor | None,
    theta_dense: np.ndarray,
    theta_observed: np.ndarray,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return the angular coordinate with highest acquisition score."""
    if gp is None:
        return None, None, None, None

    mu, sigma = gp.predict(theta_dense.reshape(-1, 1), return_std=True)

    # Curvature via central finite differences
    dmu = np.gradient(mu, theta_dense)
    d2mu = np.gradient(dmu, theta_dense)
    curvature = np.abs(d2mu)

    # Feasibility weight: penalise extreme r values
    r_norm = (mu - mu.min()) / (mu.max() - mu.min() + EPS)

    # Exploration penalty near existing observations
    penalty = np.ones_like(theta_dense)
    for th in np.atleast_1d(theta_observed):
        penalty[np.abs(theta_dense - th) < 0.08] *= 0.15

    acquisition = sigma * (1.0 + curvature) * r_norm * penalty
    gap_idx = int(np.argmax(acquisition))
    return (
        float(theta_dense[gap_idx]),
        float(mu[gap_idx]),
        float(sigma[gap_idx]),
        float(curvature[gap_idx]),
    )


def select_nearest_theta(
    population: Sequence[Function], theta_target: float
) -> Function | None:
    """Return the individual whose normalised angular coordinate is closest to theta_target."""
    funcs = [f for f in population if getattr(f, "score", None) is not None]
    if not funcs:
        return None
    scores = np.array([list(f.score) for f in funcs], dtype=float)
    ideal = np.min(scores, axis=0)
    nadir = np.max(scores, axis=0)
    denom = np.maximum(nadir - ideal, EPS)
    norm = (scores - ideal) / denom
    thetas = np.arctan2(norm[:, 1], norm[:, 0])
    idx = int(np.argmin(np.abs(thetas - theta_target)))
    return funcs[idx]


# ---------------------------------------------------------------------------
# Population class
# ---------------------------------------------------------------------------

class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        self._lock = Lock()
        self._next_gen_pop = []
        self._generation = generation

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    @property
    def population(self):
        return self._population

    @property
    def generation(self):
        return self._generation

    def register_function(self, func: Function):
        if func.score is None:
            return
        try:
            self._lock.acquire()
            self._next_gen_pop.append(func)
            if len(self._next_gen_pop) >= self._pop_size:
                merged = self._population + self._next_gen_pop
                self._population = population_management(merged, self._pop_size)
                self._next_gen_pop = []
                self._generation += 1
        except Exception:
            return
        finally:
            self._lock.release()

    def has_duplicate_function(self, func: str | Function) -> bool:
        func_str = str(func)
        func_score = getattr(func, "score", None)
        for f in self._population:
            if str(f) == func_str or (func_score is not None and func_score == f.score):
                return True
        for f in self._next_gen_pop:
            if str(f) == func_str or (func_score is not None and func_score == f.score):
                return True
        return False

    def selection(self, selection_num) -> List[Function]:
        """Tournament selection based on crowded NSGA-II front (fallback uniform)."""
        try:
            funcs = [f for f in self._population if getattr(f, "score", None) is not None]
            if len(funcs) <= selection_num:
                return funcs
            # Simple tournament: random pick, keep non-dominated if possible
            chosen = []
            for _ in range(selection_num):
                candidates = random.sample(funcs, min(3, len(funcs)))
                # pick the one with smallest crowding front rank
                fronts = fast_non_dominated_sort(candidates)
                if fronts:
                    chosen.append(candidates[fronts[0][0]])
            return chosen
        except Exception as e:
            print(e)
            return []

    def select_nearest_theta(self, theta_target: float) -> Function | None:
        return select_nearest_theta(self._population, theta_target)
