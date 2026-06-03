from __future__ import annotations

import math
import random
from threading import Lock
from typing import Iterable, Sequence

import numpy as np
from pymoo.indicators.hv import HV

from ...base import Function
from ...task.optimization.hv_utils import scale_hypervolume

EPS = 1e-12


def default_preference_vectors(objective_num: int) -> np.ndarray:
    if objective_num < 2:
        raise ValueError("PBCLLM requires at least two problem objectives.")
    if objective_num == 2:
        return np.array(
            [
                [0.9, 0.1],
                [0.7, 0.3],
                [0.5, 0.5],
                [0.3, 0.7],
                [0.1, 0.9],
            ],
            dtype=float,
        )

    dirs = []
    residual = 0.15 / (objective_num - 1)
    for i in range(objective_num):
        row = np.full(objective_num, residual, dtype=float)
        row[i] = 0.85
        dirs.append(row)
    dirs.append(np.full(objective_num, 1.0 / objective_num, dtype=float))
    for i in range(objective_num):
        for j in range(i + 1, objective_num):
            row = np.full(objective_num, 0.05 / max(1, objective_num - 2), dtype=float)
            row[i] = 0.475
            row[j] = 0.475
            row = row / np.sum(row)
            dirs.append(row)
    return np.array(dirs, dtype=float)


def _as_points(values) -> np.ndarray:
    if values is None:
        return np.empty((0, 0), dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return np.empty((0, 0), dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.size == 0:
        return np.empty((0, 0), dtype=float)
    return arr


def _nondominated_min(points: np.ndarray) -> np.ndarray:
    points = _as_points(points)
    if points.size == 0:
        return points
    points = np.unique(points, axis=0)
    keep = np.ones(len(points), dtype=bool)
    for idx, point in enumerate(points):
        if not keep[idx]:
            continue
        dominated_by_other = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        if np.any(dominated_by_other):
            keep[idx] = False
    return points[keep]


def _mean_scaled_hv(fronts: list, ref_point: np.ndarray | None) -> float:
    if ref_point is None:
        return 0.0
    ref = np.asarray(ref_point, dtype=float)
    if ref.ndim != 1 or not np.all(np.isfinite(ref)):
        return 0.0
    hv_indicator = HV(ref_point=ref)
    values = []
    for front in fronts:
        points = _nondominated_min(_as_points(front))
        if points.size == 0 or points.shape[1] != len(ref):
            values.append(0.0)
            continue
        values.append(scale_hypervolume(hv_indicator(points), ref))
    return float(np.mean(values)) if values else 0.0


def population_front_hv(functions: Sequence[Function], ref_point: np.ndarray | None) -> float:
    if ref_point is None or not functions:
        return 0.0
    ref = np.asarray(ref_point, dtype=float)
    valid_fronts = [
        (getattr(func, "pbc", None) or {}).get("fronts")
        for func in functions
        if (getattr(func, "pbc", None) or {}).get("fronts")
    ]
    if not valid_fronts:
        return 0.0
    n_instance = max(len(fronts) for fronts in valid_fronts)
    hv_indicator = HV(ref_point=ref)
    values = []
    for instance_idx in range(n_instance):
        merged = []
        for fronts in valid_fronts:
            if instance_idx >= len(fronts):
                continue
            points = _as_points(fronts[instance_idx])
            if points.size and points.shape[1] == len(ref):
                merged.append(points)
        if not merged:
            values.append(0.0)
            continue
        front = _nondominated_min(np.vstack(merged))
        values.append(scale_hypervolume(hv_indicator(front), ref))
    return float(np.mean(values)) if values else 0.0


def _normalize(points: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    return np.clip((points - ideal) / np.maximum(nadir - ideal, EPS), 0.0, None)


def _preference_scores(points: np.ndarray, preference_vectors: np.ndarray, ideal: np.ndarray, nadir: np.ndarray, rho: float) -> np.ndarray:
    if points.size == 0:
        return np.full(len(preference_vectors), 1e6, dtype=float)
    norm = _normalize(points, ideal, nadir)
    scores = []
    for vector in preference_vectors:
        weighted = norm * vector
        asf = np.max(weighted, axis=1) + rho * np.sum(weighted, axis=1)
        scores.append(float(np.min(asf)))
    return np.array(scores, dtype=float)


def _resolve_bounds(
    objective_num: int,
    trajectory,
    fallback_points: np.ndarray,
    normalization_ideal: np.ndarray | None,
    normalization_nadir: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if normalization_ideal is not None and normalization_nadir is not None:
        ideal = np.asarray(normalization_ideal, dtype=float)
        nadir = np.asarray(normalization_nadir, dtype=float)
        if ideal.shape == (objective_num,) and nadir.shape == (objective_num,):
            return ideal, nadir

    initial_points = None
    if isinstance(trajectory, list) and trajectory:
        initial_points = _as_points(trajectory[0])
        if initial_points.size and initial_points.shape[1] != objective_num:
            initial_points = None

    basis = initial_points if initial_points is not None and initial_points.size else fallback_points
    if basis.size == 0 or basis.shape[1] != objective_num:
        return None
    return np.min(basis, axis=0), np.max(basis, axis=0)


def analyze_mo_result(
    result: dict,
    preference_vectors: np.ndarray,
    rho: float = 0.05,
    *,
    normalization_ideal: np.ndarray | None = None,
    normalization_nadir: np.ndarray | None = None,
    hv_ref_point: np.ndarray | None = None,
) -> dict | None:
    if not isinstance(result, dict):
        return None
    fronts = result.get("fronts")
    trajectories = result.get("archive_trajectories")
    objective_num = int(result.get("objective_num", 0) or 0)
    if objective_num < 2 or not isinstance(fronts, list) or not isinstance(trajectories, list):
        return None

    sanitized_fronts = []
    per_instance_pbt = []
    per_instance_final = []
    for front, trajectory in zip(fronts, trajectories):
        final_points = _as_points(front)
        if final_points.size == 0 or final_points.shape[1] != objective_num:
            continue
        final_points = _nondominated_min(final_points)
        sanitized_fronts.append(final_points.tolist())

        bounds = _resolve_bounds(
            objective_num,
            trajectory,
            final_points,
            normalization_ideal,
            normalization_nadir,
        )
        if bounds is None:
            continue
        ideal, nadir = bounds

        pbt_rows = []
        max_archive = max(1, max(len(_as_points(archive_points)) for archive_points in trajectory))
        for archive_points in trajectory:
            points = _as_points(archive_points)
            if points.size == 0 or points.shape[1] != objective_num:
                preference_scores = np.full(len(preference_vectors), 1e6, dtype=float)
                archive_size = 0.0
            else:
                preference_scores = _preference_scores(points, preference_vectors, ideal, nadir, rho)
                archive_size = len(points) / max_archive
            pbt_rows.append(np.concatenate([preference_scores, [archive_size]]))

        final_scores = _preference_scores(final_points, preference_vectors, ideal, nadir, rho)
        per_instance_final.append(final_scores)
        per_instance_pbt.append(np.array(pbt_rows, dtype=float))

    if not per_instance_final:
        return None

    preference_performance = np.mean(np.vstack(per_instance_final), axis=0)
    coverage_loss = float(np.mean(preference_performance) + np.std(preference_performance))
    legacy_score = result.get("legacy_score")
    individual_hv = None
    if isinstance(legacy_score, (list, tuple)) and legacy_score:
        try:
            individual_hv = -float(legacy_score[0])
        except (TypeError, ValueError):
            individual_hv = None
    if individual_hv is None or not math.isfinite(individual_hv):
        individual_hv = _mean_scaled_hv(sanitized_fronts, hv_ref_point)
    return {
        "objective_num": objective_num,
        "fronts": sanitized_fronts,
        "individual_hv": individual_hv,
        "coverage_loss": coverage_loss,
        "preference_performance": preference_performance.tolist(),
        "pbt": [pbt.tolist() for pbt in per_instance_pbt],
        "front_count": len(per_instance_final),
    }


def dtw_distance(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.ndim != 2 or bb.ndim != 2 or len(aa) == 0 or len(bb) == 0:
        return 1e6
    n, m = len(aa), len(bb)
    dp = np.full((n + 1, m + 1), float("inf"), dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(aa[i - 1] - bb[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / (n + m))


def behavior_distance(a: Function, b: Function) -> float:
    pbc_a = getattr(a, "pbc", None) or {}
    pbc_b = getattr(b, "pbc", None) or {}
    traj_a = pbc_a.get("pbt") or []
    traj_b = pbc_b.get("pbt") or []
    if not traj_a or not traj_b:
        return 1e6
    count = min(len(traj_a), len(traj_b))
    if count == 0:
        return 1e6
    return float(np.mean([dtw_distance(traj_a[i], traj_b[i]) for i in range(count)]))


def behavior_novelty(func: Function, population: Sequence[Function], k: int = 3) -> float:
    others = [g for g in population if g is not func and getattr(g, "pbc", None)]
    if not others:
        return 1.0
    distances = sorted(behavior_distance(func, other) for other in others)
    return float(np.mean(distances[: max(1, min(k, len(distances)))]))


def _dominates_max(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def _fast_non_dominated_sort_max(items: Sequence[Function], objectives) -> list[list[Function]]:
    domination_sets: dict[int, list[int]] = {}
    dominated_counts: dict[int, int] = {}
    fronts: list[list[int]] = [[]]
    values = [objectives(item) for item in items]

    for i, value_i in enumerate(values):
        domination_sets[i] = []
        dominated_counts[i] = 0
        for j, value_j in enumerate(values):
            if i == j:
                continue
            if _dominates_max(value_i, value_j):
                domination_sets[i].append(j)
            elif _dominates_max(value_j, value_i):
                dominated_counts[i] += 1
        if dominated_counts[i] == 0:
            fronts[0].append(i)

    front_idx = 0
    while front_idx < len(fronts) and fronts[front_idx]:
        next_front = []
        for i in fronts[front_idx]:
            for j in domination_sets[i]:
                dominated_counts[j] -= 1
                if dominated_counts[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        front_idx += 1
    return [[items[idx] for idx in front] for front in fronts if front]


def _crowding_distance_max(front: Sequence[Function], objectives) -> dict[int, float]:
    if not front:
        return {}
    values = [objectives(item) for item in front]
    objective_num = len(values[0])
    distances = {id(item): 0.0 for item in front}
    if len(front) <= 2:
        return {id(item): float("inf") for item in front}

    for obj_idx in range(objective_num):
        order = sorted(range(len(front)), key=lambda idx: values[idx][obj_idx])
        min_value = values[order[0]][obj_idx]
        max_value = values[order[-1]][obj_idx]
        distances[id(front[order[0]])] = float("inf")
        distances[id(front[order[-1]])] = float("inf")
        if abs(max_value - min_value) <= EPS:
            continue
        for rank_idx in range(1, len(order) - 1):
            prev_value = values[order[rank_idx - 1]][obj_idx]
            next_value = values[order[rank_idx + 1]][obj_idx]
            distances[id(front[order[rank_idx]])] += (next_value - prev_value) / (max_value - min_value)
    return distances


class Population:
    def __init__(
        self,
        pop_size: int,
        preference_vectors: np.ndarray,
        *,
        hv_ref_point: np.ndarray | None = None,
        normalization_ideal: np.ndarray | None = None,
        normalization_nadir: np.ndarray | None = None,
        rho: float = 0.05,
        behavior_novelty_weight: float = 0.2,
        cluster_distance: float = 0.35,
        elites_per_preference: int = 2,
    ):
        self._population: list[Function] = []
        self._pop_size = int(pop_size)
        self._preference_vectors = np.asarray(preference_vectors, dtype=float)
        self._hv_ref_point = None if hv_ref_point is None else np.asarray(hv_ref_point, dtype=float)
        self._normalization_ideal = None if normalization_ideal is None else np.asarray(normalization_ideal, dtype=float)
        self._normalization_nadir = None if normalization_nadir is None else np.asarray(normalization_nadir, dtype=float)
        self._rho = float(rho)
        self._behavior_novelty_weight = float(behavior_novelty_weight)
        self._cluster_distance = float(cluster_distance)
        self._elites_per_preference = int(elites_per_preference)
        self._population_hv = 0.0
        self._generation = 0
        self._lock = Lock()

    @property
    def population(self) -> list[Function]:
        return self._population

    @property
    def generation(self) -> int:
        return self._generation

    def __len__(self):
        return len(self._population)

    def register_function(self, func: Function):
        if getattr(func, "score", None) is None or not getattr(func, "pbc", None):
            return
        with self._lock:
            self._population.append(func)
            self._refresh_behavior_novelty()
            self._population = self._pruned()
            self._refresh_population_metrics()
            self._generation += 1

    def _refresh_behavior_novelty(self):
        for func in self._population:
            novelty = behavior_novelty(func, self._population)
            func.pbc["behavior_novelty"] = novelty

    @staticmethod
    def _sp(func: Function) -> np.ndarray:
        pbc = getattr(func, "pbc", None) or {}
        return np.asarray(pbc.get("preference_performance", []), dtype=float)

    @staticmethod
    def _individual_hv(func: Function) -> float:
        value = (getattr(func, "pbc", None) or {}).get("individual_hv", 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _coverage_loss(func: Function) -> float:
        pbc = getattr(func, "pbc", None) or {}
        if pbc.get("coverage_loss") is not None:
            try:
                return float(pbc["coverage_loss"])
            except (TypeError, ValueError):
                pass
        sp = np.asarray(pbc.get("preference_performance", []), dtype=float)
        if sp.size == 0:
            return 1e6
        return float(np.mean(sp) + np.std(sp))

    @staticmethod
    def _hv_contribution(func: Function) -> float:
        value = (getattr(func, "pbc", None) or {}).get("population_hv_contribution", 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _behavior_diversity(func: Function) -> float:
        pbc = getattr(func, "pbc", None) or {}
        value = pbc.get("behavior_diversity", pbc.get("behavior_novelty", 0.0))
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _population_preference_performance(self, functions: Sequence[Function]) -> np.ndarray | None:
        if not functions or self._normalization_ideal is None or self._normalization_nadir is None:
            return None
        valid_fronts = [
            (getattr(func, "pbc", None) or {}).get("fronts")
            for func in functions
            if (getattr(func, "pbc", None) or {}).get("fronts")
        ]
        if not valid_fronts:
            return None
        instance_count = max(len(fronts) for fronts in valid_fronts)
        scores = []
        for instance_idx in range(instance_count):
            merged = []
            for fronts in valid_fronts:
                if instance_idx >= len(fronts):
                    continue
                points = _as_points(fronts[instance_idx])
                if points.size:
                    merged.append(points)
            if not merged:
                continue
            front = _nondominated_min(np.vstack(merged))
            if front.size:
                scores.append(
                    _preference_scores(
                        front,
                        self._preference_vectors,
                        self._normalization_ideal,
                        self._normalization_nadir,
                        self._rho,
                    )
                )
        if not scores:
            return None
        return np.mean(np.vstack(scores), axis=0)

    def _annotate_pool_metrics(self, pool: Sequence[Function]):
        pool_hv = population_front_hv(pool, self._hv_ref_point)
        for func in pool:
            pbc = getattr(func, "pbc", None) or {}
            without = [other for other in pool if other is not func]
            contribution = max(0.0, pool_hv - population_front_hv(without, self._hv_ref_point))
            diversity = behavior_novelty(func, pool)
            pbc["selection_population_hv"] = pool_hv
            pbc["selection_hv_gain"] = contribution
            pbc["population_hv_contribution"] = contribution
            pbc["behavior_diversity"] = diversity
            pbc["behavior_novelty"] = diversity
            pbc["coverage_loss"] = self._coverage_loss(func)
            func.pbc = pbc

    def _selection_objectives(self, func: Function) -> tuple[float, float]:
        pbc = getattr(func, "pbc", None) or {}
        return (
            float(pbc.get("selection_hv_gain", pbc.get("population_hv_contribution", 0.0))),
            float(pbc.get("behavior_diversity", pbc.get("behavior_novelty", 0.0))),
        )

    def _select_by_contribution_diversity(self, pool: Sequence[Function]) -> list[Function]:
        self._annotate_pool_metrics(pool)
        fronts = _fast_non_dominated_sort_max(pool, self._selection_objectives)
        selected: list[Function] = []
        for front in fronts:
            if len(selected) + len(front) <= self._pop_size:
                selected.extend(front)
                continue

            remaining_slots = self._pop_size - len(selected)
            crowding = _crowding_distance_max(front, self._selection_objectives)
            ranked = sorted(
                front,
                key=lambda func: (
                    crowding.get(id(func), 0.0),
                    self._individual_hv(func),
                    -self._coverage_loss(func),
                ),
                reverse=True,
            )
            selected.extend(ranked[:remaining_slots])
            break
        return selected

    def _refresh_population_metrics(self):
        self._population_hv = population_front_hv(self._population, self._hv_ref_point)
        population_preference = self._population_preference_performance(self._population)
        for func in self._population:
            pbc = getattr(func, "pbc", None) or {}
            without = [other for other in self._population if other is not func]
            contribution = max(0.0, self._population_hv - population_front_hv(without, self._hv_ref_point))
            diversity = behavior_novelty(func, self._population)
            pbc["population_hv"] = self._population_hv
            pbc["population_hv_contribution"] = contribution
            pbc["selection_hv_gain"] = contribution
            pbc["behavior_diversity"] = diversity
            pbc["behavior_novelty"] = diversity
            pbc["coverage_loss"] = self._coverage_loss(func)
            if population_preference is not None:
                pbc["population_preference_performance"] = population_preference.tolist()
            func.pbc = pbc
            func.score = [
                -contribution,
                -diversity,
            ]

    def _pruned(self) -> list[Function]:
        if len(self._population) <= self._pop_size:
            return list(self._population)
        return self._select_by_contribution_diversity(self._population)

    def weakest_preference(self) -> int:
        if not self._population:
            return random.randrange(len(self._preference_vectors))
        population_preference = self._population_preference_performance(self._population)
        if population_preference is not None and len(population_preference):
            return int(np.argmax(population_preference))
        best_scores = []
        for preference_idx in range(len(self._preference_vectors)):
            best = min(float(self._sp(func)[preference_idx]) for func in self._population if self._sp(func).size)
            best_scores.append(best)
        return int(np.argmax(best_scores))

    def select_parents(self, target_preference: int, selection_num: int = 3) -> list[Function]:
        if not self._population:
            return []
        parents: list[Function] = []
        p1 = max(self._population, key=self._individual_hv)
        parents.append(p1)

        if len(self._population) > 1:
            p2 = max((f for f in self._population if f is not p1), key=self._hv_contribution)
            parents.append(p2)

        useful = [
            func for func in self._population
            if func not in parents and self._hv_contribution(func) > 0.0
        ]
        if useful:
            p3 = max(useful, key=self._behavior_diversity)
        else:
            p3 = min(self._population, key=lambda f: float(self._sp(f)[target_preference]))
        if p3 not in parents:
            parents.append(p3)

        while len(parents) < min(selection_num, len(self._population)):
            candidate = random.choice(self._population)
            if candidate not in parents:
                parents.append(candidate)
        return parents[:selection_num]

    def behavior_summary(self, func: Function) -> str:
        pbc = getattr(func, "pbc", None) or {}
        sp = np.asarray(pbc.get("preference_performance", []), dtype=float)
        pbt = pbc.get("pbt") or []
        if sp.size == 0:
            return "No Pareto behavior data available."
        best_preference = int(np.argmin(sp))
        novelty = float(pbc.get("behavior_novelty", 0.0))
        diversity = float(pbc.get("behavior_diversity", novelty))
        individual_hv = float(pbc.get("individual_hv", 0.0))
        contribution = float(pbc.get("population_hv_contribution", 0.0))
        coverage_loss = float(pbc.get("coverage_loss", self._coverage_loss(func)))
        trend = "unknown"
        if pbt:
            arr = np.asarray(pbt[0], dtype=float)
            if arr.ndim == 2 and len(arr) >= 2 and arr.shape[1] > best_preference:
                improvement = float(arr[0, best_preference] - arr[-1, best_preference])
                trend = f"preference vector {best_preference} trajectory improvement {improvement:.3f}"
        return (
            f"individual HV={individual_hv:.6f}, population HV contribution={contribution:.6f}, "
            f"coverage loss={coverage_loss:.6f}, best preference vector index={best_preference}, preference scores="
            f"{[round(float(v), 4) for v in sp]}, behavior diversity={diversity:.4f}, {trend}."
        )
