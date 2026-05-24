from __future__ import annotations

import ast
import math
import random
import re
from copy import deepcopy
from threading import Lock
from typing import List, Sequence

import numpy as np

from ...base import Function


EPS = 1e-12


def _score_vector(score) -> list[float]:
    if score is None:
        return []
    if isinstance(score, np.ndarray):
        return [float(x) for x in score.tolist()]
    if isinstance(score, (list, tuple)):
        return [float(x) for x in score]
    return [float(score)]


def is_dominated(obj1, obj2):
    """Return True if obj1 dominates obj2 for minimization."""
    obj1 = _score_vector(obj1)
    obj2 = _score_vector(obj2)
    return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))


def dominates(func_a: Function, func_b: Function) -> bool:
    return is_dominated(func_a.score, func_b.score)


def _valid_functions(population: Sequence[Function]) -> list[Function]:
    return [func for func in population if getattr(func, "score", None) is not None]


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
            if not fronts:
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
    if not indices:
        return {}

    distances = {i: 0.0 for i in indices}
    if len(indices) <= 2:
        for i in indices:
            distances[i] = float("inf")
        return distances

    num_objectives = len(_score_vector(population[indices[0]].score))
    for m in range(num_objectives):
        sorted_indices = sorted(indices, key=lambda i: _score_vector(population[i].score)[m])
        distances[sorted_indices[0]] = float("inf")
        distances[sorted_indices[-1]] = float("inf")
        min_obj = _score_vector(population[sorted_indices[0]].score)[m]
        max_obj = _score_vector(population[sorted_indices[-1]].score)[m]
        if abs(max_obj - min_obj) <= EPS:
            continue
        for k in range(1, len(sorted_indices) - 1):
            prev_obj = _score_vector(population[sorted_indices[k - 1]].score)[m]
            next_obj = _score_vector(population[sorted_indices[k + 1]].score)[m]
            distances[sorted_indices[k]] += (next_obj - prev_obj) / (max_obj - min_obj)
    return distances


def _tokenize_code(code: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+\.\d+|\d+|==|!=|<=|>=|[-+*/%<>]", code))


def _ast_node_types(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {type(node).__name__ for node in ast.walk(tree)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def fallback_code_similarity(code_a: str, code_b: str) -> float:
    token_sim = _jaccard(_tokenize_code(code_a), _tokenize_code(code_b))
    ast_sim = _jaccard(_ast_node_types(code_a), _ast_node_types(code_b))
    return float(0.6 * token_sim + 0.4 * ast_sim)


def _codebleu_one_way(reference: str, prediction: str) -> float | None:
    try:
        from codebleu import calc_codebleu

        result = calc_codebleu(
            [reference],
            [prediction],
            lang="python",
            weights=(0.25, 0.25, 0.25, 0.25),
            tokenizer=None,
        )
        return float(result["codebleu"])
    except Exception:
        return None


def code_similarity(code_a: str, code_b: str, *, prefer_codebleu: bool = True) -> float:
    """Symmetric code similarity in [0, 1].

    Uses CodeBLEU when available. Falls back to a cheap token/AST similarity so
    LLMPSL remains runnable without extra dependencies.
    """
    if not code_a or not code_b:
        return 0.0
    if code_a == code_b:
        return 1.0

    if prefer_codebleu:
        sim_ab = _codebleu_one_way(code_a, code_b)
        sim_ba = _codebleu_one_way(code_b, code_a)
        if sim_ab is not None and sim_ba is not None:
            return max(0.0, min(1.0, 0.5 * (sim_ab + sim_ba)))

    return max(0.0, min(1.0, fallback_code_similarity(code_a, code_b)))


def code_distance(code_a: str, code_b: str, *, prefer_codebleu: bool = True) -> float:
    return 1.0 - code_similarity(code_a, code_b, prefer_codebleu=prefer_codebleu)


def code_novelty(func: str | Function, archive: Sequence[str | Function], *, k: int = 2, prefer_codebleu: bool = True) -> float:
    """Novelty is the mean distance to the k nearest existing programs."""
    code = str(func)
    codes = [str(other) for other in archive if str(other) and str(other) != code]
    if not codes:
        return 1.0

    distances = sorted(code_distance(code, other, prefer_codebleu=prefer_codebleu) for other in codes)
    k = max(1, min(k, len(distances)))
    return float(np.mean(distances[:k]))


def _normalize(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    finite_vals = [v for v in values.values() if math.isfinite(v)]
    if not finite_vals:
        return {k: 1.0 for k in values}
    min_v = min(finite_vals)
    max_v = max(finite_vals)
    if abs(max_v - min_v) <= EPS:
        return {k: 1.0 for k in values}
    return {k: (v - min_v) / (max_v - min_v) if math.isfinite(v) else 1.0 for k, v in values.items()}


def _code_diversity_to_selected(population: Sequence[Function], idx: int, selected: Sequence[int]) -> float:
    if not selected:
        return 1.0
    return min(code_distance(str(population[idx]), str(population[j]), prefer_codebleu=False) for j in selected)


def population_management(population, N):
    population = _valid_functions(population)
    if N <= 0:
        return []
    if len(population) <= N:
        return population

    fronts = fast_non_dominated_sort(population)
    selected = []

    for front in fronts:
        if len(selected) + len(front) <= N:
            selected.extend(front)
            continue

        remaining = N - len(selected)
        distances = calculate_crowding_distance(population, front)
        norm_crowding = _normalize(distances)

        chosen = []
        candidates = set(front)
        while len(chosen) < remaining and candidates:
            best_idx = None
            best_score = -float("inf")
            current_selected = selected + chosen
            div_values = {
                idx: _code_diversity_to_selected(population, idx, current_selected)
                for idx in candidates
            }
            norm_div = _normalize(div_values)
            for idx in candidates:
                score = 0.65 * norm_crowding.get(idx, 0.0) + 0.35 * norm_div.get(idx, 0.0)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            chosen.append(best_idx)
            candidates.remove(best_idx)
        selected.extend(chosen)
        break

    return [population[i] for i in selected]


def default_preference_grid(objective_num: int = 3) -> list[tuple[float, ...]]:
    if objective_num == 2:
        return [
            (0.9, 0.1),
            (0.7, 0.3),
            (0.5, 0.5),
            (0.3, 0.7),
            (0.1, 0.9),
        ]
    if objective_num == 3:
        return [
            (0.80, 0.10, 0.10),
            (0.65, 0.25, 0.10),
            (0.45, 0.45, 0.10),
            (0.25, 0.65, 0.10),
            (0.10, 0.80, 0.10),
            (0.50, 0.20, 0.30),
            (0.30, 0.30, 0.40),
            (0.20, 0.40, 0.40),
            (1 / 3, 1 / 3, 1 / 3),
        ]

    grid = []
    for i in range(objective_num):
        pref = np.full(objective_num, 0.1 / max(1, objective_num - 1))
        pref[i] = 0.9
        grid.append(tuple(float(x) for x in pref / np.sum(pref)))
    grid.append(tuple(float(1 / objective_num) for _ in range(objective_num)))
    return grid


def _fit_preference(preference: Sequence[float] | None, objective_num: int) -> np.ndarray:
    if preference is None:
        pref = np.ones(objective_num, dtype=float)
    else:
        pref = np.array([float(x) for x in preference], dtype=float)
        if len(pref) < objective_num:
            pref = np.pad(pref, (0, objective_num - len(pref)), constant_values=EPS)
        elif len(pref) > objective_num:
            pref = pref[:objective_num]
    pref = np.maximum(pref, EPS)
    return pref / np.sum(pref)


def score_bounds(population: Sequence[Function], objective_num: int | None = None):
    funcs = _valid_functions(population)
    if not funcs:
        if objective_num is None:
            objective_num = 1
        return np.zeros(objective_num), np.ones(objective_num)

    scores = np.array([_score_vector(func.score) for func in funcs], dtype=float)
    if objective_num is None:
        objective_num = scores.shape[1]
    ideal = np.min(scores[:, :objective_num], axis=0)
    nadir = np.max(scores[:, :objective_num], axis=0)
    return ideal, nadir


def _normalized_score_matrix(functions: Sequence[Function], ideal, nadir, objective_num: int | None = None) -> np.ndarray:
    funcs = _valid_functions(functions)
    if not funcs:
        if objective_num is None:
            objective_num = len(ideal)
        return np.empty((0, objective_num), dtype=float)

    scores = np.array([_score_vector(func.score) for func in funcs], dtype=float)
    if objective_num is None:
        objective_num = scores.shape[1]
    ideal = np.array(ideal[:objective_num], dtype=float)
    nadir = np.array(nadir[:objective_num], dtype=float)
    denom = np.maximum(nadir - ideal, EPS)
    return (scores[:, :objective_num] - ideal) / denom


def augmented_tchebycheff(score, preference, ideal, nadir, rho: float = 0.05) -> float:
    score = np.array(_score_vector(score), dtype=float)
    objective_num = len(score)
    preference = _fit_preference(preference, objective_num)
    ideal = np.array(ideal[:objective_num], dtype=float)
    nadir = np.array(nadir[:objective_num], dtype=float)
    denom = np.maximum(nadir - ideal, EPS)
    normalized = (score - ideal) / denom
    weighted = preference * normalized
    return float(np.max(weighted) + rho * np.sum(weighted))


def smooth_tchebycheff_scalarization(score, preference, ideal, nadir, *, rho: float = 0.05, smooth_mu: float = 0.05) -> float:
    """Smooth Tchebycheff scalarization for one heuristic program."""
    score = np.array(_score_vector(score), dtype=float)
    objective_num = len(score)
    preference = _fit_preference(preference, objective_num)
    ideal = np.array(ideal[:objective_num], dtype=float)
    nadir = np.array(nadir[:objective_num], dtype=float)
    denom = np.maximum(nadir - ideal, EPS)
    normalized = (score - ideal) / denom
    weighted = preference * normalized
    mu = max(float(smooth_mu), EPS)
    return float(mu * _logsumexp(weighted / mu) + rho * np.sum(weighted))


def _logsumexp(values: np.ndarray) -> float:
    if len(values) == 0:
        return -float("inf")
    max_value = float(np.max(values))
    return max_value + math.log(float(np.sum(np.exp(values - max_value))))


def tchebycheff_set_scalarization(
    functions: Sequence[Function],
    preference,
    ideal,
    nadir,
    *,
    rho: float = 0.05,
    smooth: bool = True,
    smooth_mu: float = 0.05,
) -> float:
    """Tchebycheff Set scalarization for a small set of heuristic programs.

    For a set H, each objective is represented by the best member in H:
        min_{h in H} f_i(h)
    and the set score is the worst weighted normalized objective:
        max_i lambda_i * min_{h in H} s_i(h)

    This follows the "Few for Many" TCH-Set idea and uses the Smooth
    Tchebycheff log-sum-exp relaxation by default. Non-smooth TCH-Set is kept
    as a fallback when smooth=False.
    """
    funcs = _valid_functions(functions)
    if not funcs:
        return float("inf")

    objective_num = len(_score_vector(funcs[0].score))
    preference = _fit_preference(preference, objective_num)
    normalized = _normalized_score_matrix(funcs, ideal, nadir, objective_num)

    if smooth:
        mu = max(float(smooth_mu), EPS)
        smooth_best = []
        for j in range(objective_num):
            smooth_best.append(-mu * _logsumexp(-normalized[:, j] / mu))
        representative = np.array(smooth_best, dtype=float)
        weighted = preference * representative
        value = mu * _logsumexp(weighted / mu)
    else:
        representative = np.min(normalized, axis=0)
        weighted = preference * representative
        value = float(np.max(weighted))

    return float(value + rho * np.sum(weighted))


def smooth_tchebycheff_set_scalarization(
    functions: Sequence[Function],
    preference,
    ideal,
    nadir,
    *,
    rho: float = 0.05,
    smooth_mu: float = 0.05,
) -> float:
    return tchebycheff_set_scalarization(
        functions,
        preference,
        ideal,
        nadir,
        rho=rho,
        smooth=True,
        smooth_mu=smooth_mu,
    )


def greedy_tchebycheff_set_selection(
    pop,
    selection_num,
    preference=None,
    *,
    rho: float = 0.05,
    smooth: bool = True,
    smooth_mu: float = 0.05,
    diversity_weight: float = 0.15,
) -> List[Function]:
    funcs = _valid_functions(pop)
    if not funcs:
        return []
    if len(funcs) <= selection_num:
        return funcs

    objective_num = len(_score_vector(funcs[0].score))
    ideal, nadir = score_bounds(funcs, objective_num)
    pref = _fit_preference(preference, objective_num)

    selected = []
    candidates = set(range(len(funcs)))
    while len(selected) < selection_num and candidates:
        best_idx = None
        best_score = float("inf")
        for idx in candidates:
            trial = [funcs[i] for i in selected + [idx]]
            set_value = tchebycheff_set_scalarization(
                trial,
                pref,
                ideal,
                nadir,
                rho=rho,
                smooth=smooth,
                smooth_mu=smooth_mu,
            )
            # Smooth Tchebycheff Set scalarization already balances the
            # objectives in the selected set. Keep code diversity out of the
            # parent score unless explicitly re-enabled for ablation.
            # diversity = _code_diversity_to_selected(funcs, idx, selected)
            # score = set_value - diversity_weight * diversity
            score = set_value
            if score < best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        candidates.remove(best_idx)

    return [funcs[i] for i in selected]


def parent_selection(
    pop,
    selection_num,
    preference=None,
    rho: float = 0.05,
    epsilon: float = 0.1,
    smooth: bool = True,
    smooth_mu: float = 0.05,
) -> List[Function]:
    funcs = _valid_functions(pop)
    if not funcs:
        return []
    if len(funcs) <= selection_num:
        return funcs

    if random.random() < epsilon:
        return random.sample(funcs, k=min(selection_num, len(funcs)))

    return greedy_tchebycheff_set_selection(
        funcs,
        selection_num,
        preference=preference,
        rho=rho,
        smooth=smooth,
        smooth_mu=smooth_mu,
    )


class Population:
    def __init__(
        self,
        pop_size,
        generation=0,
        pop: List[Function] | "Population" | None = None,
        *,
        objective_num: int = 3,
        preference_grid: Sequence[Sequence[float]] | None = None,
        novelty_k: int = 2,
        rho: float = 0.05,
        prefer_codebleu: bool = True,
        set_size: int = 3,
        smooth_set_scalarization: bool = True,
        smooth_mu: float = 0.05,
    ):
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
        self._objective_num = objective_num
        self._novelty_k = novelty_k
        self._rho = rho
        self._prefer_codebleu = prefer_codebleu
        self._set_size = max(1, int(set_size))
        self._smooth_set_scalarization = smooth_set_scalarization
        self._smooth_mu = smooth_mu
        self._preference_grid = [
            tuple(_fit_preference(pref, objective_num).tolist())
            for pref in (preference_grid or default_preference_grid(objective_num))
        ]
        self._preference_memory = {
            pref: {"set": [], "function": None, "value": float("inf"), "updated": -1}
            for pref in self._preference_grid
        }
        self.update_preference_memory()

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def generation(self):
        return self._generation

    @property
    def preference_grid(self):
        return self._preference_grid

    @property
    def preference_memory(self):
        return self._preference_memory

    def all_functions(self, *, include_pending: bool = True) -> list[Function]:
        funcs = list(self._population)
        if include_pending:
            funcs.extend(self._next_gen_pop)
        return _valid_functions(funcs)

    def compute_code_novelty(self, func: str | Function, *, archive: Sequence[Function] | None = None) -> float:
        archive = self.all_functions() if archive is None else archive
        return code_novelty(func, archive, k=self._novelty_k, prefer_codebleu=self._prefer_codebleu)

    def update_preference_memory(self):
        funcs = _valid_functions(self._population)
        if not funcs:
            return

        ideal, nadir = score_bounds(funcs, self._objective_num)
        for pref in self._preference_grid:
            selected_set = greedy_tchebycheff_set_selection(
                funcs,
                min(self._set_size, len(funcs)),
                preference=pref,
                rho=self._rho,
                smooth=self._smooth_set_scalarization,
                smooth_mu=self._smooth_mu,
            )
            best_value = tchebycheff_set_scalarization(
                selected_set,
                pref,
                ideal,
                nadir,
                rho=self._rho,
                smooth=self._smooth_set_scalarization,
                smooth_mu=self._smooth_mu,
            )
            previous = self._preference_memory[pref]
            if selected_set != previous["set"] or best_value < previous["value"] - EPS:
                self._preference_memory[pref] = {
                    "set": selected_set,
                    "function": selected_set[0] if selected_set else None,
                    "value": best_value,
                    "updated": self._generation,
                }

    def select_target_preference(self) -> tuple[float, ...]:
        empty = [pref for pref, entry in self._preference_memory.items() if not entry["set"]]
        if empty:
            return random.choice(empty)

        priorities = {}
        values = {i: entry["value"] for i, entry in enumerate(self._preference_memory.values())}
        norm_values = _normalize(values)
        prefs = list(self._preference_memory.keys())
        for i, pref in enumerate(prefs):
            entry = self._preference_memory[pref]
            stale = max(0, self._generation - entry["updated"])
            priorities[pref] = norm_values[i] + 0.05 * stale + random.random() * 0.01
        return max(priorities, key=priorities.get)

    def register_function(self, func: Function):
        if func.score is None:
            return
        if isinstance(func.score, tuple):
            func.score = list(func.score)
        try:
            self._lock.acquire()
            self._next_gen_pop.append(func)

            if len(self._next_gen_pop) >= self._pop_size:
                pop = self._population + self._next_gen_pop
                self._population = population_management(pop, self._pop_size)
                self._next_gen_pop = []
                self._generation += 1
                self.update_preference_memory()
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

    def has_near_duplicate_function(self, func: str | Function, threshold: float = 0.90) -> bool:
        for old in self.all_functions():
            if code_similarity(str(func), str(old), prefer_codebleu=self._prefer_codebleu) >= threshold:
                return True
        return False

    def selection(self, selection_num, preference=None) -> List[Function]:
        try:
            if preference is None:
                preference = self.select_target_preference()
            pref = tuple(_fit_preference(preference, self._objective_num).tolist())
            memory_entry = self._preference_memory.get(pref)
            selected = []
            if memory_entry is not None:
                selected = list(memory_entry.get("set") or [])[:selection_num]
            if len(selected) >= selection_num:
                return selected

            fill = parent_selection(
                [func for func in self._population if func not in selected],
                selection_num - len(selected),
                preference=preference,
                rho=self._rho,
                smooth=self._smooth_set_scalarization,
                smooth_mu=self._smooth_mu,
            )
            return selected + fill
        except Exception as e:
            print(e)
            return []

    def selection_cluster(self, group, indivs) -> List[Function]:
        try:
            N = len(indivs)
            all_indices = [idx for subgroup in group for idx in subgroup]
            valid_group = (
                len(group) > 1
                and sorted(all_indices) == list(range(N))
                and len(set(all_indices)) == N
            )

            if not valid_group:
                return random.sample(indivs, min(2, len(indivs)))

            group1 = random.choice(group)
            idx1 = random.choice(group1)
            parent1 = indivs[idx1]

            other_groups = [g for g in group if idx1 not in g and len(g) > 0]
            if other_groups:
                group2 = random.choice(other_groups)
                idx2 = random.choice(group2)
            else:
                idx2 = random.choice([i for i in range(N) if i != idx1])

            parent2 = indivs[idx2]
            return [parent1, parent2]
        except Exception as e:
            print(e)
            return []
