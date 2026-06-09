from __future__ import annotations

import concurrent.futures
import copy
import json
import random
import time
import traceback
from threading import Lock
from typing import Sequence

import numpy as np

from .population import Population, analyze_mo_result, default_preference_vectors
from .prompt import PBCPrompt as EoHPrompt
from .profiler import PBCProfiler
from ..LLMPFG.sampler import EoHSampler
from ...base import Evaluation, LLM, TextFunctionProgramConverter, SecureEvaluator
from ...tools.profiler import ProfilerBase


DEFAULT_EVALUATION_SEEDS = (2025,)
_DIRECT_SEEDED_EVALUATION_LOCK = Lock()


def _normalize_evaluation_seeds(evaluation_seeds: Sequence[int] | None) -> tuple[int, ...]:
    seeds = DEFAULT_EVALUATION_SEEDS if evaluation_seeds is None else evaluation_seeds
    normalized = []
    for seed in seeds:
        if isinstance(seed, bool):
            raise ValueError("PBCLLM evaluation seeds must be integers, not booleans.")
        try:
            value = int(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid PBCLLM evaluation seed: {seed!r}") from exc
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("PBCLLM requires at least one fixed evaluation seed.")
    return tuple(normalized)


def _evaluate_program_with_seed(
    evaluator: SecureEvaluator,
    program,
    seed: int,
    isolate_evaluator: bool = True,
):
    """Evaluate with an isolated evaluator so concurrent seeds cannot interfere."""
    seeded_evaluator = copy.deepcopy(evaluator) if isolate_evaluator else evaluator
    raw_evaluator = getattr(seeded_evaluator, "_evaluator", None)
    if raw_evaluator is None or not hasattr(raw_evaluator, "eval_seed"):
        raise ValueError("PBCLLM seeded evaluation requires an evaluator with an eval_seed attribute.")
    raw_evaluator.eval_seed = int(seed)
    raw_evaluator.return_mo_trace = True
    if not getattr(raw_evaluator, "safe_evaluate", True):
        with _DIRECT_SEEDED_EVALUATION_LOCK:
            return seeded_evaluator.evaluate_program_record_time(program)
    return seeded_evaluator.evaluate_program_record_time(program)


def _merge_seeded_mo_results(results: Sequence[dict], evaluation_seeds: Sequence[int]) -> dict | None:
    if not results or len(results) != len(evaluation_seeds):
        return None

    objective_num = None
    merged_fronts = []
    merged_trajectories = []
    instances_per_seed = []
    legacy_scores = []

    for result in results:
        if not isinstance(result, dict):
            return None
        current_objective_num = int(result.get("objective_num", 0) or 0)
        fronts = result.get("fronts")
        trajectories = result.get("archive_trajectories")
        if (
            current_objective_num < 2
            or not isinstance(fronts, list)
            or not isinstance(trajectories, list)
            or len(fronts) != len(trajectories)
            or not fronts
        ):
            return None
        if objective_num is None:
            objective_num = current_objective_num
        elif objective_num != current_objective_num:
            return None
        if instances_per_seed and len(fronts) != instances_per_seed[0]:
            return None

        merged_fronts.extend(fronts)
        merged_trajectories.extend(trajectories)
        instances_per_seed.append(len(fronts))

        legacy_score = result.get("legacy_score")
        if isinstance(legacy_score, (list, tuple)):
            try:
                score = np.asarray(legacy_score, dtype=float)
            except (TypeError, ValueError):
                score = np.empty(0, dtype=float)
            if score.ndim == 1 and score.size and np.all(np.isfinite(score)):
                legacy_scores.append(score)

    merged = {
        "objective_num": objective_num,
        "fronts": merged_fronts,
        "archive_trajectories": merged_trajectories,
        "evaluation_seeds": [int(seed) for seed in evaluation_seeds],
        "instances_per_seed": instances_per_seed,
    }
    if len(legacy_scores) == len(results) and len({score.shape for score in legacy_scores}) == 1:
        merged["legacy_score"] = np.mean(np.vstack(legacy_scores), axis=0).tolist()
    return merged


def _evaluator_normalization_bounds(evaluator, objective_num: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    ideal = getattr(evaluator, "normalization_ideal", None)
    nadir = getattr(evaluator, "normalization_nadir", None)
    if ideal is not None and nadir is not None:
        ideal = np.asarray(ideal, dtype=float)
        nadir = np.asarray(nadir, dtype=float)
        if ideal.shape == (objective_num,) and nadir.shape == (objective_num,):
            return ideal, nadir

    ref_point = getattr(evaluator, "ref_point", None)
    if ref_point is None:
        return None, None
    ref_point = np.asarray(ref_point, dtype=float)
    if ref_point.shape != (objective_num,) or not np.all(np.isfinite(ref_point)):
        return None, None
    if not np.all(ref_point > 0):
        return None, None
    return np.zeros(objective_num, dtype=float), ref_point


def _evaluator_hv_ref_point(evaluator, objective_num: int) -> np.ndarray | None:
    ref_point = getattr(evaluator, "ref_point", None)
    if ref_point is None:
        return None
    ref_point = np.asarray(ref_point, dtype=float)
    if ref_point.shape != (objective_num,) or not np.all(np.isfinite(ref_point)):
        return None
    return ref_point


class PBCLLM:
    """Pareto Behavior Coevolution for LLM-generated MOCO heuristics.

    PBCLLM is MO-only: the evaluator must return per-instance Pareto archive
    trajectories. Search selects heuristics by population-level HV contribution
    and Pareto behavior diversity. Preference vectors only diagnose weakly
    covered regions when selecting parents.
    """

    def __init__(
        self,
        llm: LLM,
        llm_cluster: LLM,
        evaluation: Evaluation,
        profiler: ProfilerBase | None = None,
        max_sample_nums: int | None = 80,
        pop_size: int = 12,
        selection_num: int = 3,
        use_e1_operator: bool = True,
        use_e2_operator: bool = True,
        use_m1_operator: bool = True,
        use_m2_operator: bool = True,
        num_samplers: int = 1,
        num_evaluators: int = 1,
        *,
        evaluation_seeds: Sequence[int] | None = None,
        behavior_novelty_weight: float = 0.2,
        cluster_distance: float = 0.35,
        elites_per_preference: int = 2,
        parent_selection_strategy: str = "complementary_behavior",
        rho: float = 0.05,
        debug_mode: bool = False,
        llm_review: bool = False,
        multi_thread_or_process_eval: str = "process",
        max_consecutive_sampling_errors: int = 5,
        sampling_error_backoff_seconds: float = 1.0,
        **kwargs,
    ):
        raw_eval = getattr(evaluation, "_evaluator", evaluation)
        objective_num = getattr(raw_eval, "objective_num", None)
        if objective_num is None:
            labels = getattr(raw_eval, "objective_labels", None)
            objective_num = len(labels) if labels else None
        if objective_num is None or int(objective_num) < 2:
            raise ValueError("PBCLLM requires objective_num >= 2 and cannot run on single-objective tasks.")
        if not hasattr(raw_eval, "eval_seed"):
            raise ValueError("PBCLLM requires an evaluator with an eval_seed attribute.")

        setattr(raw_eval, "return_mo_trace", True)

        self._objective_num = int(objective_num)
        self._evaluation_seeds = _normalize_evaluation_seeds(evaluation_seeds)
        self._preference_vectors = default_preference_vectors(self._objective_num)
        self._normalization_ideal, self._normalization_nadir = _evaluator_normalization_bounds(
            raw_eval,
            self._objective_num,
        )
        self._hv_ref_point = _evaluator_hv_ref_point(raw_eval, self._objective_num)
        self._rho = float(rho)
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._function_to_evolve = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._template_program = TextFunctionProgramConverter.text_to_program(self._template_program_str)
        self._max_sample_nums = max_sample_nums
        self._pop_size = int(pop_size)
        self._selection_num = int(selection_num)
        self._use_e1_operator = bool(use_e1_operator)
        self._use_e2_operator = bool(use_e2_operator)
        self._use_m1_operator = bool(use_m1_operator)
        self._use_m2_operator = bool(use_m2_operator)
        self._parent_selection_strategy = str(parent_selection_strategy)
        self._num_samplers = int(num_samplers)
        self._num_evaluators = int(num_evaluators)
        self._isolate_seed_evaluator = multi_thread_or_process_eval == "thread"
        self._debug_mode = debug_mode
        self._llm_review = bool(llm_review)
        self._tot_sample_nums = 0
        self._max_consecutive_sampling_errors = int(max_consecutive_sampling_errors)
        self._sampling_error_backoff_seconds = float(sampling_error_backoff_seconds)
        if self._max_consecutive_sampling_errors < 1:
            raise ValueError("max_consecutive_sampling_errors must be at least 1.")
        if self._sampling_error_backoff_seconds < 0:
            raise ValueError("sampling_error_backoff_seconds cannot be negative.")

        llm.debug_mode = debug_mode
        self._sampler = EoHSampler(llm, self._template_program_str)
        self._cluster_sampler = EoHSampler(llm_cluster, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler
        self._population = Population(
            self._pop_size,
            self._preference_vectors,
            hv_ref_point=self._hv_ref_point,
            normalization_ideal=self._normalization_ideal,
            normalization_nadir=self._normalization_nadir,
            rho=self._rho,
            behavior_novelty_weight=behavior_novelty_weight,
            cluster_distance=cluster_distance,
            elites_per_preference=elites_per_preference,
            parent_selection_strategy=self._parent_selection_strategy,
        )

        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)

        assert multi_thread_or_process_eval in ["thread", "process"]
        executor_cls = (
            concurrent.futures.ThreadPoolExecutor
            if multi_thread_or_process_eval == "thread"
            else concurrent.futures.ProcessPoolExecutor
        )
        self._evaluation_executor = executor_cls(max_workers=num_evaluators)

    def _continue_loop(self):
        return self._max_sample_nums is None or self._tot_sample_nums < self._max_sample_nums

    def _evaluate_program_across_seeds(self, program):
        futures = [
            self._evaluation_executor.submit(
                _evaluate_program_with_seed,
                self._evaluator,
                program,
                seed,
                getattr(self, "_isolate_seed_evaluator", True),
            )
            for seed in self._evaluation_seeds
        ]
        results = []
        total_eval_time = 0.0
        all_valid = True
        for future in futures:
            result, eval_time = future.result()
            if not isinstance(result, dict):
                all_valid = False
            else:
                results.append(result)
            total_eval_time += float(eval_time or 0.0)
        if not all_valid:
            return None, total_eval_time
        return _merge_seeded_mo_results(results, self._evaluation_seeds), total_eval_time

    def _sample_evaluate_register(self, prompt: str):
        sample_start = time.time()
        thought, func = self._sampler.get_thought_and_function(prompt)
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            return

        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            return

        try:
            result, eval_time = self._evaluate_program_across_seeds(program)
        except Exception:
            if self._debug_mode:
                traceback.print_exc()
            return

        pbc = analyze_mo_result(
            result,
            self._preference_vectors,
            rho=self._rho,
            normalization_ideal=self._normalization_ideal,
            normalization_nadir=self._normalization_nadir,
            hv_ref_point=self._hv_ref_point,
        )
        if pbc is None:
            return
        expected_front_count = sum(result.get("instances_per_seed", []))
        if pbc.get("front_count") != expected_front_count:
            return

        func.score = [-pbc["individual_hv"], pbc["coverage_loss"]]
        func.pbc = pbc
        func.algorithm = thought
        func.evaluate_time = eval_time
        func.sample_time = sample_time

        self._tot_sample_nums += 1
        self._population.register_function(func)

        if self._profiler is not None:
            self._profiler.register_function(func)
            if isinstance(self._profiler, PBCProfiler):
                self._profiler.register_population(self._population)

    def _init_population(self):
        while len(self._population) < self._pop_size and self._continue_loop():
            prompt = EoHPrompt.get_prompt_i1(
                self._task_description_str,
                self._function_to_evolve,
            )
            self._sample_evaluate_register(prompt)

    @staticmethod
    def _parse_cluster_groups(group_response, parent_count: int) -> list[list[int]] | None:
        if isinstance(group_response, dict):
            group = group_response.get("Group")
        elif isinstance(group_response, str):
            try:
                start = group_response.find("{")
                end = group_response.rfind("}")
                payload = group_response[start:end + 1] if start >= 0 and end >= start else group_response
                group = json.loads(payload).get("Group")
            except Exception:
                return None
        else:
            group = group_response

        if not isinstance(group, list) or len(group) < 2:
            return None
        cleaned = []
        seen = []
        for subgroup in group:
            if not isinstance(subgroup, list):
                return None
            values = []
            for idx in subgroup:
                if not isinstance(idx, int) or idx < 0 or idx >= parent_count:
                    return None
                values.append(idx)
                seen.append(idx)
            if values:
                cleaned.append(values)
        if sorted(seen) != list(range(parent_count)) or len(set(seen)) != parent_count:
            return None
        return cleaned

    @classmethod
    def _select_clustered_parents(cls, group_response, parents):
        groups = cls._parse_cluster_groups(group_response, len(parents))
        if not groups:
            return random.sample(parents, min(2, len(parents)))

        group1 = random.choice(groups)
        idx1 = random.choice(group1)
        other_groups = [group for group in groups if idx1 not in group]
        if not other_groups:
            return random.sample(parents, min(2, len(parents)))
        idx2 = random.choice(random.choice(other_groups))
        return [parents[idx1], parents[idx2]]

    def _select_operator_parents(self, selection_num: int):
        if not self._population.population:
            return []
        target_preference = self._population.weakest_preference()
        parents = self._population.select_parents(target_preference, selection_num)
        return [parent for parent in parents if parent is not None]

    def _maybe_cluster_parents(self, parents):
        if len(parents) < 3:
            return parents
        try:
            prompt_cluster = EoHPrompt.get_prompt_cluster(
                self._task_description_str,
                parents,
                self._function_to_evolve,
            )
            group = self._cluster_sampler.get_thought(prompt_cluster)
            clustered = self._select_clustered_parents(group, parents)
            return clustered if clustered else parents
        except Exception:
            if self._debug_mode:
                traceback.print_exc()
            return parents

    def _make_e_prompt(self, operator: str):
        parents = self._select_operator_parents(self._selection_num)
        if not parents:
            return EoHPrompt.get_prompt_i1(self._task_description_str, self._function_to_evolve)
        parents = self._maybe_cluster_parents(parents)
        if self._llm_review:
            suggestion = EoHPrompt.get_prompt_suggestions_only(
                self._task_description_str,
                parents,
                self._function_to_evolve,
            )
        else:
            suggestion = None
        if operator == "e2":
            return EoHPrompt.get_prompt_e2(
                self._task_description_str,
                parents,
                self._function_to_evolve,
                suggestion,
            )
        return EoHPrompt.get_prompt_e1(
            self._task_description_str,
            parents,
            self._function_to_evolve,
            suggestion,
        )

    def _make_m_prompt(self, operator: str):
        parents = self._select_operator_parents(1)
        parent = parents[0] if parents else None
        if parent is None:
            return EoHPrompt.get_prompt_i1(self._task_description_str, self._function_to_evolve)
        if operator == "m2":
            return EoHPrompt.get_prompt_m2(
                self._task_description_str,
                parent,
                self._function_to_evolve,
            )
        return EoHPrompt.get_prompt_m1(
            self._task_description_str,
            parent,
            self._function_to_evolve,
        )

    def _evolve(self):
        consecutive_errors = 0
        while self._continue_loop():
            operators = []
            if self._use_e1_operator:
                operators.append(("e1", self._make_e_prompt))
            if self._use_e2_operator:
                operators.append(("e2", self._make_e_prompt))
            if self._use_m1_operator:
                operators.append(("m1", self._make_m_prompt))
            if self._use_m2_operator:
                operators.append(("m2", self._make_m_prompt))
            if not operators:
                operators.append(("i1", lambda _: EoHPrompt.get_prompt_i1(
                    self._task_description_str,
                    self._function_to_evolve,
                )))

            for operator_name, prompt_builder in operators:
                if not self._continue_loop():
                    break
                try:
                    prompt = prompt_builder(operator_name)
                    if self._debug_mode:
                        print(prompt)
                    self._sample_evaluate_register(prompt)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    consecutive_errors += 1
                    print(
                        "PBCLLM sampling error "
                        f"{consecutive_errors}/{self._max_consecutive_sampling_errors}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    if self._debug_mode:
                        traceback.print_exc()
                    if consecutive_errors >= self._max_consecutive_sampling_errors:
                        raise RuntimeError(
                            "PBCLLM stopped after consecutive sampling errors to avoid "
                            "an infinite retry loop."
                        ) from exc
                    if self._sampling_error_backoff_seconds:
                        time.sleep(
                            self._sampling_error_backoff_seconds
                            * min(2 ** (consecutive_errors - 1), 8)
                        )
                    continue
                consecutive_errors = 0

    def run(self):
        try:
            self._init_population()
            self._evolve()
        finally:
            try:
                self._evaluation_executor.shutdown(cancel_futures=True)
            except Exception:
                pass
        if self._profiler is not None:
            self._profiler.finish()
