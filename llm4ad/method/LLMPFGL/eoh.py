from __future__ import annotations

import concurrent.futures
import json
import re
import time
import traceback
from threading import Thread

import numpy as np

EPS = 1e-12

from .population import (
    Population,
    extract_front_points,
    fit_front_gp,
    detect_surgical_gap,
    select_nearest_theta,
)
from .profiler import EoHProfiler
from .prompt import PFGLPrompt
from .sampler import EoHSampler
from ..LLMPFG.eoh import MPaGE as _MPaGE
from ...base import (
    Evaluation,
    LLM,
    Function,
    Program,
    TextFunctionProgramConverter,
    SecureEvaluator,
)
from ...tools.profiler import ProfilerBase


class LLMPFGL(_MPaGE):
    """LLM-driven Pareto Front Geometry Learning.

    This method is **strictly multi-objective**. It fits a Gaussian-Process
    manifold over the emerging Pareto front in normalised objective space,
    detects geometric gaps (high uncertainty × high curvature), and prompts the
    LLM to synthesise heuristics that densify those specific trade-off sectors.

    In addition to the PFGL gap-infill operator, the full suite of E1/E2/M1/M2
    evolutionary operators from LLMPFG is retained and interleaved, so that
    geometry-guided exploration is balanced with classical code-level variation.
    """

    def __init__(
        self,
        llm: LLM,
        llm_cluster: LLM,
        evaluation: Evaluation,
        profiler: ProfilerBase | None = None,
        max_generations: int | None = 10,
        max_sample_nums: int | None = 100,
        pop_size: int = 0,
        selection_num: int = 2,
        num_samplers: int = 1,
        num_evaluators: int = 1,
        *,
        gp_refit_interval: int = 3,
        n_theta_dense: int = 200,
        resume_mode: bool = False,
        initial_sample_num: int | None = None,
        initial_sample_nums_max: int = 50,
        debug_mode: bool = False,
        llm_review: bool = False,
        multi_thread_or_process_eval: str = "process",
        **kwargs,
    ):
        # --- strictly MO guard ------------------------------------------------
        raw_eval = getattr(evaluation, "_evaluator", evaluation)
        obj_num = getattr(raw_eval, "objective_num", None)
        if obj_num is None:
            labels = getattr(raw_eval, "objective_labels", None)
            obj_num = len(labels) if labels else None
        if obj_num is None:
            # Infer from evaluator return: try once with a dummy score
            try:
                test_score = getattr(raw_eval, "evaluate_program", lambda _a, _b: (0, 0))(None, None)
                if isinstance(test_score, (tuple, list, np.ndarray)):
                    obj_num = len(test_score)
                else:
                    obj_num = 1
            except Exception:
                obj_num = 2  # default assumption for MO problems in this benchmark
        if obj_num is None or obj_num < 2:
            raise ValueError(
                "LLM-PFGL is a strictly multi-objective method. "
                f"Detected objective_num={obj_num}. Use LLMPFG for single-objective problems."
            )
        self._objective_num = int(obj_num)

        # --- base initialisation (reuse MPaGE wiring) -------------------------
        super().__init__(
            llm=llm,
            llm_cluster=llm_cluster,
            evaluation=evaluation,
            profiler=profiler,
            max_generations=max_generations,
            max_sample_nums=max_sample_nums,
            pop_size=pop_size,
            selection_num=selection_num,
            use_e1_operator=True,
            use_e2_operator=True,
            use_m1_operator=True,
            use_m2_operator=True,
            num_samplers=num_samplers,
            num_evaluators=num_evaluators,
            resume_mode=resume_mode,
            initial_sample_num=initial_sample_num,
            initial_sample_nums_max=initial_sample_nums_max,
            debug_mode=debug_mode,
            llm_review=llm_review,
            multi_thread_or_process_eval=multi_thread_or_process_eval,
            **kwargs,
        )

        # Replace population created by MPaGE with PFGL-aware population
        self._population = Population(pop_size=self._pop_size)

        # PFGL-specific state
        self._gp_refit_interval = int(gp_refit_interval)
        self._n_theta_dense = int(n_theta_dense)
        self._gp = None
        self._theta_dense = np.linspace(-np.pi / 2, np.pi / 2, self._n_theta_dense)

    # -----------------------------------------------------------------------
    # Override sampling / evolution loop
    # -----------------------------------------------------------------------

    def _thread_init_population(self):
        """Initialise by random sampling (no geometry yet)."""
        while self._population.generation == 0 and self._continue_loop():
            try:
                prompt = PFGLPrompt.get_prompt_i1(
                    self._task_description_str,
                    self._function_to_evolve,
                    geometry_context=None,
                )
                self._sample_evaluate_register(prompt)
                if self._tot_sample_nums > self._initial_sample_nums_max:
                    print(
                        f"Warning: Initialization not accomplished in "
                        f"{self._initial_sample_nums_max} samples !!!"
                    )
                    break
            except Exception as exc:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

    def _geometry_context(self):
        """Return current gap geometry + parent info, or None if GP not ready."""
        theta_obs, r_obs, nd_scores, ideal, nadir = extract_front_points(
            self._population.population
        )

        if len(theta_obs) >= 3:
            if (
                self._gp is None
                or self._population.generation % self._gp_refit_interval == 0
            ):
                self._gp = fit_front_gp(theta_obs, r_obs)

        theta_gap, mu_gap, sigma_gap, curv_gap = detect_surgical_gap(
            self._gp, self._theta_dense, theta_obs
        )

        if theta_gap is None:
            return None

        theta_deg = float(np.degrees(theta_gap))
        target_norm = np.array(
            [np.cos(theta_gap), np.sin(theta_gap)], dtype=float
        )
        target_norm = target_norm / (np.linalg.norm(target_norm) + EPS)
        target_obj = tuple(
            float(ideal[i] + target_norm[i] * (nadir[i] - ideal[i]))
            for i in range(2)
        )

        if curv_gap > 2.0:
            shape_desc = (
                "SHARPLY CURVED — small changes in trade-off angle cause "
                "large objective shifts"
            )
        elif curv_gap > 0.5:
            shape_desc = "moderately curved"
        else:
            shape_desc = "relatively flat"

        rel_sigma = sigma_gap / (mu_gap + EPS)
        conf_desc = (
            "HIGH UNCERTAINTY — this region is underexplored"
            if rel_sigma > 0.15
            else "moderate confidence"
        )

        parent_a = select_nearest_theta(
            self._population.population, theta_gap - 0.12
        )
        parent_b = select_nearest_theta(
            self._population.population, theta_gap + 0.12
        )

        return {
            "theta_gap_deg": theta_deg,
            "target_obj": target_obj,
            "shape": shape_desc,
            "confidence": conf_desc,
            "parent_a": parent_a,
            "parent_b": parent_b,
            "theta_gap": theta_gap,
        }

    def _maybe_cluster_select(self, indivs):
        if len(indivs) < 3:
            return indivs
        prompt_cluster = PFGLPrompt.get_prompt_cluster(
            self._task_description_str, indivs, self._function_to_evolve
        )
        group_response = self._cluster_sampler.get_thought(prompt_cluster)
        group = self._parse_cluster_response(group_response)
        selected = self._population.selection_cluster(group, indivs)
        return selected if selected else indivs

    @staticmethod
    def _parse_cluster_response(response):
        if isinstance(response, list):
            return response
        if not isinstance(response, str):
            return None
        try:
            data = json.loads(response)
        except Exception:
            match = re.search(r"\{.*\}", response, flags=re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except Exception:
                return None
        if isinstance(data, dict):
            return data.get("Group")
        return None

    def _suggestions_for(self, indivs):
        if not self.review:
            return None
        suggestion_prompt = PFGLPrompt.get_prompt_suggestions_only(
            self._task_description_str, indivs, self._function_to_evolve
        )
        return self._cluster_sampler.get_thought(suggestion_prompt)

    def _thread_do_evolutionary_operator(self):
        """One while-loop iteration fires ALL enabled operators in sequence:

        G1 (PFGL gap infill) → E1 → E2 (optional) → M1 (optional) → M2 (optional).
        This matches the LLMPFG pattern where each generation step produces
        multiple offspring before the population is trimmed by NSGA-II.
        """
        while self._continue_loop():
            try:
                # Compute geometry once per iteration and inject into every prompt
                geo = self._geometry_context()

                # ---- G1: PFGL gap infill ------------------------------------
                if geo is None:
                    indivs = self._population.selection(max(self._selection_num, 1))
                    if len(indivs) >= 2:
                        prompt = PFGLPrompt.get_prompt_gap_infill(
                            self._task_description_str,
                            self._function_to_evolve,
                            theta_gap_deg=45.0,
                            target_obj=(0.5, 0.5),
                            confidence="low (exploration fallback)",
                            shape="unknown",
                            parent_a=indivs[0],
                            parent_b=indivs[-1] if len(indivs) > 1 else None,
                        )
                    else:
                        prompt = PFGLPrompt.get_prompt_i1(
                            self._task_description_str,
                            self._function_to_evolve,
                            geometry_context=None,
                        )
                else:
                    prompt = PFGLPrompt.get_prompt_gap_infill(
                        self._task_description_str,
                        self._function_to_evolve,
                        theta_gap_deg=geo["theta_gap_deg"],
                        target_obj=geo["target_obj"],
                        confidence=geo["confidence"],
                        shape=geo["shape"],
                        parent_a=geo["parent_a"],
                        parent_b=geo["parent_b"],
                    )
                self._sample_evaluate_register(prompt)
                if not self._continue_loop():
                    break

                # ---- E1 ------------------------------------------------------
                indivs = self._population.selection(self._selection_num)
                if len(indivs) >= 3:
                    group_response = self._cluster_sampler.get_thought(
                        PFGLPrompt.get_prompt_cluster(
                            self._task_description_str, indivs, self._function_to_evolve
                        )
                    )
                    group = self._parse_cluster_response(group_response)
                    indivs = self._population.selection_cluster(group, indivs)
                suggestions = self._suggestions_for(indivs)
                prompt = PFGLPrompt.get_prompt_e1(
                    self._task_description_str,
                    indivs,
                    self._function_to_evolve,
                    suggestions,
                    geometry_context=geo,
                )
                self._sample_evaluate_register(prompt)
                if not self._continue_loop():
                    break

                # ---- E2 ------------------------------------------------------
                if self._use_e2_operator:
                    indivs = self._population.selection(self._selection_num)
                    if len(indivs) >= 3:
                        group_response = self._cluster_sampler.get_thought(
                            PFGLPrompt.get_prompt_cluster(
                                self._task_description_str, indivs, self._function_to_evolve
                            )
                        )
                        group = self._parse_cluster_response(group_response)
                        indivs = self._population.selection_cluster(group, indivs)
                    suggestions = self._suggestions_for(indivs)
                    prompt = PFGLPrompt.get_prompt_e2(
                        self._task_description_str,
                        indivs,
                        self._function_to_evolve,
                        suggestions,
                        geometry_context=geo,
                    )
                    self._sample_evaluate_register(prompt)
                    if not self._continue_loop():
                        break

                # ---- M1 ------------------------------------------------------
                if self._use_m1_operator:
                    indiv = self._population.selection(1)
                    if indiv:
                        prompt = PFGLPrompt.get_prompt_m1(
                            self._task_description_str,
                            indiv[0],
                            self._function_to_evolve,
                            geometry_context=geo,
                        )
                        self._sample_evaluate_register(prompt)
                    if not self._continue_loop():
                        break

                # ---- M2 ------------------------------------------------------
                if self._use_m2_operator:
                    indiv = self._population.selection(1)
                    if indiv:
                        prompt = PFGLPrompt.get_prompt_m2(
                            self._task_description_str,
                            indiv[0],
                            self._function_to_evolve,
                            geometry_context=geo,
                        )
                        self._sample_evaluate_register(prompt)
                    if not self._continue_loop():
                        break

            except KeyboardInterrupt:
                break
            except Exception as exc:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

        # shutdown executor
        try:
            self._evaluation_executor.shutdown(cancel_futures=True)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Sample → evaluate → register (reuse base logic exactly)
    # -----------------------------------------------------------------------

    def _sample_evaluate_register(self, prompt):
        sample_start = time.time()
        thought, func = self._sampler.get_thought_and_function(prompt)
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            return

        program = TextFunctionProgramConverter.function_to_program(
            func, self._template_program
        )
        if program is None:
            return

        try:
            score, eval_time = self._evaluation_executor.submit(
                self._evaluator.evaluate_program_record_time, program
            ).result()
        except Exception:
            return

        if score is None:
            return

        # Enforce multi-objective: score must be a sequence of length ≥ 2
        score_vec = list(score) if isinstance(score, (tuple, list, np.ndarray)) else [score]
        if len(score_vec) < self._objective_num:
            return

        func.score = score_vec[: self._objective_num]
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time

        if self._profiler is not None:
            self._profiler.register_function(func)
            if isinstance(self._profiler, EoHProfiler):
                self._profiler.register_population(self._population)

        self._tot_sample_nums += 1
        self._population.register_function(func)
