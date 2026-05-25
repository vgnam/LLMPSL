from __future__ import annotations

import json
import re
import time
import traceback
import hashlib

from .population import Population
from .profiler import EoHProfiler
from .prompt import LLMPSLPrompt
from ..LLMPFG.eoh import MPaGE as _LLMPFGMPaGE
from ...base import Function, Program, TextFunctionProgramConverter


class LLMPSL(_LLMPFGMPaGE):
    """Preference-indexed Pareto set learning for LLM-driven heuristic search."""

    def __init__(
        self,
        *args,
        objective_num: int = 3,
        novelty_k: int = 2,
        novelty_threshold: float = 0.90,
        prefer_codebleu: bool = True,
        preference_grid=None,
        tchebycheff_rho: float = 0.05,
        tchebycheff_set_size: int = 3,
        smooth_set_scalarization: bool = True,
        smooth_mu: float = 0.05,
        use_psl_interpolation: bool = True,
        use_novelty_repair: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._objective_num = objective_num
        self._novelty_k = novelty_k
        self._novelty_threshold = novelty_threshold
        self._prefer_codebleu = prefer_codebleu
        self._preference_grid = preference_grid
        self._tchebycheff_rho = tchebycheff_rho
        self._tchebycheff_set_size = tchebycheff_set_size
        self._smooth_set_scalarization = smooth_set_scalarization
        self._smooth_mu = smooth_mu
        self._use_psl_interpolation = use_psl_interpolation
        self._use_novelty_repair = use_novelty_repair
        self._llm_attempt_count = 0
        self._accepted_sample_count = 0
        self._rejected_sample_count = 0

        # LLMPFG creates its population before adjusting pop_size. Replace it
        # with the final LLMPSL population and preference memory.
        self._population = Population(
            pop_size=self._pop_size,
            objective_num=self._objective_num,
            preference_grid=self._preference_grid,
            novelty_k=self._novelty_k,
            rho=self._tchebycheff_rho,
            prefer_codebleu=self._prefer_codebleu,
            set_size=self._tchebycheff_set_size,
            smooth_set_scalarization=self._smooth_set_scalarization,
            smooth_mu=self._smooth_mu,
        )

    def _logger(self):
        if self._profiler is not None and hasattr(self._profiler, "get_logger"):
            return self._profiler.get_logger()
        return None

    @staticmethod
    def _short_text(value, limit=300):
        if value is None:
            return ""
        text = " ".join(str(value).split())
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _function_summary(func: Function | None):
        if func is None:
            return "function=None"
        func_str = str(func).strip()
        digest = hashlib.sha1(func_str.encode("utf-8")).hexdigest()[:10]
        body_lines = len([line for line in (func.body or "").splitlines() if line.strip()])
        return (
            f"name={getattr(func, 'name', '<unknown>')} "
            f"body_lines={body_lines} chars={len(func_str)} sha1={digest}"
        )

    def _log_event(self, event, **fields):
        logger = self._logger()
        if logger is None:
            return
        parts = [f"{key}={value}" for key, value in fields.items()]
        logger.info("[LLMPSL:%s] %s", event, " ".join(parts))

    def _continue_loop(self):
        if self._max_generations is None and self._max_sample_nums is None:
            return True
        if self._max_generations is not None and self._population.generation >= self._max_generations:
            return False
        if self._max_sample_nums is not None and self._tot_sample_nums >= self._max_sample_nums:
            return False
        return True

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

    def _maybe_cluster_select(self, indivs):
        if len(indivs) < 3:
            return indivs
        prompt_cluster = LLMPSLPrompt.get_prompt_cluster(
            self._task_description_str,
            indivs,
            self._function_to_evolve,
        )
        group_response = self._cluster_sampler.get_thought(prompt_cluster)
        group = self._parse_cluster_response(group_response)
        selected = self._population.selection_cluster(group, indivs)
        return selected or indivs

    def _sample_repair_if_needed(self, func: Function, preference):
        if not self._use_novelty_repair:
            return func
        if not self._population.has_near_duplicate_function(func, threshold=self._novelty_threshold):
            return func

        prompt = LLMPSLPrompt.get_prompt_novelty_repair(
            self._task_description_str,
            func,
            self._function_to_evolve,
            preference,
        )
        self._log_event(
            "repair_requested",
            preference=self._short_text(preference, 120),
            candidate=self._function_summary(func),
        )
        try:
            thought, repaired = self._sampler.get_thought_and_function(prompt)
        except Exception as exc:
            self._log_event("repair_llm_error", error=self._short_text(exc, 240))
            return func
        if repaired is None:
            self._log_event(
                "repair_rejected",
                reason="parse_failed",
                response_preview=self._short_text(getattr(self._sampler, "last_response", None)),
            )
            return func
        repaired.algorithm = thought
        self._log_event("repair_accepted", repaired=self._function_summary(repaired))
        return repaired

    def _extend_score_with_novelty(self, func: Function, score):
        if score is None:
            return None
        if isinstance(score, tuple):
            score = list(score)
        elif not isinstance(score, list):
            score = [score]

        score = [float(x) for x in score]
        if len(score) >= self._objective_num:
            return score[: self._objective_num]

        novelty = self._population.compute_code_novelty(func)
        func.code_novelty = novelty
        score.append(-float(novelty))
        return score

    def _sample_evaluate_register(self, prompt, preference=None):
        """Generate, evaluate, add CodeBLEU novelty, and register one program."""
        self._llm_attempt_count += 1
        attempt = self._llm_attempt_count
        operator = self._infer_operator_name(prompt)
        self._log_event(
            "sample_start",
            attempt=attempt,
            operator=operator,
            preference=self._short_text(preference, 120),
            prompt_chars=len(str(prompt)),
        )
        sample_start = time.time()
        try:
            thought, func = self._sampler.get_thought_and_function(prompt)
        except Exception as exc:
            self._rejected_sample_count += 1
            self._log_event(
                "sample_rejected",
                attempt=attempt,
                operator=operator,
                reason="llm_call_error",
                error=self._short_text(exc, 300),
            )
            return
        sample_time = time.time() - sample_start
        if thought is None:
            self._rejected_sample_count += 1
            self._log_event(
                "sample_rejected",
                attempt=attempt,
                operator=operator,
                reason="missing_algorithm_block",
                sample_time=f"{sample_time:.3f}",
                response_preview=self._short_text(getattr(self._sampler, "last_response", None)),
            )
            return
        if func is None:
            self._rejected_sample_count += 1
            self._log_event(
                "sample_rejected",
                attempt=attempt,
                operator=operator,
                reason="function_parse_failed",
                sample_time=f"{sample_time:.3f}",
                response_preview=self._short_text(getattr(self._sampler, "last_response", None)),
            )
            return

        if self._population.has_duplicate_function(func):
            self._rejected_sample_count += 1
            self._log_event(
                "sample_rejected",
                attempt=attempt,
                operator=operator,
                reason="duplicate_function",
                sample_time=f"{sample_time:.3f}",
                candidate=self._function_summary(func),
            )
            return

        func.algorithm = thought
        func = self._sample_repair_if_needed(func, preference)

        program: Program | None = TextFunctionProgramConverter.function_to_program(
            func,
            self._template_program,
        )
        if program is None:
            self._rejected_sample_count += 1
            self._log_event(
                "sample_rejected",
                attempt=attempt,
                operator=operator,
                reason="program_conversion_failed",
                sample_time=f"{sample_time:.3f}",
                candidate=self._function_summary(func),
            )
            return

        try:
            score, eval_time = self._evaluation_executor.submit(
                self._evaluator.evaluate_program_record_time,
                program,
            ).result()
        except Exception as exc:
            self._rejected_sample_count += 1
            self._log_event(
                "sample_rejected",
                attempt=attempt,
                operator=operator,
                reason="evaluation_exception",
                sample_time=f"{sample_time:.3f}",
                candidate=self._function_summary(func),
                error=self._short_text(exc, 300),
            )
            return

        score = self._extend_score_with_novelty(func, score)
        if score is None:
            self._rejected_sample_count += 1
            self._log_event(
                "sample_rejected",
                attempt=attempt,
                operator=operator,
                reason="evaluation_returned_none",
                sample_time=f"{sample_time:.3f}",
                eval_time=f"{eval_time:.3f}" if eval_time is not None else None,
                candidate=self._function_summary(func),
            )
            return

        func.score = score
        func.evaluate_time = eval_time
        func.sample_time = sample_time

        if self._profiler is not None:
            self._profiler.register_function(func)
            if isinstance(self._profiler, EoHProfiler):
                self._profiler.register_population(self._population)

        self._tot_sample_nums += 1
        self._accepted_sample_count += 1
        self._log_event(
            "sample_accepted",
            attempt=attempt,
            operator=operator,
            sample_order=self._tot_sample_nums,
            sample_time=f"{sample_time:.3f}",
            eval_time=f"{eval_time:.3f}" if eval_time is not None else None,
            score=self._short_text(score, 160),
            candidate=self._function_summary(func),
            accepted=self._accepted_sample_count,
            rejected=self._rejected_sample_count,
        )
        self._population.register_function(func)

    @staticmethod
    def _infer_operator_name(prompt):
        prompt_str = str(prompt)
        markers = (
            ("interpolate", "interpolate"),
            ("extrapolate", "extrapolate"),
            ("E1", "e1"),
            ("E2", "e2"),
            ("M1", "m1"),
            ("M2", "m2"),
            ("initial", "i1"),
        )
        lower_prompt = prompt_str.lower()
        for marker, name in markers:
            if marker.lower() in lower_prompt:
                return name
        return "unknown"

    def _suggestions_for(self, indivs, preference):
        if not self.review:
            return None
        suggestion_prompt = LLMPSLPrompt.get_prompt_suggestions_only(
            self._task_description_str,
            indivs,
            self._function_to_evolve,
            preference,
        )
        return self._cluster_sampler.get_thought(suggestion_prompt)

    def _thread_do_evolutionary_operator(self):
        while self._continue_loop():
            try:
                preference = self._population.select_target_preference()
                indivs = self._population.selection(max(self._selection_num, 2), preference=preference)
                indivs = self._maybe_cluster_select(indivs)
                suggestions = self._suggestions_for(indivs, preference)

                if self._use_psl_interpolation and len(indivs) >= 2:
                    prompt = LLMPSLPrompt.get_prompt_interpolate(
                        self._task_description_str,
                        indivs,
                        self._function_to_evolve,
                        preference,
                    )
                else:
                    prompt = LLMPSLPrompt.get_prompt_e1(
                        self._task_description_str,
                        indivs,
                        self._function_to_evolve,
                        suggestions,
                        preference,
                    )

                if self._debug_mode:
                    print(prompt)
                    input()

                self._sample_evaluate_register(prompt, preference=preference)
                if not self._continue_loop():
                    break

                if self._use_e2_operator:
                    preference = self._population.select_target_preference()
                    indivs = self._population.selection(max(self._selection_num, 2), preference=preference)
                    indivs = self._maybe_cluster_select(indivs)
                    suggestions = self._suggestions_for(indivs, preference)
                    prompt = LLMPSLPrompt.get_prompt_e2(
                        self._task_description_str,
                        indivs,
                        self._function_to_evolve,
                        suggestions,
                        preference,
                    )

                    if self._debug_mode:
                        print(prompt)
                        input()

                    self._sample_evaluate_register(prompt, preference=preference)
                    if not self._continue_loop():
                        break

                if self._use_m1_operator:
                    preference = self._population.select_target_preference()
                    indiv = self._population.selection(1, preference=preference)
                    if indiv:
                        prompt = LLMPSLPrompt.get_prompt_m1(
                            self._task_description_str,
                            indiv,
                            self._function_to_evolve,
                            preference,
                        )

                        if self._debug_mode:
                            print(prompt)
                            input()

                        self._sample_evaluate_register(prompt, preference=preference)
                    if not self._continue_loop():
                        break

                if self._use_m2_operator:
                    preference = self._population.select_target_preference()
                    indiv = self._population.selection(1, preference=preference)
                    if indiv:
                        prompt = LLMPSLPrompt.get_prompt_m2(
                            self._task_description_str,
                            indiv,
                            self._function_to_evolve,
                            preference,
                        )

                        if self._debug_mode:
                            print(prompt)
                            input()

                        self._sample_evaluate_register(prompt, preference=preference)
                    if not self._continue_loop():
                        break
            except KeyboardInterrupt:
                break
            except Exception as exc:
                self._log_event("evolution_loop_error", error=self._short_text(exc, 300))
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

        try:
            self._evaluation_executor.shutdown(cancel_futures=True)
        except Exception:
            pass

    def _thread_init_population(self):
        while self._population.generation == 0 and self._continue_loop():
            try:
                preference = self._population.select_target_preference()
                prompt = LLMPSLPrompt.get_prompt_i1(
                    self._task_description_str,
                    self._function_to_evolve,
                    preference,
                )
                self._sample_evaluate_register(prompt, preference=preference)
                if self._tot_sample_nums > self._initial_sample_nums_max:
                    print(f"Warning: Initialization not accomplished in {self._initial_sample_nums_max} samples !!!")
                    break
            except Exception as exc:
                self._log_event("init_loop_error", error=self._short_text(exc, 300))
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue


# Compatibility alias for scripts that expect a method class named MPaGE.
MPaGE = LLMPSL
