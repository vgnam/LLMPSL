from __future__ import annotations

import unittest

import numpy as np

from llm4ad.base import Function, TextFunctionProgramConverter
from llm4ad.method.LLMPFG.sampler import EoHSampler
from llm4ad.method.PBCLLM.eoh import PBCLLM


class _Sampler:
    def __init__(self, thought, func):
        self.thought = thought
        self.func = func

    def get_thought_and_function(self, prompt):
        return self.thought, self.func


class _Population:
    def __init__(self):
        self.functions = []

    def register_function(self, func):
        self.functions.append(func)


def _valid_result():
    return {
        "objective_num": 2,
        "fronts": [[[1.0, 2.0]]],
        "archive_trajectories": [[[[2.0, 3.0]], [[1.0, 2.0]]]],
        "legacy_score": [-0.5, 0.1],
        "evaluation_seeds": [2025],
        "instances_per_seed": [1],
    }


def _method(sampler) -> PBCLLM:
    method = PBCLLM.__new__(PBCLLM)
    method._sampler = sampler
    method._template_program = TextFunctionProgramConverter.text_to_program(
        "def heuristic(x):\n    return x\n"
    )
    method._evaluate_program_across_seeds = lambda program: (_valid_result(), 0.1)
    method._preference_vectors = np.array([[0.5, 0.5]])
    method._rho = 0.05
    method._normalization_ideal = np.array([0.0, 0.0])
    method._normalization_nadir = np.array([10.0, 10.0])
    method._hv_ref_point = np.array([10.0, 10.0])
    method._hv_ideal_point = np.array([0.0, 0.0])
    method._population = _Population()
    method._profiler = None
    method._debug_mode = False
    method._tot_sample_nums = 0
    method._llm_sample_calls = 0
    method._llm_sample_fail_parse = 0
    method._llm_sample_fail_eval = 0
    method._llm_sample_fail_pbc = 0
    method._llm_sample_success = 0
    method._llm_cluster_calls = 0
    method._llm_cluster_fail = 0
    return method


class PBCLLMSamplingTests(unittest.TestCase):
    def test_multiline_boxed_thought_is_accepted(self):
        response = "{First line\nsecond line}\ndef heuristic(x):\n    return x\n"

        self.assertEqual(
            EoHSampler.trim_thought_from_response(response),
            "{First line\nsecond line}",
        )

    def test_missing_boxed_thought_gets_fallback_description(self):
        response = "def heuristic(x):\n    return x\n"

        self.assertEqual(
            EoHSampler.trim_thought_from_response(response),
            "{Generated heuristic.}",
        )

    def test_valid_function_is_accepted_when_sampler_returns_no_thought(self):
        func = Function(name="heuristic", args="x", body="    return x")
        method = _method(_Sampler(None, func))

        accepted = method._sample_evaluate_register("prompt")

        self.assertTrue(accepted)
        self.assertEqual(method._llm_sample_success, 1)
        self.assertEqual(method._population.functions[0].algorithm, "{Generated heuristic.}")

    def test_parse_rejection_is_counted(self):
        method = _method(_Sampler("{bad}", None))

        accepted = method._sample_evaluate_register("prompt")

        self.assertFalse(accepted)
        self.assertEqual(method._llm_sample_fail_parse, 1)

    def test_continue_loop_only_checks_valid_sample_budget(self):
        method = PBCLLM.__new__(PBCLLM)
        method._max_sample_nums = 200
        method._tot_sample_nums = 199

        self.assertTrue(method._continue_loop())

        method._tot_sample_nums = 200
        self.assertFalse(method._continue_loop())


if __name__ == "__main__":
    unittest.main()
