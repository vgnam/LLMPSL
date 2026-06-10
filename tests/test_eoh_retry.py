from __future__ import annotations

import concurrent.futures
import math
import unittest

from llm4ad.base import Function, TextFunctionProgramConverter
from llm4ad.method.eoh.eoh import EoH
from llm4ad.method.eoh.population import Population
from llm4ad.tools.llm.llm_api_litellm import HttpsApiLiteLLM


class _SequenceSampler:
    def __init__(self, functions):
        self.functions = iter(functions)
        self.calls = 0

    def get_thought_and_function(self, prompt):
        self.calls += 1
        return f"thought {self.calls}", next(self.functions)


class _SequenceEvaluator:
    def __init__(self, scores):
        self.scores = iter(scores)

    def evaluate_program_record_time(self, program):
        return next(self.scores), 0.01


def _function(body: str, score=None) -> Function:
    return Function(name="heuristic", args="x", body=body, score=score)


class EoHRetryTests(unittest.TestCase):
    def test_failed_evaluation_samples_fresh_function_once(self):
        method = EoH.__new__(EoH)
        method._sampler = _SequenceSampler([
            _function("    return missing_name"),
            _function("    return x"),
        ])
        method._template_program = TextFunctionProgramConverter.text_to_program(
            "def heuristic(x):\n    return x\n"
        )
        method._evaluation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        method._evaluator = _SequenceEvaluator([None, -0.5])
        method._profiler = None
        method._population = Population(pop_size=10)
        method._max_evaluation_retries = 1
        method._tot_sample_nums = 0
        method._max_sample_nums = 10
        method._max_generations = None

        try:
            method._sample_evaluate_register("prompt")
        finally:
            method._evaluation_executor.shutdown()

        self.assertEqual(method._sampler.calls, 2)
        self.assertEqual(method._tot_sample_nums, 2)
        self.assertEqual(len(method._population._next_gen_pop), 1)
        self.assertEqual(method._population._next_gen_pop[0].score, -0.5)

    def test_population_rejects_invalid_and_duplicate_functions(self):
        population = Population(pop_size=10)
        valid = _function("    return x", score=-0.5)

        self.assertFalse(population.register_function(_function("    return x + 1", score=None)))
        self.assertFalse(population.register_function(_function("    return x + 2", score=float("-inf"))))
        self.assertTrue(population.register_function(valid))
        self.assertFalse(population.register_function(_function("    return x", score=-0.4)))
        self.assertEqual(len(population._next_gen_pop), 1)
        self.assertTrue(all(not math.isinf(func.score) for func in population._next_gen_pop))

    def test_litellm_default_temperature_is_point_seven(self):
        llm = HttpsApiLiteLLM(base_url="https://example.test", api_key="key", model="model")
        self.assertEqual(llm._temperature, 0.7)


if __name__ == "__main__":
    unittest.main()
