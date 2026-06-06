from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

from llm4ad.method.PBCLLM.eoh import (
    DEFAULT_EVALUATION_SEEDS,
    PBCLLM,
    _evaluate_program_with_seed,
    _merge_seeded_mo_results,
    _normalize_evaluation_seeds,
)


class _FakeRawEvaluator:
    def __init__(self):
        self.eval_seed = None
        self.return_mo_trace = False


class _FakeSecureEvaluator:
    def __init__(self):
        self._evaluator = _FakeRawEvaluator()

    def evaluate_program_record_time(self, program):
        return {
            "seed": self._evaluator.eval_seed,
            "trace": self._evaluator.return_mo_trace,
            "program": program,
        }, 1.5


class _FakeMOSecureEvaluator:
    def __init__(self):
        self._evaluator = _FakeRawEvaluator()

    def evaluate_program_record_time(self, program):
        seed = self._evaluator.eval_seed
        return _result(float(seed), [-float(seed), float(seed)]), 2.0


def _result(value: float, legacy_score):
    return {
        "objective_num": 2,
        "fronts": [[[value, value + 1.0]]],
        "archive_trajectories": [[[[value, value + 1.0]]]],
        "legacy_score": legacy_score,
    }


class PBCLLMMultiSeedTests(unittest.TestCase):
    def test_default_uses_one_common_seed(self):
        self.assertEqual(DEFAULT_EVALUATION_SEEDS, (2025,))

    def test_seed_normalization_deduplicates_while_preserving_order(self):
        self.assertEqual(_normalize_evaluation_seeds([7, 3, 7]), (7, 3))

    def test_seeded_evaluation_does_not_mutate_original_evaluator(self):
        evaluator = _FakeSecureEvaluator()

        result, eval_time = _evaluate_program_with_seed(evaluator, "program", 17)

        self.assertEqual(result["seed"], 17)
        self.assertTrue(result["trace"])
        self.assertEqual(eval_time, 1.5)
        self.assertIsNone(evaluator._evaluator.eval_seed)
        self.assertFalse(evaluator._evaluator.return_mo_trace)

    def test_merge_treats_each_seed_instance_as_an_evaluation_case(self):
        merged = _merge_seeded_mo_results(
            [_result(1.0, [-0.2, 2.0]), _result(3.0, [-0.6, 4.0])],
            [2025, 2026],
        )

        self.assertEqual(merged["evaluation_seeds"], [2025, 2026])
        self.assertEqual(merged["instances_per_seed"], [1, 1])
        self.assertEqual(len(merged["fronts"]), 2)
        self.assertEqual(len(merged["archive_trajectories"]), 2)
        self.assertEqual(merged["legacy_score"], [-0.4, 3.0])

    def test_merge_rejects_missing_seed_result(self):
        self.assertIsNone(_merge_seeded_mo_results([_result(1.0, [-0.2, 2.0])], [1, 2]))

    def test_merge_rejects_different_instance_counts_across_seeds(self):
        second = _result(2.0, [-0.4, 3.0])
        second["fronts"].append([[4.0, 5.0]])
        second["archive_trajectories"].append([[[4.0, 5.0]]])

        self.assertIsNone(
            _merge_seeded_mo_results(
                [_result(1.0, [-0.2, 2.0]), second],
                [1, 2],
            )
        )

    def test_pbcllm_evaluates_program_on_every_common_seed(self):
        method = PBCLLM.__new__(PBCLLM)
        method._evaluation_seeds = (11, 13)
        method._evaluator = _FakeMOSecureEvaluator()
        with ThreadPoolExecutor(max_workers=2) as executor:
            method._evaluation_executor = executor
            merged, eval_time = method._evaluate_program_across_seeds("program")

        self.assertEqual(merged["evaluation_seeds"], [11, 13])
        self.assertEqual(merged["instances_per_seed"], [1, 1])
        self.assertEqual(merged["legacy_score"], [-12.0, 12.0])
        self.assertEqual(eval_time, 4.0)


if __name__ == "__main__":
    unittest.main()
