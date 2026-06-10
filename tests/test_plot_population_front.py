from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from llm4ad.tools.plot_population_front import _evaluate_log_front


class _FakeEvaluator:
    def __init__(self, eval_seed):
        self.eval_seed = eval_seed
        self.n_instance = 1
        self.problem_size = 100
        self.ref_point = np.array([100.0, 100.0])
        self.search_iterations = 2000
        self.return_mo_trace = False
        self.safe_evaluate = True


class PlotPopulationFrontTests(unittest.TestCase):
    def test_repeated_eval_seeds_are_merged_into_one_front(self):
        def build_problem(*args, **kwargs):
            return _FakeEvaluator(kwargs["eval_seed"])

        def evaluate_record(record, evaluator):
            value = float(evaluator.eval_seed)
            return {"status": "ok", "fronts": [[[value, 10.0 - value]]]}

        records = [{"function": "def f():\n    pass\n", "score": [0.0]}]
        with (
            patch("llm4ad.tools.plot_population_front.build_problem", side_effect=build_problem),
            patch("llm4ad.tools.plot_population_front.load_final_records", return_value=records),
            patch("llm4ad.tools.plot_population_front._evaluate_trace_record", side_effect=evaluate_record),
        ):
            front, _, summary = _evaluate_log_front(
                Path("run"),
                problem="bi_tsp",
                problem_size=100,
                n_instance=1,
                seed=2025,
                eval_seed=1,
                eval_repeats=3,
                search_iterations=10000,
                timeout_seconds=None,
                top_k=0,
            )

        self.assertEqual(len(front), 3)
        self.assertEqual(summary["eval_seeds"], [1, 2, 3])
        self.assertEqual(summary["eval_repeats"], 3)
        self.assertEqual(summary["num_valid"], 3)
        self.assertEqual(summary["search_iterations"], 10000)

    def test_rejects_non_positive_repeat_count(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            _evaluate_log_front(
                Path("run"),
                problem="bi_tsp",
                problem_size=100,
                n_instance=1,
                seed=2025,
                eval_seed=1,
                eval_repeats=0,
                search_iterations=None,
                timeout_seconds=None,
                top_k=0,
            )


if __name__ == "__main__":
    unittest.main()
