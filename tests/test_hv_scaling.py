from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pymoo.indicators.hv import HV

from llm4ad.task.optimization.bi_kp.evaluation import BIKPEvaluation, fixed_ideal_point
from llm4ad.task.optimization.bi_cvrp.evaluation import BICVRPEvaluation
from llm4ad.task.optimization.hv_utils import (
    DEFAULT_SEARCH_ITERATIONS,
    reference_hypervolume_scale,
    scale_hypervolume,
)
from llm4ad.task.optimization.tri_tsp_semo.evaluation import TRITSPEvaluation
from llm4ad.tools.evaluate_population_front import (
    _additive_epsilon_to_ideal,
    evaluate_population_front_all_sizes,
    _population_front_summary,
    _spacing,
)


class HypervolumeScalingTests(unittest.TestCase):
    def test_ideal_aware_scale_uses_bounding_box_volume(self):
        ideal = np.array([-20.0, -20.0])
        ref = np.array([-2.0, -3.0])

        self.assertEqual(reference_hypervolume_scale(ref, ideal), 18.0 * 17.0)

    def test_ideal_aware_hv_is_bounded_for_points_inside_box(self):
        ideal = np.array([-20.0, -20.0])
        ref = np.array([-2.0, -3.0])
        front = np.array([[-15.0, -10.0], [-10.0, -15.0]])

        normalized_hv = scale_hypervolume(HV(ref_point=ref)(front), ref, ideal)

        self.assertGreaterEqual(normalized_hv, 0.0)
        self.assertLessEqual(normalized_hv, 1.0)

    def test_bi_kp_uses_run_independent_fixed_ideal_point(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = BIKPEvaluation(
                problem_size=20,
                n_instance=1,
                seed=2025,
                data_dir=first_dir,
            )
            second = BIKPEvaluation(
                problem_size=20,
                n_instance=2,
                seed=999,
                data_dir=second_dir,
            )

        np.testing.assert_array_equal(first.ideal_point, [-11.0, -11.0])
        np.testing.assert_array_equal(second.ideal_point, first.ideal_point)
        np.testing.assert_array_equal(first.ideal_point, fixed_ideal_point(20))
        self.assertEqual(first.hv_normalization_policy, "fixed_by_problem_size")
        self.assertEqual(first.hv_fixed_ideal_magnitude_by_size[20], 11.0)
        np.testing.assert_array_equal(first.normalization_ideal, first.ideal_point)
        np.testing.assert_array_equal(first.normalization_nadir, first.ref_point)

    def test_all_non_bi_tsp_evaluators_use_paper_search_budget(self):
        with tempfile.TemporaryDirectory() as tri_dir, tempfile.TemporaryDirectory() as cvrp_dir, tempfile.TemporaryDirectory() as kp_dir:
            evaluators = [
                TRITSPEvaluation(n_instance=1, problem_size=4, seed=2025, data_dir=tri_dir),
                BICVRPEvaluation(n_instance=1, problem_size=20, seed=2025, data_dir=cvrp_dir),
                BIKPEvaluation(n_instance=1, problem_size=20, seed=2025, data_dir=kp_dir),
            ]

        self.assertEqual(DEFAULT_SEARCH_ITERATIONS, 2000)
        self.assertTrue(all(evaluator.search_iterations == 2000 for evaluator in evaluators))

    def test_additive_epsilon_to_ideal_uses_balanced_best_point(self):
        ideal = np.array([0.0, 0.0])
        nadir = np.array([10.0, 10.0])
        front = np.array([[2.0, 8.0], [5.0, 5.0], [8.0, 2.0]])

        self.assertAlmostEqual(_additive_epsilon_to_ideal(front, ideal, nadir), 0.5)

    def test_spacing_is_zero_for_evenly_spaced_front(self):
        ideal = np.array([0.0, 0.0])
        nadir = np.array([10.0, 10.0])
        front = np.array([[2.0, 8.0], [5.0, 5.0], [8.0, 2.0]])

        self.assertAlmostEqual(_spacing(front, ideal, nadir), 0.0)

    def test_spacing_detects_uneven_front(self):
        ideal = np.array([0.0, 0.0])
        nadir = np.array([10.0, 10.0])
        front = np.array([[2.0, 8.0], [2.1, 7.9], [8.0, 2.0]])

        self.assertGreater(_spacing(front, ideal, nadir), 0.0)

    def test_population_summary_reports_metric_mean_std_and_instances(self):
        evaluator = SimpleNamespace(
            ref_point=np.array([10.0, 10.0]),
            normalization_ideal=np.array([0.0, 0.0]),
            normalization_nadir=np.array([10.0, 10.0]),
        )
        traces = [
            {
                "status": "ok",
                "fronts": [
                    [[2.0, 8.0], [5.0, 5.0], [8.0, 2.0]],
                    [[2.0, 8.0], [2.1, 7.9], [8.0, 2.0]],
                ],
                "wall_time": 1.0,
            }
        ]

        summary = _population_front_summary(traces, evaluator, 2)

        self.assertEqual(len(summary["instance_additive_epsilon"]), 2)
        self.assertEqual(len(summary["instance_spacing"]), 2)
        self.assertAlmostEqual(summary["mean_additive_epsilon"], 0.645)
        self.assertGreater(summary["mean_spacing"], 0.0)
        self.assertGreater(summary["spacing_std"], 0.0)

    def test_population_front_evaluation_accepts_multiple_problem_sizes(self):
        size_configs = [
            {"problem_size": 20, "n_instance": 10},
            {"problem_size": 20, "n_instance": 20},
            {"problem_size": 50, "n_instance": 20},
            {"problem_size": 100, "n_instance": 20},
            {"problem_size": 200, "n_instance": 20},
        ]
        with (
            patch("llm4ad.tools.evaluate_population_front.load_final_records", return_value=[]),
            patch("llm4ad.tools.evaluate_population_front.discover_size_configs", return_value=size_configs),
            patch("llm4ad.tools.evaluate_population_front.build_problem") as build_problem,
        ):
            report = evaluate_population_front_all_sizes(
                Path("unused"),
                method="test",
                problem="bi_tsp",
                problem_sizes=[20, 50, 100],
            )

        self.assertEqual(
            [(item["problem_size"], item["n_instance"]) for item in report["sizes"]],
            [(20, 10), (20, 20), (50, 20), (100, 20)],
        )
        self.assertEqual(build_problem.call_count, 4)

    def test_population_front_evaluation_rejects_both_size_filters(self):
        with self.assertRaisesRegex(ValueError, "either problem_size or problem_sizes"):
            evaluate_population_front_all_sizes(
                Path("unused"),
                method="test",
                problem="bi_tsp",
                problem_size=20,
                problem_sizes=[20, 50],
            )


if __name__ == "__main__":
    unittest.main()
