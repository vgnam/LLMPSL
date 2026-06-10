import unittest
import tempfile

import numpy as np

from llm4ad.task.optimization.bi_tsp_semo.evaluation import BITSPEvaluation
from llm4ad.task.optimization.bi_tsp_semo.evaluation import check_constraint as check_bi_tsp
from llm4ad.task.optimization.tri_tsp_semo.evaluation import check_constraint as check_tri_tsp


class TSPConstraintTest(unittest.TestCase):
    def test_bi_tsp_default_search_budget_remains_2000(self):
        with tempfile.TemporaryDirectory() as data_dir:
            evaluator = BITSPEvaluation(
                n_instance=1,
                problem_size=4,
                seed=2025,
                data_dir=data_dir,
            )

        self.assertEqual(evaluator.search_iterations, 2000)

    def test_accepts_integer_permutation(self):
        solution = np.array([2, 0, 3, 1], dtype=np.int64)

        self.assertTrue(check_bi_tsp(solution, 4))
        self.assertTrue(check_tri_tsp(solution, 4))

    def test_rejects_coordinate_values_used_as_node_ids(self):
        solution = np.array([0, 1, 2, 0.75])

        self.assertFalse(check_bi_tsp(solution, 4))
        self.assertFalse(check_tri_tsp(solution, 4))

    def test_rejects_float_permutation_and_duplicate_nodes(self):
        float_solution = np.array([0.0, 1.0, 2.0, 3.0])
        duplicate_solution = np.array([0, 1, 1, 3])

        self.assertFalse(check_bi_tsp(float_solution, 4))
        self.assertFalse(check_tri_tsp(float_solution, 4))
        self.assertFalse(check_bi_tsp(duplicate_solution, 4))
        self.assertFalse(check_tri_tsp(duplicate_solution, 4))


if __name__ == "__main__":
    unittest.main()
