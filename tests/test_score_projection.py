from __future__ import annotations

import pickle
import multiprocessing
import unittest
from types import SimpleNamespace

from main import ScoreProjectionEvaluation


def _read_custom_attribute(evaluation, result_queue):
    result_queue.put(evaluation.custom_attribute)


def _base_evaluation():
    return SimpleNamespace(
        template_program="def heuristic(x):\n    return x\n",
        task_description="test",
        use_numba_accelerate=False,
        use_protected_div=False,
        protected_div_delta=1e-5,
        random_seed=None,
        timeout_seconds=60,
        exec_code=True,
        safe_evaluate=True,
        daemon_eval_process=False,
        custom_attribute="available after unpickle",
    )


class ScoreProjectionEvaluationTests(unittest.TestCase):
    def test_pickle_round_trip_does_not_recurse_during_spawn_restore(self):
        evaluation = ScoreProjectionEvaluation(_base_evaluation(), "negative_hv")

        restored = pickle.loads(pickle.dumps(evaluation))

        self.assertEqual(restored.score_projection, "negative_hv")
        self.assertEqual(restored.custom_attribute, "available after unpickle")

    def test_windows_spawn_can_restore_projection_wrapper(self):
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(
            target=_read_custom_attribute,
            args=(ScoreProjectionEvaluation(_base_evaluation(), "negative_hv"), result_queue),
        )

        process.start()
        process.join(timeout=15)

        self.assertEqual(process.exitcode, 0)
        self.assertEqual(result_queue.get(timeout=1), "available after unpickle")


if __name__ == "__main__":
    unittest.main()
