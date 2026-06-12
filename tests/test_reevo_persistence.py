from __future__ import annotations

import concurrent.futures
import json
import tempfile
import unittest
from pathlib import Path

from llm4ad.base import Function
from llm4ad.method.reevo.population import Population
from llm4ad.method.reevo.profiler import ReEvoProfiler
from llm4ad.method.reevo.reevo import ReEvo


class _Sampler:
    def draw_sample(self, prompt):
        return "    return archive[0][0].copy()"


class _Evaluator:
    def __init__(self, score):
        self.score = score

    def evaluate_program_record_time(self, program):
        return self.score, 0.01


def _method(log_dir: str, score):
    profiler = ReEvoProfiler(log_dir=log_dir, final_log_dir=log_dir)
    profiler._samples_json_dir = str(Path(log_dir) / "samples")
    Path(profiler._samples_json_dir).mkdir(parents=True)

    method = ReEvo.__new__(ReEvo)
    method._sampler = _Sampler()
    method._template_program = (
        "def select_neighbor(archive, instance, distance_matrix_1, distance_matrix_2):\n"
        "    return archive[0][0].copy()\n"
    )
    method._evaluation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    method._evaluator = _Evaluator(score)
    method._profiler = profiler
    method._population = Population(pop_size=1)
    method._tot_sample_nums = 0
    return method


class ReEvoPersistenceTests(unittest.TestCase):
    def test_valid_sample_is_saved_and_checkpointed_after_population_update(self):
        with tempfile.TemporaryDirectory() as log_dir:
            method = _method(log_dir, -0.5)
            try:
                method._sample_evaluate_register("prompt")
            finally:
                method._evaluation_executor.shutdown()

            records = json.loads(
                (Path(log_dir) / "samples" / "samples_1~200.json").read_text(encoding="utf-8")
            )
            checkpoint = json.loads(
                (Path(log_dir) / "population" / "pop_1.json").read_text(encoding="utf-8")
            )

            self.assertEqual(method._tot_sample_nums, 1)
            self.assertEqual(method._population.generation, 1)
            self.assertEqual(records[0]["algorithm"], "{Generated heuristic.}")
            self.assertEqual(len(checkpoint), 1)

    def test_failed_evaluation_is_saved_and_consumes_budget(self):
        with tempfile.TemporaryDirectory() as log_dir:
            method = _method(log_dir, None)
            try:
                method._sample_evaluate_register("prompt")
            finally:
                method._evaluation_executor.shutdown()

            records = json.loads(
                (Path(log_dir) / "samples" / "samples_1~200.json").read_text(encoding="utf-8")
            )

            self.assertEqual(method._tot_sample_nums, 1)
            self.assertEqual(records[0]["score"], None)
            self.assertEqual(method._population.generation, 0)
            self.assertEqual(list((Path(log_dir) / "population").glob("pop_*.json")), [])

    def test_initialization_stops_when_sample_budget_is_exhausted(self):
        method = ReEvo.__new__(ReEvo)
        method._population = Population(pop_size=10)
        method._tot_sample_nums = 0
        method._max_sample_nums = 3
        method._task_description_str = "task"
        method._function_to_evolve = Function("heuristic", "x", "    return x")
        method._debug_mode = False

        def consume_sample(prompt):
            method._tot_sample_nums += 1

        method._sample_evaluate_register = consume_sample
        method._iteratively_init_population()

        self.assertEqual(method._tot_sample_nums, 3)


if __name__ == "__main__":
    unittest.main()
