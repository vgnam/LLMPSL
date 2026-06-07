from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from llm4ad.base import TextFunctionProgramConverter
from llm4ad.method.PBCLLM.population import Population, default_preference_vectors
from llm4ad.method.PBCLLM.profiler import PBCProfiler
from llm4ad.method.PBCLLM.resume import resume_pbcllm


def _old_record(index: int) -> dict:
    return {
        "algorithm": f"algorithm {index}",
        "function": f"def heuristic_{index}(x):\n    return x + {index}\n",
        "score": [-float(index), 0.5],
        "pbc": {
            "individual_hv": float(index),
            "coverage_loss": 0.5,
            "preference_performance": [0.5] * 5,
        },
    }


class _FakePBCLLM(SimpleNamespace):
    def _evaluate_program_across_seeds(self, program):
        self.evaluation_count += 1
        value = float(self.evaluation_count)
        return {
            "objective_num": 2,
            "fronts": [[[value, value + 1.0]]],
            "archive_trajectories": [[[[value + 1.0, value + 2.0]], [[value, value + 1.0]]]],
            "legacy_score": [-value, 0.5],
            "evaluation_seeds": [2025],
            "instances_per_seed": [1],
        }, 1.0


class PBCLLMResumeTests(unittest.TestCase):
    def _population(self) -> Population:
        return Population(
            2,
            default_preference_vectors(2),
            hv_ref_point=np.array([10.0, 10.0]),
            normalization_ideal=np.array([0.0, 0.0]),
            normalization_nadir=np.array([10.0, 10.0]),
        )

    def _method(self, log_dir: str) -> _FakePBCLLM:
        return _FakePBCLLM(
            _profiler=PBCProfiler(log_dir=log_dir, final_log_dir=log_dir),
            _population=self._population(),
            _objective_num=2,
            _evaluation_seeds=(2025,),
            _template_program=TextFunctionProgramConverter.text_to_program(
                "def heuristic(x):\n    return x\n"
            ),
            _preference_vectors=default_preference_vectors(2),
            _rho=0.05,
            _normalization_ideal=np.array([0.0, 0.0]),
            _normalization_nadir=np.array([10.0, 10.0]),
            _hv_ref_point=np.array([10.0, 10.0]),
            _max_sample_nums=5,
            _tot_sample_nums=0,
            evaluation_count=0,
        )

    def test_resume_rebuilds_initial_log_and_keeps_using_same_directory(self):
        with tempfile.TemporaryDirectory() as log_dir:
            os.makedirs(os.path.join(log_dir, "population"))
            os.makedirs(os.path.join(log_dir, "samples"))
            population_path = os.path.join(log_dir, "population", "pop_1.json")
            with open(population_path, "w", encoding="utf-8") as file:
                json.dump([_old_record(1), _old_record(2)], file)
            with open(os.path.join(log_dir, "samples", "samples_0~200.json"), "w", encoding="utf-8") as file:
                json.dump(
                    [
                        {"sample_order": index, **_old_record(index)}
                        for index in range(1, 4)
                    ],
                    file,
                )

            method = self._method(log_dir)
            resume_pbcllm(method)

            self.assertEqual(method._profiler._log_dir, log_dir)
            self.assertEqual(method._tot_sample_nums, 3)
            self.assertEqual(method._population.generation, 1)
            self.assertEqual(len(method._population._next_gen_pop), 1)
            self.assertEqual(method.evaluation_count, 3)
            with open(population_path, "r", encoding="utf-8") as file:
                upgraded = json.load(file)
            self.assertIn("individual_hv", upgraded[0]["pbc"])

    def test_compact_checkpoint_resumes_by_re_evaluating_pbc_state(self):
        with tempfile.TemporaryDirectory() as log_dir:
            os.makedirs(os.path.join(log_dir, "population"))
            os.makedirs(os.path.join(log_dir, "samples"))
            method = self._method(log_dir)
            records = []
            for index in range(1, 3):
                func = TextFunctionProgramConverter.text_to_function(_old_record(index)["function"])
                func.algorithm = f"algorithm {index}"
                func.pbc = {
                    "objective_num": 2,
                    "fronts": [[[1.0, 2.0]]],
                    "pbt": [[[0.5, 0.5, 1.0]]],
                    "individual_hv": 1.0,
                    "coverage_loss": 0.5,
                    "preference_performance": [0.5] * 5,
                    "front_count": 1,
                    "evaluation_seeds": [2025],
                    "instances_per_seed": [1],
                }
                func.score = [-1.0, 0.5]
                records.append(method._profiler._record_payload(func))
            with open(os.path.join(log_dir, "population", "pop_1.json"), "w", encoding="utf-8") as file:
                json.dump(records, file)
            with open(os.path.join(log_dir, "samples", "samples_0~200.json"), "w", encoding="utf-8") as file:
                json.dump(
                    [
                        {"sample_order": index, **record}
                        for index, record in enumerate(records, start=1)
                    ],
                    file,
                )

            resume_pbcllm(method)

            self.assertEqual(method.evaluation_count, 2)
            self.assertEqual(method._tot_sample_nums, 2)

    def test_legacy_full_checkpoint_resumes_without_re_evaluation(self):
        with tempfile.TemporaryDirectory() as log_dir:
            os.makedirs(os.path.join(log_dir, "population"))
            os.makedirs(os.path.join(log_dir, "samples"))
            method = self._method(log_dir)
            records = []
            for index in range(1, 3):
                record = _old_record(index)
                record["pbc"].update(
                    {
                        "objective_num": 2,
                        "fronts": [[[1.0, 2.0]]],
                        "pbt": [[[0.5, 0.5, 1.0]]],
                        "evaluation_seeds": [2025],
                        "instances_per_seed": [1],
                    }
                )
                records.append(record)
            with open(os.path.join(log_dir, "population", "pop_1.json"), "w", encoding="utf-8") as file:
                json.dump(records, file)
            with open(os.path.join(log_dir, "samples", "samples_0~200.json"), "w", encoding="utf-8") as file:
                json.dump(
                    [
                        {"sample_order": index, **record}
                        for index, record in enumerate(records, start=1)
                    ],
                    file,
                )

            resume_pbcllm(method)

            self.assertEqual(method.evaluation_count, 0)
            self.assertEqual(len(method._profiler._latest_pbc_records), 2)

    def test_resume_initial_log_before_first_population_checkpoint(self):
        with tempfile.TemporaryDirectory() as log_dir:
            os.makedirs(os.path.join(log_dir, "population"))
            os.makedirs(os.path.join(log_dir, "samples"))
            with open(os.path.join(log_dir, "samples", "samples_0~200.json"), "w", encoding="utf-8") as file:
                json.dump([{"sample_order": 1, **_old_record(1)}], file)

            method = self._method(log_dir)
            resume_pbcllm(method)

            self.assertEqual(method._population.generation, 0)
            self.assertEqual(len(method._population.population), 0)
            self.assertEqual(len(method._population._next_gen_pop), 1)
            self.assertEqual(method.evaluation_count, 1)
            self.assertEqual(method._tot_sample_nums, 1)

    def test_cli_resume_resolver_accepts_pbcllm_log_directory(self):
        from main import resolve_resume_log_dir

        with tempfile.TemporaryDirectory() as log_dir:
            os.makedirs(os.path.join(log_dir, "population"))
            os.makedirs(os.path.join(log_dir, "samples"))
            args = SimpleNamespace(
                resume_log_dir=log_dir,
                resume_latest=False,
                method="pbcllm",
                problem="bi_tsp",
            )

            self.assertEqual(resolve_resume_log_dir(args), os.path.abspath(log_dir))


if __name__ == "__main__":
    unittest.main()
