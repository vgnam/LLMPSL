from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from llm4ad.base import LLM, SampleTrimmer, TextFunctionProgramConverter
from llm4ad.method.eoh.population import Population as EoHPopulation
from llm4ad.method.eoh.resume import resume_eoh
from llm4ad.method.funsearch.config import ProgramsDatabaseConfig
from llm4ad.method.funsearch.resume import resume_funsearch
from llm4ad.method.meoh.population import Population as MEoHPopulation
from llm4ad.method.meoh.resume import resume_meoh
from llm4ad.method.moead.population import Population as MOEADPopulation
from llm4ad.method.moead.resume import resume_moead
from llm4ad.method.nsga2.population import Population as NSGA2Population
from llm4ad.method.nsga2.resume import resume_nsga2


def _record(index: int, score, **extra) -> dict:
    return {
        "sample_order": index,
        "algorithm": f"algorithm {index}",
        "function": f"def heuristic(x):\n    return x + {index}\n",
        "score": score,
        **extra,
    }


def _write_samples(log_dir: str, records: list[dict]) -> Path:
    samples_dir = Path(log_dir) / "samples"
    samples_dir.mkdir(parents=True)
    path = samples_dir / "samples_1~200.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


class BaselineResumeTests(unittest.TestCase):
    def test_llm_and_sample_trimmer_support_baseline_shutdown(self):
        class FakeLLM(LLM):
            def draw_sample(self, prompt, *args, **kwargs):
                return str(prompt)

        llm = FakeLLM()
        trimmer = SampleTrimmer(llm)

        self.assertIs(trimmer.llm, llm)
        self.assertIsNone(trimmer.llm.close())

    def test_population_resume_replays_history_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as log_dir:
            history_path = _write_samples(
                log_dir,
                [_record(1, 3.0), _record(2, 2.0), _record(3, 1.0)],
            )
            population_dir = Path(log_dir) / "population"
            population_dir.mkdir()
            (population_dir / "pop_1.json").write_text("[]", encoding="utf-8")
            original_history = history_path.read_text(encoding="utf-8")

            profiler = SimpleNamespace(_log_dir=log_dir, _cur_gen=0)
            method = SimpleNamespace(
                _profiler=profiler,
                _population=EoHPopulation(pop_size=2),
                _pop_size=2,
                _resume_mode=False,
                _tot_sample_nums=0,
                _max_sample_nums=10,
            )

            resume_eoh(method, log_dir)

            self.assertTrue(method._resume_mode)
            self.assertEqual(method._tot_sample_nums, 3)
            self.assertEqual(method._population.generation, 1)
            self.assertEqual(len(method._population.population), 2)
            self.assertEqual(len(method._population._next_gen_pop), 1)
            self.assertEqual(profiler._num_samples, 3)
            self.assertEqual(history_path.read_text(encoding="utf-8"), original_history)

    def test_partial_initial_population_continues_initialization(self):
        with tempfile.TemporaryDirectory() as log_dir:
            _write_samples(log_dir, [_record(1, 2.0), _record(2, 1.0)])
            (Path(log_dir) / "population").mkdir()
            method = SimpleNamespace(
                _profiler=SimpleNamespace(_log_dir=log_dir, _cur_gen=0),
                _population=EoHPopulation(pop_size=10),
                _pop_size=10,
                _resume_mode=True,
                _tot_sample_nums=0,
                _max_sample_nums=10,
            )

            resume_eoh(method, log_dir)

            self.assertFalse(method._resume_mode)
            self.assertEqual(len(method._population._next_gen_pop), 2)

    def test_funsearch_resume_restores_island_assignments(self):
        with tempfile.TemporaryDirectory() as log_dir:
            _write_samples(
                log_dir,
                [_record(1, 2.0, island_id=1), _record(2, 1.0, island_id=1)],
            )
            (Path(log_dir) / "population").mkdir()
            template = TextFunctionProgramConverter.text_to_program(
                "def heuristic(x):\n    return x\n"
            )
            method = SimpleNamespace(
                _profiler=SimpleNamespace(_log_dir=log_dir, _prog_db_order=0),
                _resume_mode=False,
                _tot_sample_nums=0,
                _max_sample_nums=10,
                db_config=ProgramsDatabaseConfig(num_islands=2),
                _template_program=template,
                _function_to_evolve_name="heuristic",
            )

            resume_funsearch(method, log_dir)

            self.assertTrue(method._resume_mode)
            self.assertEqual(method._tot_sample_nums, 2)
            self.assertEqual(method._database.islands[0].get_num_programs(), 0)
            self.assertEqual(method._database.islands[1].get_num_programs(), 2)

    def test_vector_population_methods_resume_from_multi_record_history(self):
        cases = (
            (resume_meoh, MEoHPopulation),
            (resume_nsga2, NSGA2Population),
            (resume_moead, MOEADPopulation),
        )
        for resume_method, population_cls in cases:
            with self.subTest(method=resume_method.__name__), tempfile.TemporaryDirectory() as log_dir:
                _write_samples(
                    log_dir,
                    [_record(index, [float(index), float(10 - index)]) for index in range(1, 6)],
                )
                (Path(log_dir) / "population").mkdir()
                method = SimpleNamespace(
                    _profiler=SimpleNamespace(_log_dir=log_dir, _cur_gen=0),
                    _population=population_cls(pop_size=20),
                    _pop_size=20,
                    _resume_mode=False,
                    _tot_sample_nums=0,
                    _max_sample_nums=10,
                )

                resume_method(method, log_dir)

                self.assertTrue(method._resume_mode)
                self.assertEqual(method._tot_sample_nums, 5)
                self.assertGreater(len(method._population.population), 0)

    def test_cli_accepts_resume_for_every_baseline(self):
        from main import resolve_resume_log_dir

        with tempfile.TemporaryDirectory() as log_dir:
            os.makedirs(os.path.join(log_dir, "samples"))
            os.makedirs(os.path.join(log_dir, "population"))
            for method in ("eoh", "funsearch", "reevo", "meoh", "nsga2", "moead"):
                args = SimpleNamespace(
                    resume_log_dir=log_dir,
                    resume_latest=False,
                    method=method,
                    problem="bi_tsp",
                )
                self.assertEqual(resolve_resume_log_dir(args), os.path.abspath(log_dir))

    def test_cli_finds_latest_log_for_every_baseline(self):
        from main import METHOD_LOG_LABELS, resolve_resume_log_dir

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as root:
            try:
                os.chdir(root)
                for method in ("eoh", "funsearch", "reevo", "meoh", "nsga2", "moead"):
                    log_dir = Path("logs") / METHOD_LOG_LABELS[method] / "bi_tsp" / "run"
                    (log_dir / "samples").mkdir(parents=True)
                    (log_dir / "population").mkdir()
                    args = SimpleNamespace(
                        resume_log_dir=None,
                        resume_latest=True,
                        method=method,
                        problem="bi_tsp",
                    )
                    self.assertEqual(resolve_resume_log_dir(args), str(log_dir.resolve()))
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
