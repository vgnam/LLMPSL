from __future__ import annotations

import copy
import json
import os
from threading import Lock

from ...base import Function
from ...tools.profiler import ProfilerBase


class PBCProfiler(ProfilerBase):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("method_name", "PBCLLM")
        super().__init__(*args, **kwargs)
        self._pop_lock = Lock()
        self._cur_gen = 0
        self._latest_pbc_records: list[dict] = []
        if self._log_dir:
            self._ckpt_dir = os.path.join(self._log_dir, "population")
            os.makedirs(self._ckpt_dir, exist_ok=True)

    def _record_payload(self, function: Function) -> dict:
        return {
            "algorithm": getattr(function, "algorithm", None),
            "function": str(function),
            "score": function.score,
        }

    def _write_json(self, function: Function, *, record_type="history", record_sep=200):
        if not self._log_dir:
            return
        sample_order = getattr(self.__class__, "_num_samples", 0)
        content = {"sample_order": sample_order, **self._record_payload(function)}
        filename = (
            f"samples_{(sample_order // record_sep) * record_sep}~{(sample_order // record_sep) * record_sep + record_sep}.json"
            if record_type == "history"
            else "samples_best.json"
        )
        path = os.path.join(self._samples_json_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        data.append(content)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def register_population(self, pop):
        if not self._log_dir:
            return
        with self._pop_lock:
            if pop.generation == 0 or pop.generation == self._cur_gen:
                return
            self._latest_pbc_records = [
                copy.deepcopy(getattr(func, "pbc", None) or {})
                for func in pop.population
            ]
            funcs_json = [self._record_payload(func) for func in pop.population]
            path = os.path.join(self._ckpt_dir, f"pop_{pop.generation}.json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump(funcs_json, file, indent=2)
            self._cur_gen = pop.generation

    def finish(self):
        if not self._log_dir:
            return
        records = self._latest_pbc_records
        preference_scores = [
            pbc.get("preference_performance")
            for pbc in records
            if isinstance(pbc.get("preference_performance"), list)
        ]
        summary = {
            "num_records": len(records),
            "method": "PBCLLM",
            "population_hv": None,
            "best_individual_hv": None,
            "best_hv_contribution": None,
            "best_preference_scores": [],
            "mean_behavior_diversity": None,
            "evaluation_seeds": None,
            "log_dir": self._log_dir,
        }

        seed_lists = [pbc.get("evaluation_seeds") for pbc in records if pbc.get("evaluation_seeds")]
        population_hvs = [pbc.get("population_hv") for pbc in records if pbc.get("population_hv") is not None]
        individual_hvs = [pbc.get("individual_hv") for pbc in records if pbc.get("individual_hv") is not None]
        contributions = [
            pbc.get("population_hv_contribution")
            for pbc in records
            if pbc.get("population_hv_contribution") is not None
        ]
        diversity = [
            pbc.get("behavior_diversity")
            for pbc in records
            if pbc.get("behavior_diversity") is not None
        ]
        if seed_lists:
            summary["evaluation_seeds"] = seed_lists[0]
        if population_hvs:
            summary["population_hv"] = max(float(value) for value in population_hvs)
        if individual_hvs:
            summary["best_individual_hv"] = max(float(value) for value in individual_hvs)
        if contributions:
            summary["best_hv_contribution"] = max(float(value) for value in contributions)
        if preference_scores:
            summary["best_preference_scores"] = [
                min(float(value) for value in column)
                for column in zip(*preference_scores)
            ]
        if diversity:
            summary["mean_behavior_diversity"] = sum(float(value) for value in diversity) / len(diversity)

        with open(os.path.join(self._log_dir, "pbc_report.json"), "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        with open(os.path.join(self._log_dir, "pbc_report.md"), "w", encoding="utf-8") as file:
            file.write("# PBC-LLM Report\n\n")
            for key, value in summary.items():
                file.write(f"- `{key}`: {value}\n")
