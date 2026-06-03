from __future__ import annotations

import json
import os
from threading import Lock
from typing import Any

from ...base import Function
from ...tools.profiler import ProfilerBase


class PBCProfiler(ProfilerBase):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("method_name", "PBCLLM")
        super().__init__(*args, **kwargs)
        self._pop_lock = Lock()
        self._cur_gen = 0
        if self._log_dir:
            self._ckpt_dir = os.path.join(self._log_dir, "population")
            os.makedirs(self._ckpt_dir, exist_ok=True)

    def _record_payload(self, function: Function) -> dict[str, Any]:
        payload = {
            "algorithm": getattr(function, "algorithm", None),
            "function": str(function),
            "score": function.score,
        }
        pbc = getattr(function, "pbc", None)
        if pbc:
            payload["pbc"] = {
                "individual_hv": pbc.get("individual_hv"),
                "population_hv": pbc.get("population_hv"),
                "population_hv_contribution": pbc.get("population_hv_contribution"),
                "coverage_loss": pbc.get("coverage_loss"),
                "preference_performance": pbc.get("preference_performance"),
                "behavior_diversity": pbc.get("behavior_diversity"),
                "front_count": pbc.get("front_count"),
            }
        return payload

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
            funcs_json = [self._record_payload(func) for func in pop.population]
            path = os.path.join(self._ckpt_dir, f"pop_{pop.generation}.json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump(funcs_json, file, indent=2)
            self._cur_gen = pop.generation

    def finish(self):
        if not self._log_dir:
            return
        pop_dir = os.path.join(self._log_dir, "population")
        latest = None
        if os.path.isdir(pop_dir):
            for name in os.listdir(pop_dir):
                if not name.startswith("pop_") or not name.endswith(".json"):
                    continue
                try:
                    order = int(name[4:-5])
                except ValueError:
                    continue
                if latest is None or order > latest[0]:
                    latest = (order, os.path.join(pop_dir, name))
        records = []
        if latest is not None:
            with open(latest[1], "r", encoding="utf-8") as file:
                records = json.load(file)

        preference_scores = [
            rec.get("pbc", {}).get("preference_performance")
            for rec in records
        ]
        preference_scores = [scores for scores in preference_scores if isinstance(scores, list)]
        summary = {
            "num_records": len(records),
            "method": "PBCLLM",
            "population_hv": None,
            "best_individual_hv": None,
            "best_hv_contribution": None,
            "best_preference_scores": [],
            "mean_behavior_diversity": None,
            "log_dir": self._log_dir,
        }
        population_hvs = [
            rec.get("pbc", {}).get("population_hv")
            for rec in records
            if rec.get("pbc", {}).get("population_hv") is not None
        ]
        if population_hvs:
            summary["population_hv"] = max(float(v) for v in population_hvs)
        individual_hvs = [
            rec.get("pbc", {}).get("individual_hv")
            for rec in records
            if rec.get("pbc", {}).get("individual_hv") is not None
        ]
        if individual_hvs:
            summary["best_individual_hv"] = max(float(v) for v in individual_hvs)
        contributions = [
            rec.get("pbc", {}).get("population_hv_contribution")
            for rec in records
            if rec.get("pbc", {}).get("population_hv_contribution") is not None
        ]
        if contributions:
            summary["best_hv_contribution"] = max(float(v) for v in contributions)
        if preference_scores:
            cols = list(zip(*preference_scores))
            summary["best_preference_scores"] = [min(float(v) for v in col) for col in cols]
        diversity = [
            rec.get("pbc", {}).get("behavior_diversity")
            for rec in records
            if rec.get("pbc", {}).get("behavior_diversity") is not None
        ]
        if diversity:
            summary["mean_behavior_diversity"] = sum(float(v) for v in diversity) / len(diversity)

        with open(os.path.join(self._log_dir, "pbc_report.json"), "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        with open(os.path.join(self._log_dir, "pbc_report.md"), "w", encoding="utf-8") as file:
            file.write("# PBC-LLM Report\n\n")
            for key, value in summary.items():
                file.write(f"- `{key}`: {value}\n")
