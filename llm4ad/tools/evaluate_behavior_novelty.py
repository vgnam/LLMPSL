from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from llm4ad.base import Function, SecureEvaluator, TextFunctionProgramConverter
from llm4ad.method.PBCLLM.population import (
    analyze_mo_result,
    behavior_novelty,
    default_preference_vectors,
)


_MISSING = object()


def _safe_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _function_hash(func: Function) -> str:
    return hashlib.sha1(str(func).encode("utf-8")).hexdigest()[:10]


def _objective_num(evaluator) -> int | None:
    value = getattr(evaluator, "objective_num", None)
    if value is not None:
        return int(value)
    labels = getattr(evaluator, "objective_labels", None)
    if labels:
        return len(labels)
    return None


def _normalization_bounds(evaluator, objective_num: int):
    ideal = getattr(evaluator, "normalization_ideal", None)
    nadir = getattr(evaluator, "normalization_nadir", None)
    if ideal is not None and nadir is not None:
        ideal = np.asarray(ideal, dtype=float)
        nadir = np.asarray(nadir, dtype=float)
        if ideal.shape == (objective_num,) and nadir.shape == (objective_num,):
            return ideal, nadir

    ref_point = getattr(evaluator, "ref_point", None)
    if ref_point is None:
        return None, None
    ref_point = np.asarray(ref_point, dtype=float)
    if ref_point.shape != (objective_num,) or not np.all(np.isfinite(ref_point)):
        return None, None
    if not np.all(ref_point > 0):
        return None, None
    return np.zeros(objective_num, dtype=float), ref_point


def _hv_ref_point(evaluator, objective_num: int):
    ref_point = getattr(evaluator, "ref_point", None)
    if ref_point is None:
        return None
    ref_point = np.asarray(ref_point, dtype=float)
    if ref_point.shape != (objective_num,) or not np.all(np.isfinite(ref_point)):
        return None
    return ref_point


def _evaluate_trace(func: Function, evaluator, preference_vectors, rho: float):
    program = TextFunctionProgramConverter.function_to_program(func, evaluator.template_program)
    if program is None:
        return None, {"status": "invalid", "error": "program_conversion_failed"}

    result, wall_time = SecureEvaluator(evaluator).evaluate_program_record_time(program)
    if not isinstance(result, dict):
        return None, {
            "status": "invalid",
            "error": "evaluation_failed_or_timeout",
            "wall_time": wall_time,
        }

    objective_num = int(result.get("objective_num", 0) or 0)
    ideal, nadir = _normalization_bounds(evaluator, objective_num)
    pbc = analyze_mo_result(
        result,
        preference_vectors,
        rho=rho,
        normalization_ideal=ideal,
        normalization_nadir=nadir,
        hv_ref_point=_hv_ref_point(evaluator, objective_num),
    )
    if pbc is None:
        return None, {
            "status": "invalid",
            "error": "missing_or_invalid_mo_trace",
            "wall_time": wall_time,
        }

    func.pbc = pbc
    return pbc, {
        "status": "ok",
        "function_sha1": _function_hash(func),
        "source_score": getattr(func, "score", None),
        "legacy_score": result.get("legacy_score"),
        "individual_hv": pbc.get("individual_hv"),
        "front_count": pbc.get("front_count"),
        "wall_time": wall_time,
    }


def evaluate_population_behavior_novelty(
    functions: Sequence[Function],
    evaluator,
    *,
    method: str,
    problem: str,
    log_dir: str | Path | None = None,
    k: int = 3,
    rho: float = 0.05,
    eval_seed: int | None = None,
) -> dict[str, Any]:
    functions = [func for func in functions if func is not None]
    objective_num = _objective_num(evaluator)
    report: dict[str, Any] = {
        "method": method,
        "problem": problem,
        "metric": "mean_behavior_novelty",
        "log_dir": str(log_dir) if log_dir is not None else None,
        "k": int(k),
        "rho": float(rho),
        "eval_seed": eval_seed,
        "num_evaluated": len(functions),
        "num_valid": 0,
        "mean_behavior_novelty": None,
        "std_behavior_novelty": None,
        "min_behavior_novelty": None,
        "max_behavior_novelty": None,
        "records": [],
    }
    if objective_num is None or objective_num < 2:
        report["error"] = "behavior novelty requires objective_num >= 2"
        return report

    old_trace = getattr(evaluator, "return_mo_trace", _MISSING)
    old_safe = getattr(evaluator, "safe_evaluate", _MISSING)
    old_eval_seed = getattr(evaluator, "eval_seed", _MISSING)
    try:
        setattr(evaluator, "return_mo_trace", True)
        # Post-training behavior logging reuses already-valid final programs.
        # Direct evaluation avoids Windows subprocess permission failures seen
        # in trace post-evaluation.
        setattr(evaluator, "safe_evaluate", False)
        if eval_seed is not None and hasattr(evaluator, "eval_seed"):
            setattr(evaluator, "eval_seed", eval_seed)

        preference_vectors = default_preference_vectors(objective_num)
        valid_functions: list[Function] = []
        records: list[dict[str, Any]] = []
        record_by_id: dict[int, dict[str, Any]] = {}
        for index, func in enumerate(functions):
            pbc, record = _evaluate_trace(func, evaluator, preference_vectors, rho)
            record["index"] = index
            if pbc is not None:
                valid_functions.append(func)
                record_by_id[id(func)] = record
            records.append(record)

        novelties = []
        for func in valid_functions:
            novelty = behavior_novelty(func, valid_functions, k=k)
            func.pbc["post_train_behavior_novelty"] = novelty
            record_by_id[id(func)]["behavior_novelty"] = novelty
            novelties.append(novelty)

        report["records"] = records
        report["num_valid"] = len(valid_functions)
        if novelties:
            values = np.asarray(novelties, dtype=float)
            report.update(
                {
                    "mean_behavior_novelty": _safe_float(np.mean(values)),
                    "std_behavior_novelty": _safe_float(np.std(values)),
                    "min_behavior_novelty": _safe_float(np.min(values)),
                    "max_behavior_novelty": _safe_float(np.max(values)),
                }
            )
    finally:
        if old_trace is not _MISSING:
            setattr(evaluator, "return_mo_trace", old_trace)
        if old_safe is not _MISSING:
            setattr(evaluator, "safe_evaluate", old_safe)
        if old_eval_seed is not _MISSING:
            setattr(evaluator, "eval_seed", old_eval_seed)

    return report


def _update_metrics_report(log_dir: Path, report: dict[str, Any]) -> None:
    metrics_path = log_dir / "metrics_report.json"
    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        metrics = {}
    if not isinstance(metrics, dict):
        metrics = {}

    for key in (
        "mean_behavior_novelty",
        "std_behavior_novelty",
        "min_behavior_novelty",
        "max_behavior_novelty",
    ):
        metrics[key] = report.get(key)
    metrics["num_behavior_records"] = report.get("num_valid")
    metrics["behavior_novelty_k"] = report.get("k")

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    md_path = log_dir / "metrics_report.md"
    with md_path.open("w", encoding="utf-8") as file:
        file.write("# Metrics Report\n\n")
        for key, value in metrics.items():
            file.write(f"- `{key}`: {value}\n")


def write_behavior_novelty_report(report: dict[str, Any]) -> None:
    if not report.get("log_dir"):
        return
    log_dir = Path(report["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    json_path = log_dir / "behavior_novelty_report.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    md_path = log_dir / "behavior_novelty_report.md"
    with md_path.open("w", encoding="utf-8") as file:
        file.write(f"# Behavior Novelty Report: {report['method']} / {report['problem']}\n\n")
        for key in (
            "num_evaluated",
            "num_valid",
            "k",
            "eval_seed",
            "mean_behavior_novelty",
            "std_behavior_novelty",
            "min_behavior_novelty",
            "max_behavior_novelty",
        ):
            file.write(f"- `{key}`: {report.get(key)}\n")

    _update_metrics_report(log_dir, report)
