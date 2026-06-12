from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pymoo.indicators.hv import HV

from llm4ad.base import SecureEvaluator, TextFunctionProgramConverter
from llm4ad.task.optimization.hv_utils import reference_hypervolume_scale, scale_hypervolume
from llm4ad.task.optimization.registry import PROBLEM_CONFIGS, build_problem, normalize_problem_name
from llm4ad.tools.evaluate_all_sizes import (
    _fmt,
    _function_text,
    _select_records,
    discover_size_configs,
)
from llm4ad.tools.report_metrics import load_final_records


def _as_points(values) -> np.ndarray:
    if values is None:
        return np.empty((0, 0), dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return np.empty((0, 0), dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.size == 0:
        return np.empty((0, 0), dtype=float)
    return arr


def _nondominated_min(points: np.ndarray) -> np.ndarray:
    points = _as_points(points)
    if points.size == 0:
        return points
    points = np.unique(points, axis=0)
    keep = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        if not keep[i]:
            continue
        dominates_i = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        if np.any(dominates_i):
            keep[i] = False
    return points[keep]


def _normalization_bounds(evaluator, objective_num: int) -> tuple[np.ndarray, np.ndarray] | None:
    ideal = getattr(evaluator, "normalization_ideal", None)
    nadir = getattr(evaluator, "normalization_nadir", None)
    if ideal is not None and nadir is not None:
        ideal = np.asarray(ideal, dtype=float)
        nadir = np.asarray(nadir, dtype=float)
        if (
            ideal.shape == (objective_num,)
            and nadir.shape == (objective_num,)
            and np.all(np.isfinite(ideal))
            and np.all(np.isfinite(nadir))
            and np.all(nadir > ideal)
        ):
            return ideal, nadir

    ref_point = np.asarray(getattr(evaluator, "ref_point", None), dtype=float)
    if (
        ref_point.shape == (objective_num,)
        and np.all(np.isfinite(ref_point))
        and np.all(ref_point > 0)
    ):
        return np.zeros(objective_num, dtype=float), ref_point
    return None


def _normalize_front(front: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    return np.maximum((front - ideal) / (nadir - ideal), 0.0)


def _additive_epsilon_to_ideal(front: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> float | None:
    front = _as_points(front)
    if front.size == 0:
        return None
    normalized = _normalize_front(front, ideal, nadir)
    return float(np.min(np.max(normalized, axis=1)))


def _spacing(front: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> float | None:
    front = np.unique(_as_points(front), axis=0)
    if front.size == 0:
        return None
    if len(front) <= 2:
        return 0.0
    normalized = _normalize_front(front, ideal, nadir)
    distances = np.sum(np.abs(normalized[:, None, :] - normalized[None, :, :]), axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    return float(np.sqrt(np.sum((nearest - np.mean(nearest)) ** 2) / (len(nearest) - 1)))


def _finite_mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=float,
    )
    if finite.size == 0:
        return None, None
    return float(np.mean(finite)), float(np.std(finite))


def _program_from_record(record: dict[str, Any], evaluator):
    function_text = _function_text(record)
    if function_text is None:
        return None, None, "missing_function"
    func = TextFunctionProgramConverter.text_to_function(function_text)
    if func is None:
        return None, None, "parse_failed"
    program = TextFunctionProgramConverter.function_to_program(func, evaluator.template_program)
    if program is None:
        return None, None, "program_conversion_failed"
    return func, program, None


def _evaluate_trace_record(record: dict[str, Any], evaluator) -> dict[str, Any]:
    func, program, error = _program_from_record(record, evaluator)
    if error is not None:
        return {"status": "invalid", "error": error}

    result, wall_time = SecureEvaluator(evaluator).evaluate_program_record_time(program)
    if not isinstance(result, dict):
        return {
            "status": "invalid",
            "error": "evaluation_failed_or_timeout",
            "wall_time": wall_time,
        }

    fronts = result.get("fronts")
    if not isinstance(fronts, list):
        return {
            "status": "invalid",
            "error": "missing_fronts",
            "wall_time": wall_time,
        }

    function_hash = hashlib.sha1(str(func).encode("utf-8")).hexdigest()[:10]
    return {
        "status": "ok",
        "source_index": record.get("_source_index"),
        "sample_order": record.get("sample_order"),
        "function_sha1": function_hash,
        "source_score": record.get("score"),
        "legacy_score": result.get("legacy_score"),
        "fronts": fronts,
        "wall_time": wall_time,
    }


def _population_front_summary(trace_results: list[dict[str, Any]], evaluator, n_instance: int) -> dict[str, Any]:
    valid = [item for item in trace_results if item.get("status") == "ok"]
    summary: dict[str, Any] = {
        "num_evaluated": len(trace_results),
        "num_valid": len(valid),
    }
    if not valid:
        return summary

    ref_point = np.asarray(getattr(evaluator, "ref_point"), dtype=float)
    ideal_point = getattr(evaluator, "ideal_point", None)
    summary.update(
        {
            "hv_reference_point": ref_point.tolist(),
            "hv_ideal_point": None if ideal_point is None else np.asarray(ideal_point, dtype=float).tolist(),
            "hv_normalization_area": reference_hypervolume_scale(ref_point, ideal_point),
        }
    )
    hv_indicator = HV(ref_point=ref_point)
    bounds = _normalization_bounds(evaluator, len(ref_point))
    instance_hvs: list[float] = []
    instance_points: list[int] = []
    instance_epsilons: list[float | None] = []
    instance_spacings: list[float | None] = []

    for instance_idx in range(n_instance):
        merged = []
        for item in valid:
            fronts = item.get("fronts") or []
            if instance_idx >= len(fronts):
                continue
            points = _as_points(fronts[instance_idx])
            if points.size:
                merged.append(points)
        if merged:
            front = _nondominated_min(np.vstack(merged))
        else:
            front = np.empty((0, len(ref_point)), dtype=float)
        if front.size:
            hv_value = hv_indicator(front)
            scaled_hv = scale_hypervolume(hv_value, ref_point, ideal_point)
        else:
            scaled_hv = 0.0
        instance_hvs.append(float(scaled_hv))
        instance_points.append(int(len(front)))
        if bounds is None:
            instance_epsilons.append(None)
            instance_spacings.append(None)
        else:
            ideal, nadir = bounds
            instance_epsilons.append(_additive_epsilon_to_ideal(front, ideal, nadir))
            instance_spacings.append(_spacing(front, ideal, nadir))

    wall_times = [
        float(item.get("wall_time"))
        for item in valid
        if item.get("wall_time") is not None and math.isfinite(float(item.get("wall_time")))
    ]
    epsilon_mean, epsilon_std = _finite_mean_std(instance_epsilons)
    spacing_mean, spacing_std = _finite_mean_std(instance_spacings)
    summary.update(
        {
            "population_front_hv": float(np.mean(instance_hvs)),
            "population_front_hv_std": float(np.std(instance_hvs)),
            "mean_additive_epsilon": epsilon_mean,
            "additive_epsilon_std": epsilon_std,
            "mean_spacing": spacing_mean,
            "spacing_std": spacing_std,
            "mean_population_front_points": float(np.mean(instance_points)),
            "min_population_front_points": int(np.min(instance_points)),
            "max_population_front_points": int(np.max(instance_points)),
            "mean_trace_wall_time": float(np.mean(wall_times)) if wall_times else None,
            "total_trace_wall_time": float(np.sum(wall_times)) if wall_times else None,
            "instance_hv": instance_hvs,
            "instance_additive_epsilon": instance_epsilons,
            "instance_spacing": instance_spacings,
            "instance_front_points": instance_points,
        }
    )
    return summary


def evaluate_population_front_all_sizes(
    log_dir: str | Path,
    *,
    method: str,
    problem: str,
    top_k: int = 0,
    seed: int = 2025,
    eval_seed: int | None = 2025,
    timeout_seconds: int | None = None,
    problem_size: int | None = None,
    problem_sizes: list[int] | None = None,
    n_instance: int | None = None,
) -> dict[str, Any]:
    if problem_size is not None and problem_sizes is not None:
        raise ValueError("Use either problem_size or problem_sizes, not both.")

    problem = normalize_problem_name(problem)
    log_dir = Path(log_dir)
    records = load_final_records(log_dir)
    selected_records = _select_records(records, top_k=top_k)
    size_configs = discover_size_configs(problem, seed=seed)
    if problem_size is not None:
        size_configs = [item for item in size_configs if item["problem_size"] == problem_size]
    if problem_sizes is not None:
        selected_sizes = set(problem_sizes)
        size_configs = [item for item in size_configs if item["problem_size"] in selected_sizes]
    if n_instance is not None:
        size_configs = [item for item in size_configs if item["n_instance"] == n_instance]

    report: dict[str, Any] = {
        "method": method,
        "problem": problem,
        "metric": "population_front_hv",
        "front_quality_metrics": {
            "additive_epsilon": "normalized additive epsilon-to-ideal; lower is better",
            "spacing": "normalized Manhattan nearest-neighbor spacing; lower is better",
        },
        "log_dir": str(log_dir),
        "num_records": len(records),
        "num_selected": len(selected_records),
        "top_k": top_k,
        "seed": seed,
        "eval_seed": eval_seed,
        "timeout_seconds": timeout_seconds,
        "sizes": [],
    }

    for size_config in size_configs:
        evaluator = build_problem(
            problem,
            n_instance=size_config["n_instance"],
            problem_size=size_config["problem_size"],
            seed=seed,
            eval_seed=eval_seed,
            timeout_seconds=timeout_seconds,
        )
        for key in ("hv_normalization_policy", "hv_fixed_ideal_magnitude_by_size"):
            value = getattr(evaluator, key, None)
            if value is not None:
                report[key] = value
        setattr(evaluator, "return_mo_trace", True)
        # The population-front metric needs the full trace object. In this
        # Windows workspace, spawning a safe subprocess can fail with
        # WinError 5, so this post-evaluation runs the already-saved final
        # population directly.
        evaluator.safe_evaluate = False
        trace_results = [
            _evaluate_trace_record(record, evaluator)
            for record in selected_records
        ]
        compact_results = [
            {key: value for key, value in item.items() if key != "fronts"}
            for item in trace_results
        ]
        size_report = {
            **size_config,
            "trace_results": compact_results,
            **_population_front_summary(trace_results, evaluator, size_config["n_instance"]),
        }
        report["sizes"].append(size_report)

    return report


def write_population_front_report(report: dict[str, Any]) -> None:
    log_dir = Path(report["log_dir"])
    problem = report["problem"]
    json_path = log_dir / f"population_front_evaluation_{problem}.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    md_path = log_dir / f"population_front_evaluation_{problem}.md"
    with md_path.open("w", encoding="utf-8") as file:
        file.write(f"# Population-Front Evaluation: {report['method']} / {problem}\n\n")
        file.write(f"- `log_dir`: {report['log_dir']}\n")
        file.write(f"- `num_selected`: {report['num_selected']}\n")
        file.write(f"- `eval_seed`: {report['eval_seed']}\n")
        if report.get("hv_normalization_policy") is not None:
            file.write(f"- `hv_normalization_policy`: {report['hv_normalization_policy']}\n")
        if report.get("hv_fixed_ideal_magnitude_by_size") is not None:
            file.write(
                f"- `hv_fixed_ideal_magnitude_by_size`: "
                f"{report['hv_fixed_ideal_magnitude_by_size']}\n"
            )
        file.write("\n")
        file.write(
            "| size | n_instance | valid | population_front_hv | hv_std | "
            "epsilon_mean | epsilon_std | spacing_mean | spacing_std | "
            "mean_front_points | min_front_points | max_front_points | mean_trace_wall_time |\n"
        )
        file.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for item in report["sizes"]:
            file.write(
                "| {problem_size} | {n_instance} | {num_valid}/{num_evaluated} | "
                "{population_front_hv} | {population_front_hv_std} | "
                "{mean_additive_epsilon} | {additive_epsilon_std} | "
                "{mean_spacing} | {spacing_std} | "
                "{mean_population_front_points} | {min_population_front_points} | "
                "{max_population_front_points} | {mean_trace_wall_time} |\n".format(
                    problem_size=item.get("problem_size"),
                    n_instance=item.get("n_instance"),
                    num_valid=item.get("num_valid", 0),
                    num_evaluated=item.get("num_evaluated", 0),
                    population_front_hv=_fmt(item.get("population_front_hv")),
                    population_front_hv_std=_fmt(item.get("population_front_hv_std")),
                    mean_additive_epsilon=_fmt(item.get("mean_additive_epsilon")),
                    additive_epsilon_std=_fmt(item.get("additive_epsilon_std")),
                    mean_spacing=_fmt(item.get("mean_spacing")),
                    spacing_std=_fmt(item.get("spacing_std")),
                    mean_population_front_points=_fmt(item.get("mean_population_front_points")),
                    min_population_front_points=_fmt(item.get("min_population_front_points")),
                    max_population_front_points=_fmt(item.get("max_population_front_points")),
                    mean_trace_wall_time=_fmt(item.get("mean_trace_wall_time")),
                )
            )


def format_population_front_table(reports: list[dict[str, Any]]) -> str:
    rows = [
        "| method | problem | size | n_instance | valid | population_front_hv | hv_std | epsilon_mean | epsilon_std | spacing_mean | spacing_std | mean_front_points | mean_trace_wall_time |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        for item in report["sizes"]:
            rows.append(
                "| {method} | {problem} | {problem_size} | {n_instance} | {valid} | "
                "{hv} | {std} | {epsilon} | {epsilon_std} | {spacing} | {spacing_std} | "
                "{points} | {wall} |".format(
                    method=report["method"],
                    problem=report["problem"],
                    problem_size=item.get("problem_size"),
                    n_instance=item.get("n_instance"),
                    valid=f"{item.get('num_valid', 0)}/{item.get('num_evaluated', 0)}",
                    hv=_fmt(item.get("population_front_hv")),
                    std=_fmt(item.get("population_front_hv_std")),
                    epsilon=_fmt(item.get("mean_additive_epsilon")),
                    epsilon_std=_fmt(item.get("additive_epsilon_std")),
                    spacing=_fmt(item.get("mean_spacing")),
                    spacing_std=_fmt(item.get("spacing_std")),
                    points=_fmt(item.get("mean_population_front_points")),
                    wall=_fmt(item.get("mean_trace_wall_time")),
                )
            )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the union Pareto front produced by a final heuristic population.")
    parser.add_argument("--problem", choices=sorted(PROBLEM_CONFIGS), default="bi_tsp")
    parser.add_argument("--method", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--top-k", type=int, default=0, help="0 means evaluate the whole final population.")
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--problem-size", type=int, default=None)
    size_group.add_argument(
        "--problem-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate only these problem sizes, for example: --problem-sizes 20 50 100.",
    )
    parser.add_argument("--n-instance", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--eval-seed", type=int, default=2025)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    report = evaluate_population_front_all_sizes(
        args.log_dir,
        method=args.method,
        problem=args.problem,
        top_k=args.top_k,
        seed=args.seed,
        eval_seed=args.eval_seed,
        timeout_seconds=args.timeout_seconds,
        problem_size=args.problem_size,
        problem_sizes=args.problem_sizes,
        n_instance=args.n_instance,
    )
    if not args.no_write:
        write_population_front_report(report)
    print(format_population_front_table([report]))


if __name__ == "__main__":
    main()
