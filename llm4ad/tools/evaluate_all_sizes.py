from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.base import SecureEvaluator, TextFunctionProgramConverter
from llm4ad.task.optimization.registry import (
    PROBLEM_CONFIGS,
    build_problem,
    get_problem_config,
    normalize_problem_name,
)
from llm4ad.tools.report_metrics import load_final_records


METHOD_LABELS = {
    "mpage": "MPaGE",
    "llmpfg": "LLMPFG",
    "pbcllm": "PBCLLM",
}


def _score_vector(score) -> list[float] | None:
    if score is None:
        return None
    if not isinstance(score, (list, tuple)):
        score = [score]
    try:
        values = [float(x) for x in score]
    except (TypeError, ValueError):
        return None
    if not values or not all(math.isfinite(x) for x in values):
        return None
    return values


def _function_text(record: dict[str, Any]) -> str | None:
    value = record.get("function") or record.get("code")
    return value if isinstance(value, str) and value.strip() else None


def _record_key(record: dict[str, Any]):
    score = _score_vector(record.get("score"))
    if score is None:
        return (float("inf"), float("inf"))
    second = score[1] if len(score) > 1 else float("inf")
    return (score[0], second)


def _record_code_novelty(record: dict[str, Any]) -> float | None:
    score = _score_vector(record.get("score"))
    if score is not None and len(score) >= 3:
        return -score[2]

    value = record.get("code_novelty")
    try:
        novelty = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(novelty):
        return None
    return novelty


def _select_records(records: list[dict[str, Any]], top_k: int = 0) -> list[dict[str, Any]]:
    selected = [
        {**record, "_source_index": idx}
        for idx, record in enumerate(records)
        if _function_text(record) is not None and _score_vector(record.get("score")) is not None
    ]
    selected.sort(key=_record_key)
    if top_k and top_k > 0:
        selected = selected[:top_k]
    return selected


def _looks_like_log_dir(path: Path) -> bool:
    return path.is_dir() and ((path / "population").exists() or (path / "samples").exists())


def latest_method_log(method: str, logs_root: str | Path = "logs") -> Path | None:
    method = method.lower()
    label = METHOD_LABELS[method]
    logs_root = Path(logs_root)
    candidates = []

    method_root = logs_root / label
    if method_root.exists():
        candidates.extend(path for path in method_root.rglob("*") if _looks_like_log_dir(path))

    candidates.extend(
        path
        for path in logs_root.rglob(f"*_{label}")
        if _looks_like_log_dir(path)
    )
    candidates = list({path.resolve(): path for path in candidates}.values())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def discover_size_configs(problem: str, seed: int = 2025) -> list[dict[str, int]]:
    config = get_problem_config(problem)
    pattern = re.compile(
        rf"instances_n(?P<n>\d+)_{re.escape(config.size_label)}(?P<size>\d+)_seed(?P<seed>\d+)\.npz$"
    )
    discovered = []
    for path in sorted(config.data_dir.glob("*.npz")):
        match = pattern.match(path.name)
        if not match:
            continue
        if int(match.group("seed")) != seed:
            continue
        discovered.append(
            {
                "n_instance": int(match.group("n")),
                "problem_size": int(match.group("size")),
            }
        )

    if not discovered:
        discovered.append(
            {
                "n_instance": config.default_n_instance,
                "problem_size": config.default_problem_size,
            }
        )

    return sorted(discovered, key=lambda item: item["problem_size"])


def _evaluate_one_record(record: dict[str, Any], evaluator) -> dict[str, Any]:
    function_text = _function_text(record)
    if function_text is None:
        return {"status": "invalid", "error": "missing_function"}

    func = TextFunctionProgramConverter.text_to_function(function_text)
    if func is None:
        return {"status": "invalid", "error": "parse_failed"}

    program = TextFunctionProgramConverter.function_to_program(func, evaluator.template_program)
    if program is None:
        return {"status": "invalid", "error": "program_conversion_failed"}

    score, wall_time = SecureEvaluator(evaluator).evaluate_program_record_time(program)
    score_vec = _score_vector(score)
    if score_vec is None:
        return {
            "status": "invalid",
            "error": "evaluation_failed_or_timeout",
            "wall_time": wall_time,
        }

    function_hash = hashlib.sha1(str(func).encode("utf-8")).hexdigest()[:10]
    result = {
        "status": "ok",
        "source_index": record.get("_source_index"),
        "sample_order": record.get("sample_order"),
        "function_sha1": function_hash,
        "source_score": record.get("score"),
        "score": score_vec,
        "inner_hv": -score_vec[0],
        "wall_time": wall_time,
    }
    code_novelty = _record_code_novelty(record)
    if code_novelty is not None:
        result["code_novelty"] = code_novelty
    if len(score_vec) > 1:
        result["runtime"] = score_vec[1]
    return result


def _summarize_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in evaluations if item.get("status") == "ok"]
    summary = {
        "num_evaluated": len(evaluations),
        "num_valid": len(valid),
    }
    if not valid:
        return summary

    inner_hv = np.array([item["inner_hv"] for item in valid], dtype=float)
    best_idx = int(np.argmax(inner_hv))
    best = valid[best_idx]
    summary.update(
        {
            "best_inner_hv": float(np.max(inner_hv)),
            "mean_inner_hv": float(np.mean(inner_hv)),
            "best_function_sha1": best.get("function_sha1"),
            "runtime_at_best_inner_hv": best.get("runtime"),
        }
    )

    runtimes = [item.get("runtime") for item in valid if item.get("runtime") is not None]
    if runtimes:
        best_runtime_idx = min(
            (idx for idx, item in enumerate(valid) if item.get("runtime") is not None),
            key=lambda idx: valid[idx]["runtime"],
        )
        summary["best_runtime"] = float(np.min(runtimes))
        summary["mean_runtime"] = float(np.mean(runtimes))
        summary["function_sha1_at_best_runtime"] = valid[best_runtime_idx].get("function_sha1")

    novelties = [
        item.get("code_novelty")
        for item in valid
        if item.get("code_novelty") is not None
    ]
    if novelties:
        best_novelty_idx = max(
            (idx for idx, item in enumerate(valid) if item.get("code_novelty") is not None),
            key=lambda idx: valid[idx]["code_novelty"],
        )
        summary["best_code_novelty"] = float(np.max(novelties))
        summary["mean_code_novelty"] = float(np.mean(novelties))
        summary["function_sha1_at_best_code_novelty"] = valid[best_novelty_idx].get("function_sha1")
    return summary


def evaluate_log_all_sizes(
    log_dir: str | Path,
    *,
    method: str,
    problem: str,
    top_k: int = 0,
    seed: int = 2025,
    eval_seed: int | None = 2025,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    problem = normalize_problem_name(problem)
    log_dir = Path(log_dir)
    records = load_final_records(log_dir)
    selected_records = _select_records(records, top_k=top_k)
    size_configs = discover_size_configs(problem, seed=seed)

    report: dict[str, Any] = {
        "method": method,
        "problem": problem,
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
        evaluations = [
            _evaluate_one_record(record, evaluator)
            for record in selected_records
        ]
        size_report = {
            **size_config,
            "evaluations": evaluations,
            **_summarize_evaluations(evaluations),
        }
        report["sizes"].append(size_report)

    return report


def write_all_size_report(report: dict[str, Any]) -> None:
    log_dir = Path(report["log_dir"])
    problem = report["problem"]
    json_path = log_dir / f"all_size_evaluation_{problem}.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    md_path = log_dir / f"all_size_evaluation_{problem}.md"
    with md_path.open("w", encoding="utf-8") as file:
        file.write(f"# All-Size Evaluation: {report['method']} / {problem}\n\n")
        file.write(f"- `log_dir`: {report['log_dir']}\n")
        file.write(f"- `num_selected`: {report['num_selected']}\n")
        file.write(f"- `eval_seed`: {report['eval_seed']}\n\n")
        file.write(
            "| size | n_instance | valid | best_inner_hv | mean_inner_hv | "
            "runtime_at_best_hv | best_runtime | best_code_novelty |\n"
        )
        file.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for item in report["sizes"]:
            file.write(
                "| {problem_size} | {n_instance} | {num_valid}/{num_evaluated} | "
                "{best_inner_hv} | {mean_inner_hv} | {runtime_at_best_inner_hv} | "
                "{best_runtime} | {best_code_novelty} |\n".format(
                    problem_size=item.get("problem_size"),
                    n_instance=item.get("n_instance"),
                    num_valid=item.get("num_valid", 0),
                    num_evaluated=item.get("num_evaluated", 0),
                    best_inner_hv=_fmt(item.get("best_inner_hv")),
                    mean_inner_hv=_fmt(item.get("mean_inner_hv")),
                    runtime_at_best_inner_hv=_fmt(item.get("runtime_at_best_inner_hv")),
                    best_runtime=_fmt(item.get("best_runtime")),
                    best_code_novelty=_fmt(item.get("best_code_novelty")),
                )
            )


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_reports_table(reports: list[dict[str, Any]]) -> str:
    rows = [
        "| method | problem | size | n_instance | valid | best_inner_hv | mean_inner_hv | best_runtime | runtime_at_best_hv | best_code_novelty |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        for item in report["sizes"]:
            rows.append(
                "| {method} | {problem} | {problem_size} | {n_instance} | {valid} | "
                "{best} | {mean} | {best_runtime} | {runtime} | {best_code_novelty} |".format(
                    method=report["method"],
                    problem=report["problem"],
                    problem_size=item.get("problem_size"),
                    n_instance=item.get("n_instance"),
                    valid=f"{item.get('num_valid', 0)}/{item.get('num_evaluated', 0)}",
                    best=_fmt(item.get("best_inner_hv")),
                    mean=_fmt(item.get("mean_inner_hv")),
                    best_runtime=_fmt(item.get("best_runtime")),
                    runtime=_fmt(item.get("runtime_at_best_inner_hv")),
                    best_code_novelty=_fmt(item.get("best_code_novelty")),
                )
            )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate final MPaGE populations on all available instance sizes.")
    parser.add_argument("--problem", choices=sorted(PROBLEM_CONFIGS), default="bi_tsp")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_LABELS), default=["mpage"])
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--mpage-log", default=None)
    parser.add_argument("--top-k", type=int, default=0, help="0 means evaluate the whole final population.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--eval-seed", type=int, default=2025)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--no-write", action="store_true", help="Print only; do not write reports into log dirs.")
    args = parser.parse_args()

    explicit_logs = {
        "mpage": args.mpage_log,
    }
    reports = []
    for method in args.methods:
        log_dir = explicit_logs.get(method)
        log_dir = Path(log_dir) if log_dir else latest_method_log(method, args.logs_root)
        if log_dir is None:
            print(f"Skipping {method}: no log directory found.")
            continue
        report = evaluate_log_all_sizes(
            log_dir,
            method=method,
            problem=args.problem,
            top_k=args.top_k,
            seed=args.seed,
            eval_seed=args.eval_seed,
            timeout_seconds=args.timeout_seconds,
        )
        if not args.no_write:
            write_all_size_report(report)
        reports.append(report)

    if reports:
        print(format_reports_table(reports))


if __name__ == "__main__":
    main()
