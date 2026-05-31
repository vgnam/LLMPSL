from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


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


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _latest_population_file(log_dir: Path) -> Path | None:
    pop_dir = log_dir / "population"
    if not pop_dir.exists():
        return None
    pop_files = []
    for path in pop_dir.glob("pop_*.json"):
        try:
            order = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        pop_files.append((order, path))
    if not pop_files:
        return None
    return max(pop_files, key=lambda item: item[0])[1]


def load_final_records(log_dir: str | Path) -> list[dict[str, Any]]:
    log_dir = Path(log_dir)
    pop_file = _latest_population_file(log_dir)
    if pop_file is not None:
        return _load_json_records(pop_file)

    sample_dir = log_dir / "samples"
    records = []
    if sample_dir.exists():
        for path in sorted(sample_dir.glob("*.json")):
            records.extend(_load_json_records(path))
    return records


def _score_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    scores = []
    for record in records:
        score = _score_vector(record.get("score"))
        if score is not None:
            scores.append(score)
    if not scores:
        return np.empty((0, 0), dtype=float)
    dim = min(len(score) for score in scores)
    return np.array([score[:dim] for score in scores], dtype=float)


def _is_dominated(point: np.ndarray, other: np.ndarray) -> bool:
    return bool(np.all(other <= point) and np.any(other < point))


def nondominated_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    keep = []
    for i, point in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i != j and _is_dominated(point, other):
                dominated = True
                break
        if not dominated:
            keep.append(point)
    if not keep:
        return np.empty((0, points.shape[1]), dtype=float)
    return np.unique(np.array(keep, dtype=float), axis=0)


def default_reference_point(points: np.ndarray) -> np.ndarray:
    max_vals = np.max(points, axis=0)
    min_vals = np.min(points, axis=0)
    span = max_vals - min_vals
    margin = np.where(span > 0, 0.1 * span, 1.0)
    return max_vals + margin


def hypervolume_minimization(points: np.ndarray, reference_point: np.ndarray | None = None) -> float:
    """Exact dominated hypervolume for small minimization fronts.

    The implementation recursively sweeps the first objective. It is intended
    for the small final heuristic populations written by the method profilers.
    """
    if points.size == 0:
        return 0.0
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")
    if reference_point is None:
        reference_point = default_reference_point(points)
    reference_point = np.asarray(reference_point, dtype=float)

    valid = np.all(points < reference_point, axis=1)
    points = points[valid]
    if len(points) == 0:
        return 0.0

    points = nondominated_points(points)
    dim = points.shape[1]
    if dim == 1:
        return max(0.0, float(reference_point[0] - np.min(points[:, 0])))

    coords = sorted(set(float(x) for x in points[:, 0] if x < reference_point[0]))
    volume = 0.0
    for idx, coord in enumerate(coords):
        next_coord = coords[idx + 1] if idx + 1 < len(coords) else float(reference_point[0])
        width = max(0.0, next_coord - coord)
        if width <= 0:
            continue
        active = points[points[:, 0] <= coord + 1e-12][:, 1:]
        volume += width * hypervolume_minimization(active, reference_point[1:])
    return float(volume)


def _safe_metric(value: float | None):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = _score_matrix(records)
    summary: dict[str, Any] = {
        "num_records": len(records),
        "num_valid_scores": int(len(scores)),
        "score_dim": int(scores.shape[1]) if scores.size else 0,
    }
    if scores.size == 0:
        return summary

    inner_hv = -scores[:, 0]
    summary.update(
        {
            "best_inner_hv": _safe_metric(float(np.max(inner_hv))),
            "mean_inner_hv": _safe_metric(float(np.mean(inner_hv))),
            "best_quality_loss": _safe_metric(float(np.min(scores[:, 0]))),
        }
    )

    if scores.shape[1] >= 2:
        qt_points = scores[:, :2]
        qt_ref = default_reference_point(qt_points)
        summary.update(
            {
                "best_runtime": _safe_metric(float(np.min(scores[:, 1]))),
                "outer_hv_quality_runtime": _safe_metric(hypervolume_minimization(qt_points, qt_ref)),
                "outer_ref_quality_runtime": qt_ref.tolist(),
                "nondominated_quality_runtime": int(len(nondominated_points(qt_points))),
            }
        )

    if scores.shape[1] >= 3:
        all_ref = default_reference_point(scores)
        novelty = -scores[:, 2]
        summary.update(
            {
                "best_code_novelty": _safe_metric(float(np.max(novelty))),
                "mean_code_novelty": _safe_metric(float(np.mean(novelty))),
                "outer_hv_all_objectives": _safe_metric(hypervolume_minimization(scores, all_ref)),
                "outer_ref_all_objectives": all_ref.tolist(),
                "nondominated_all_objectives": int(len(nondominated_points(scores))),
            }
        )
    else:
        summary.update(
            {
                "outer_hv_all_objectives": summary.get("outer_hv_quality_runtime"),
                "outer_ref_all_objectives": summary.get("outer_ref_quality_runtime"),
                "nondominated_all_objectives": summary.get("nondominated_quality_runtime"),
            }
        )

    return summary


def summarize_log_dir(log_dir: str | Path) -> dict[str, Any]:
    log_dir = Path(log_dir)
    records = load_final_records(log_dir)
    summary = summarize_records(records)
    summary["log_dir"] = str(log_dir)
    summary["method"] = log_dir.name.split("_")[-1] if "_" in log_dir.name else log_dir.name
    return summary


def write_report(log_dir: str | Path) -> dict[str, Any]:
    log_dir = Path(log_dir)
    summary = summarize_log_dir(log_dir)

    report_json = log_dir / "metrics_report.json"
    with report_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    report_md = log_dir / "metrics_report.md"
    with report_md.open("w", encoding="utf-8") as file:
        file.write("# Metrics Report\n\n")
        for key, value in summary.items():
            file.write(f"- `{key}`: {value}\n")

    return summary


def _format_table(summaries: list[dict[str, Any]]) -> str:
    columns = [
        "method",
        "num_valid_scores",
        "score_dim",
        "best_inner_hv",
        "mean_inner_hv",
        "best_runtime",
        "outer_hv_quality_runtime",
        "outer_hv_all_objectives",
        "best_code_novelty",
    ]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, sep]
    for summary in summaries:
        values = []
        for col in columns:
            value = summary.get(col)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append("" if value is None else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Report inner/outer HV metrics for LLMPFG logs.")
    parser.add_argument("log_dirs", nargs="+", help="Method log directories to summarize.")
    parser.add_argument("--write", action="store_true", help="Write metrics_report.json/md into each log directory.")
    args = parser.parse_args()

    summaries = []
    for log_dir in args.log_dirs:
        summary = write_report(log_dir) if args.write else summarize_log_dir(log_dir)
        summaries.append(summary)
    print(_format_table(summaries))


if __name__ == "__main__":
    main()
