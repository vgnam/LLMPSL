from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.registry import PROBLEM_CONFIGS, build_problem, normalize_problem_name
from llm4ad.tools.evaluate_all_sizes import _select_records
from llm4ad.tools.evaluate_population_front import (
    _as_points,
    _evaluate_trace_record,
    _nondominated_min,
)
from llm4ad.tools.report_metrics import load_final_records


DEFAULT_LABELS = {
    "llmpfg": "LLMPFG",
    "mpage": "LLMPFG",
    "pbcllm": "PBCLLM",
}


def _merge_population_front(trace_results: list[dict[str, Any]], n_instance: int) -> np.ndarray:
    merged = []
    for instance_idx in range(n_instance):
        instance_points = []
        for item in trace_results:
            if item.get("status") != "ok":
                continue
            fronts = item.get("fronts") or []
            if instance_idx >= len(fronts):
                continue
            points = _as_points(fronts[instance_idx])
            if points.size:
                instance_points.append(points)
        if instance_points:
            merged.append(_nondominated_min(np.vstack(instance_points)))
    if not merged:
        return np.empty((0, 0), dtype=float)
    return _nondominated_min(np.vstack(merged))


def _evaluate_log_front(
    log_dir: Path,
    *,
    problem: str,
    problem_size: int | None,
    n_instance: int | None,
    seed: int,
    eval_seed: int | None,
    eval_repeats: int = 1,
    search_iterations: int | None = None,
    timeout_seconds: int | None,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if eval_repeats < 1:
        raise ValueError("eval_repeats must be at least 1.")
    if search_iterations is not None and search_iterations < 1:
        raise ValueError("search_iterations must be at least 1.")

    records = load_final_records(log_dir)
    selected = _select_records(records, top_k=top_k)
    eval_seeds = [None] * eval_repeats if eval_seed is None else [eval_seed + idx for idx in range(eval_repeats)]
    trace_results = []
    evaluator = None
    for repeat_seed in eval_seeds:
        evaluator = build_problem(
            problem,
            n_instance=n_instance,
            problem_size=problem_size,
            seed=seed,
            eval_seed=repeat_seed,
            timeout_seconds=timeout_seconds,
        )
        if search_iterations is not None:
            if not hasattr(evaluator, "search_iterations"):
                raise ValueError(f"{problem} does not support a plot-only search iteration override.")
            evaluator.search_iterations = search_iterations
        setattr(evaluator, "return_mo_trace", True)
        evaluator.safe_evaluate = False
        trace_results.extend(_evaluate_trace_record(record, evaluator) for record in selected)

    if evaluator is None:
        raise RuntimeError("No evaluator was created.")

    front = _merge_population_front(trace_results, evaluator.n_instance)
    ref_point = np.asarray(getattr(evaluator, "ref_point"), dtype=float)

    summary = {
        "log_dir": str(log_dir),
        "num_records": len(records),
        "num_selected": len(selected),
        "num_valid": sum(1 for item in trace_results if item.get("status") == "ok"),
        "eval_repeats": eval_repeats,
        "eval_seeds": eval_seeds,
        "search_iterations": getattr(evaluator, "search_iterations", None),
        "problem_size": evaluator.problem_size,
        "n_instance": evaluator.n_instance,
        "ref_point": ref_point.tolist(),
        "front_points": int(len(front)) if front.size else 0,
    }
    return front, ref_point, summary


def _plot_fronts(fronts: dict[str, np.ndarray], ref_point: np.ndarray, output: Path, title: str):
    if output.suffix.lower() == ".svg":
        _plot_fronts_svg(fronts, ref_point, output, title)
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        svg_output = output.with_suffix(".svg")
        _plot_fronts_svg(fronts, ref_point, svg_output, title)
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=160)
    colors = {
        "LLMPFG": "#2563eb",
        "PBCLLM": "#dc2626",
    }
    markers = {
        "LLMPFG": "o",
        "PBCLLM": "s",
    }

    for label, front in fronts.items():
        if front.size == 0:
            continue
        normalized = front / ref_point
        normalized = normalized[np.argsort(normalized[:, 0])]
        ax.scatter(
            normalized[:, 0],
            normalized[:, 1],
            s=24,
            alpha=0.78,
            marker=markers.get(label, "o"),
            color=colors.get(label),
            label=f"{label} ({len(normalized)} pts)",
        )
        ax.plot(
            normalized[:, 0],
            normalized[:, 1],
            linewidth=1.4,
            alpha=0.65,
            color=colors.get(label),
        )

    ax.set_title(title)
    ax.set_xlabel("Objective 1 / reference point")
    ax.set_ylabel("Objective 2 / reference point")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _plot_fronts_svg(fronts: dict[str, np.ndarray], ref_point: np.ndarray, output: Path, title: str):
    width = 900
    height = 640
    left = 90
    right = 30
    top = 70
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {
        "LLMPFG": "#2563eb",
        "PBCLLM": "#dc2626",
    }

    normalized_fronts = {}
    all_points = []
    for label, front in fronts.items():
        if front.size == 0:
            continue
        normalized = front / ref_point
        normalized = normalized[np.argsort(normalized[:, 0])]
        normalized_fronts[label] = normalized
        all_points.append(normalized)
    if not all_points:
        raise RuntimeError("No valid front points to plot.")

    points = np.vstack(all_points)
    x_min = max(0.0, float(np.min(points[:, 0])) * 0.95)
    x_max = float(np.max(points[:, 0])) * 1.05
    y_min = max(0.0, float(np.min(points[:, 1])) * 0.95)
    y_max = float(np.max(points[:, 1])) * 1.05
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    def sx(x):
        return left + (float(x) - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return top + (y_max - float(y)) / (y_max - y_min) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" fill="#111827">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>',
    ]

    for tick in range(6):
        x_val = x_min + (x_max - x_min) * tick / 5
        x = sx(x_val)
        elements.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e5e7eb" stroke-width="1"/>')
        elements.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#374151">{x_val:.2f}</text>')
        y_val = y_min + (y_max - y_min) * tick / 5
        y = sy(y_val)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        elements.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#374151">{y_val:.2f}</text>')

    elements.append(f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="15" fill="#111827">Objective 1 / reference point</text>')
    elements.append(f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" font-family="Arial" font-size="15" fill="#111827" transform="rotate(-90 24 {top + plot_h / 2})">Objective 2 / reference point</text>')

    legend_y = 58
    legend_x = width - 270
    for idx, (label, normalized) in enumerate(normalized_fronts.items()):
        color = colors.get(label, "#111827")
        path_points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in normalized)
        if path_points:
            elements.append(f'<polyline points="{path_points}" fill="none" stroke="{color}" stroke-width="2" opacity="0.75"/>')
        for x, y in normalized:
            elements.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="{color}" opacity="0.82"/>')
        y = legend_y + idx * 24
        elements.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color}" opacity="0.82"/>')
        elements.append(f'<text x="{legend_x + 14}" y="{y + 5}" font-family="Arial" font-size="14" fill="#111827">{label} ({len(normalized)} pts)</text>')

    elements.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(elements), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Plot Bi-TSP population Pareto fronts for two or more method logs.")
    parser.add_argument("--problem", choices=sorted(PROBLEM_CONFIGS), default="bi_tsp")
    parser.add_argument("--method-log", nargs=2, action="append", metavar=("METHOD", "LOG_DIR"), required=True)
    parser.add_argument("--problem-size", type=int, default=20)
    parser.add_argument("--n-instance", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--eval-seed", type=int, default=2025)
    parser.add_argument(
        "--eval-repeats",
        type=int,
        default=1,
        help="Number of consecutive eval seeds to merge into each plotted front.",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=None,
        help="Plot-only override for the inner search iterations; normal evaluation defaults are unchanged.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    problem = normalize_problem_name(args.problem)
    output = Path(args.output) if args.output else Path("logs") / "plots" / f"{problem}_population_front.png"

    fronts: dict[str, np.ndarray] = {}
    summaries = []
    ref_point = None
    for method, log_dir in args.method_log:
        label = DEFAULT_LABELS.get(method.lower(), method)
        front, method_ref, summary = _evaluate_log_front(
            Path(log_dir),
            problem=problem,
            problem_size=args.problem_size,
            n_instance=args.n_instance,
            seed=args.seed,
            eval_seed=args.eval_seed,
            eval_repeats=args.eval_repeats,
            search_iterations=args.search_iterations,
            timeout_seconds=args.timeout_seconds,
            top_k=args.top_k,
        )
        fronts[label] = front
        summaries.append({"method": label, **summary})
        if ref_point is None:
            ref_point = method_ref

    if ref_point is None:
        raise RuntimeError("No fronts were evaluated.")

    title = (
        f"{problem} population Pareto front, size={args.problem_size}, "
        f"n={args.n_instance}, repeats={args.eval_repeats}"
    )
    if args.search_iterations is not None:
        title += f", iterations={args.search_iterations}"
    _plot_fronts(fronts, ref_point, output, title)

    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps({"plot": str(output), "methods": summaries}, indent=2), encoding="utf-8")
    print(f"Wrote plot: {output}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
