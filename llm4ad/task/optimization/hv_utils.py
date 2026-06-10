from __future__ import annotations

import math

import numpy as np

DEFAULT_SEARCH_ITERATIONS = 2000


def reference_hypervolume_scale(
    ref_point: np.ndarray,
    ideal_point: np.ndarray | None = None,
) -> float:
    """Scale factor for normalizing per-instance hypervolume values."""
    ref_point = np.asarray(ref_point, dtype=float)
    if ideal_point is None:
        spans = np.abs(ref_point)
    else:
        ideal_point = np.asarray(ideal_point, dtype=float)
        if ideal_point.shape != ref_point.shape or not np.all(np.isfinite(ideal_point)):
            return 1.0
        spans = ref_point - ideal_point
        if np.any(spans <= 0):
            return 1.0
    scale = float(np.prod(np.maximum(spans, 1e-12)))
    if not math.isfinite(scale) or scale <= 0:
        return 1.0
    return scale


def scale_hypervolume(
    hv_value: float,
    ref_point: np.ndarray,
    ideal_point: np.ndarray | None = None,
) -> float:
    return float(hv_value) / reference_hypervolume_scale(ref_point, ideal_point)
