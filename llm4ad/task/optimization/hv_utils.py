from __future__ import annotations

import math

import numpy as np


def reference_hypervolume_scale(ref_point: np.ndarray) -> float:
    """Scale factor for normalizing per-instance hypervolume values."""
    ref_point = np.asarray(ref_point, dtype=float)
    scale = float(np.prod(np.maximum(np.abs(ref_point), 1e-12)))
    if not math.isfinite(scale) or scale <= 0:
        return 1.0
    return scale


def scale_hypervolume(hv_value: float, ref_point: np.ndarray) -> float:
    return float(hv_value) / reference_hypervolume_scale(ref_point)
