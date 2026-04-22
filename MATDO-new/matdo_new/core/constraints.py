from __future__ import annotations

import math


def clamp_ratio(value: float, lower: float = 0.0, upper: float = 0.999_999) -> float:
    """Clamp utilization-like values into a safe range."""
    if not math.isfinite(value):
        raise ValueError("ratio must be finite")
    return min(upper, max(lower, value))


def positive_int(value: float | int, minimum: int = 1) -> int:
    """Round to an integer while preserving a positive lower bound."""
    if not math.isfinite(float(value)):
        raise ValueError("value must be finite")
    return max(minimum, int(round(value)))


def available_scope_blocks(rho_hbm: float, total_blocks: int) -> int:
    """Translate HBM utilization into a positive number of live scope blocks."""
    usable_fraction = 1.0 - clamp_ratio(rho_hbm)
    return positive_int(total_blocks * usable_fraction, minimum=1)
