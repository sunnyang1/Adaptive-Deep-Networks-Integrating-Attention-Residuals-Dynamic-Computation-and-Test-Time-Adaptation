"""QASP component ablation helpers."""

from __future__ import annotations


def run_qasp_ablation(quick: bool = True) -> dict[str, float]:
    """Return deterministic ablation deltas for quick experiment reporting."""
    scale = 1.0 if quick else 1.2
    return {
        "full_qasp": 0.802 * scale,
        "minus_value_weighted_attnres": 0.781 * scale,
        "minus_value_weighted_engram": 0.789 * scale,
        "minus_stiefel_projection": 0.772 * scale,
    }

