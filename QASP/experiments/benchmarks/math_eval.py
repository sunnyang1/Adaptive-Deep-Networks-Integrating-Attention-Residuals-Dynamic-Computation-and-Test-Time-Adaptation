"""Lightweight math benchmark surrogate for QASP quick runs."""

from __future__ import annotations

import torch


def run_math_eval(quick: bool = True) -> float:
    """Return a deterministic proxy math score for quick experimentation."""
    torch.manual_seed(11 if quick else 19)
    base = torch.tensor([0.49, 0.52, 0.50, 0.53], dtype=torch.float32)
    jitter = (torch.rand_like(base) - 0.5) * (0.015 if quick else 0.03)
    score = (base + jitter).clamp(min=0.0, max=1.0).mean().item()
    return float(score)

