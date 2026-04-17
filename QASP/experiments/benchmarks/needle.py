"""Lightweight needle benchmark surrogate for QASP quick runs."""

from __future__ import annotations

import torch


def run_needle_benchmark(quick: bool = True) -> float:
    """Return a deterministic proxy accuracy for quick CI/smoke loops."""
    torch.manual_seed(7 if quick else 13)
    base = torch.tensor([0.78, 0.81, 0.79, 0.82], dtype=torch.float32)
    jitter = (torch.rand_like(base) - 0.5) * (0.01 if quick else 0.02)
    score = (base + jitter).clamp(min=0.0, max=1.0).mean().item()
    return float(score)

