"""Unit tests for QASP quality score computation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QASP.adaptation.quality_score import compute_quality_score


def test_quality_scores_are_bounded_in_unit_interval() -> None:
    """Quality scores must always stay in [0, 1]."""

    torch.manual_seed(0)
    signal = torch.randn(6, 128)

    quality = compute_quality_score(signal, low_pass_ratio=0.25)

    assert quality.shape == (6,)
    assert torch.all(quality >= 0.0)
    assert torch.all(quality <= 1.0)


def test_smoother_signal_has_higher_quality_than_noisy_signal() -> None:
    """Low-frequency signal should score higher than high-frequency noise."""

    t = torch.linspace(0.0, 2.0 * torch.pi, 256)
    smooth = torch.sin(t).repeat(4, 1)
    noisy = torch.randn(4, 256)

    smooth_quality = compute_quality_score(smooth, low_pass_ratio=0.2).mean()
    noisy_quality = compute_quality_score(noisy, low_pass_ratio=0.2).mean()

    assert smooth_quality > noisy_quality

