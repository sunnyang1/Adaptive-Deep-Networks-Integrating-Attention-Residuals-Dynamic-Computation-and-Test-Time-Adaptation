"""Quality scoring for QASP adaptation decisions."""

from __future__ import annotations

import torch


def compute_quality_score(
    signal: torch.Tensor,
    low_pass_ratio: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute a quality score in [0, 1] from frequency utility statistics.

    The score combines:
    1) Per-sample low-pass FFT energy ratio.
    2) Batch-mean utility (global quality context).
    """

    if low_pass_ratio <= 0.0 or low_pass_ratio > 1.0:
        raise ValueError("`low_pass_ratio` must be in (0, 1].")

    if signal.ndim == 1:
        signal = signal.unsqueeze(0)
    if signal.ndim < 2:
        raise ValueError("`signal` must have shape [batch, ...].")

    batch = signal.shape[0]
    flattened = signal.reshape(batch, -1)

    fft_vals = torch.fft.rfft(flattened, dim=-1)
    power = fft_vals.real.square() + fft_vals.imag.square()
    total_power = power.sum(dim=-1).clamp_min(eps)

    freq_bins = power.shape[-1]
    cutoff = max(1, int(freq_bins * low_pass_ratio))
    low_power = power[..., :cutoff].sum(dim=-1)

    low_pass_utility = (low_power / total_power).clamp(0.0, 1.0)
    batch_mean_utility = low_pass_utility.mean()
    quality = 0.5 * low_pass_utility + 0.5 * batch_mean_utility
    return quality.clamp(0.0, 1.0)

