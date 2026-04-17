"""Ponder gate for deciding whether adaptation should be triggered."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class PonderGate:
    """Gate adaptation using entropy and confidence heuristics."""

    entropy_threshold: float = 0.7
    confidence_threshold: float = 0.4
    eps: float = 1e-8

    def should_adapt(self, logits: torch.Tensor) -> bool:
        """Return True when outputs are uncertain and low-confidence."""

        if logits.ndim < 2:
            raise ValueError("`logits` must be at least 2D [..., vocab].")

        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(self.eps))).sum(dim=-1)
        max_entropy = torch.log(
            torch.tensor(float(logits.shape[-1]), device=logits.device, dtype=logits.dtype)
        ).clamp_min(self.eps)
        normalized_entropy = (entropy / max_entropy).mean()

        confidence = probs.max(dim=-1).values.mean()
        return bool(
            normalized_entropy >= self.entropy_threshold
            and confidence <= self.confidence_threshold
        )

