"""True incremental prefill / step API for QASP models.

``prefill`` runs the full forward pass once while snapshotting per-layer K/V
caches. ``step`` then does a 1-token forward whose cost is O(L · T) (one new
query attending over the cached keys), matching the paper's §5.4 promise of an
O(L) per-token inner loop when the cache is reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from QASP.inference.kv_cache import KVCache


class _IncrementalModel(Protocol):
    """Structural type for models that expose incremental prefill/step APIs."""

    def prefill(self, input_ids: Tensor) -> tuple[Tensor, KVCache]: ...

    def step(self, last_token: Tensor, cache: KVCache) -> Tensor: ...

    def eval(self) -> "_IncrementalModel": ...


@dataclass
class IncrementalState:
    """Mutable inference state carried across decoding steps."""

    cache: KVCache
    next_logits: Tensor

    @property
    def seq_len(self) -> int:
        return self.cache.seq_len


class IncrementalInference:
    """Incremental wrapper that relies on the model's prefill/step API."""

    def __init__(self, model: _IncrementalModel) -> None:
        if not hasattr(model, "prefill") or not hasattr(model, "step"):
            raise TypeError(
                "`model` must expose `prefill(input_ids)` and "
                "`step(last_token, cache)` (see QASPTransformer)."
            )
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def prefill(self, input_ids: Tensor) -> IncrementalState:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")

        logits, cache = self.model.prefill(input_ids)
        return IncrementalState(cache=cache, next_logits=logits[:, -1, :])

    @torch.no_grad()
    def step(self, state: IncrementalState) -> Tensor:
        """Emit one token from ``state.next_logits`` and advance the cache."""

        if state.next_logits.ndim != 2:
            raise ValueError("state.next_logits must have shape [B, vocab_size]")

        next_token = torch.argmax(state.next_logits, dim=-1, keepdim=True)
        state.next_logits = self.model.step(next_token, state.cache)
        return next_token
