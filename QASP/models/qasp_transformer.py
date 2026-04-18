"""Top-level QASP transformer model and factory helper."""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

import torch
import torch.nn as nn
from torch import Tensor

from QASP.adaptation.matrix_qasp import matrix_qasp_update
from QASP.adaptation.ponder_gate import PonderGate
from QASP.configs.qasp import QASPConfig
from QASP.inference.kv_cache import KVCache
from QASP.inference.rabitq import RaBitQCodec
from QASP.models.components import QASPTransformerConfig, RMSNorm, compute_block_representations
from QASP.models.ngram_memory import NgramMemory
from QASP.models.qasp_layer import QASPLayer


LayerLossFn = Callable[[int, Tensor], Tensor]


class QASPTransformer(nn.Module):
    """Minimal runnable transformer with QASP AttnRes / Engram hooks."""

    def __init__(self, config: QASPTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([QASPLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.engram_memory: NgramMemory | None
        if config.use_engram:
            self.engram_memory = NgramMemory(
                table_size=config.engram_table_size,
                hidden_size=config.hidden_size,
                n_gram=config.engram_n_gram,
            )
        else:
            self.engram_memory = None

        self.kv_codec: RaBitQCodec | None
        if config.quantize_kv:
            if config.hidden_size % config.num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads for KV quantization.")
            head_dim = config.hidden_size // config.num_heads
            self.kv_codec = RaBitQCodec(head_dim, seed=config.kv_codec_seed)
        else:
            self.kv_codec = None

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError("input sequence length exceeds max_position_embeddings")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)

        memory_vector: Tensor | None = None
        memory_quality: Tensor | None = None
        if self.config.use_engram and self.engram_memory is not None:
            memory_vector, memory_quality = self.engram_memory.batch_lookup(input_ids)

        for module in self.layers:
            layer = cast(QASPLayer, module)
            block_repr = block_quality = None
            if self.config.use_attnres:
                block_repr, block_quality = compute_block_representations(
                    hidden_states,
                    num_blocks=self.config.attnres_blocks,
                )

            hidden_states = layer(
                hidden_states,
                block_representations=block_repr,
                block_quality=block_quality,
                memory_vector=memory_vector,
                memory_quality=memory_quality,
            )

        hidden_states = self.norm(hidden_states)
        return cast(Tensor, self.lm_head(hidden_states))

    @torch.no_grad()
    def prefill(self, input_ids: Tensor) -> tuple[Tensor, KVCache]:
        """Run the full forward pass while capturing per-layer K/V caches.

        Returns ``(logits, cache)`` where ``logits`` has the same shape as
        :meth:`forward` (``[B, T, V]``) and ``cache`` is a populated
        :class:`KVCache` ready for O(L) :meth:`step` calls.
        """

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError("input sequence length exceeds max_position_embeddings")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)

        memory_vector = memory_quality = None
        if self.config.use_engram and self.engram_memory is not None:
            memory_vector, memory_quality = self.engram_memory.batch_lookup(input_ids)

        cache = KVCache.from_input_ids(input_ids, num_layers=len(self.layers))

        for idx, module in enumerate(self.layers):
            layer = cast(QASPLayer, module)
            cache.layer_inputs[idx] = hidden_states

            block_repr = block_quality = None
            if self.config.use_attnres:
                block_repr, block_quality = compute_block_representations(
                    hidden_states,
                    num_blocks=self.config.attnres_blocks,
                )

            hidden_states, k, v = layer.forward_with_cache(
                hidden_states,
                block_representations=block_repr,
                block_quality=block_quality,
                memory_vector=memory_vector,
                memory_quality=memory_quality,
                kv_codec=self.kv_codec,
            )
            cache.layer_keys[idx] = k
            cache.layer_values[idx] = v

        hidden_states = self.norm(hidden_states)
        logits = cast(Tensor, self.lm_head(hidden_states))
        return logits, cache

    @torch.no_grad()
    def step(self, last_token: Tensor, cache: KVCache) -> Tensor:
        """Incrementally decode one token using cached K/V; returns ``[B, V]`` logits."""

        if last_token.ndim != 2 or last_token.shape[1] != 1:
            raise ValueError("last_token must have shape [B, 1]")
        if cache.num_layers != len(self.layers):
            raise ValueError("cache was not initialised for this model's layer count")
        if last_token.shape[0] != cache.batch_size:
            raise ValueError("last_token batch size must match cache batch size")

        new_position = cache.seq_len
        if new_position + 1 > self.config.max_position_embeddings:
            raise ValueError("cache length would exceed max_position_embeddings")

        cache.append(last_token)

        batch_size = last_token.size(0)
        pos_idx = torch.full(
            (batch_size, 1),
            new_position,
            dtype=torch.long,
            device=last_token.device,
        )
        hidden = self.token_embedding(last_token) + self.position_embedding(pos_idx)

        memory_vec_new: Optional[Tensor] = None
        memory_qual_new: Optional[Tensor] = None
        if self.config.use_engram and self.engram_memory is not None:
            mem_vec_full, mem_qual_full = self.engram_memory.batch_lookup(cache.input_ids)
            memory_vec_new = mem_vec_full[:, -1:, :]
            memory_qual_new = mem_qual_full[:, -1:]

        for idx, module in enumerate(self.layers):
            layer = cast(QASPLayer, module)
            previous_inputs = cache.layer_inputs[idx]
            if previous_inputs is None:
                layer_input_history = hidden
            else:
                layer_input_history = torch.cat([previous_inputs, hidden], dim=1)
            cache.layer_inputs[idx] = layer_input_history

            block_repr = block_quality = None
            if self.config.use_attnres:
                block_repr, block_quality = compute_block_representations(
                    layer_input_history,
                    num_blocks=self.config.attnres_blocks,
                )

            hidden, new_k, new_v = layer.step(
                hidden,
                cached_k=cache.layer_keys[idx],
                cached_v=cache.layer_values[idx],
                block_representations=block_repr,
                block_quality=block_quality,
                memory_vector=memory_vec_new,
                memory_quality=memory_qual_new,
                kv_codec=self.kv_codec,
            )
            cache.layer_keys[idx] = new_k
            cache.layer_values[idx] = new_v

        hidden = self.norm(hidden)
        return cast(Tensor, self.lm_head(hidden).squeeze(1))

    def adapt_at_test_time(
        self,
        loss_fn_for_layer: LayerLossFn,
        logits: Tensor,
        quality_scores: Optional[Tensor] = None,
        qasp_config: Optional[QASPConfig] = None,
    ) -> bool:
        """Run the paper's ponder-gated QASP update on every layer's ``W_ℓ``.

        Returns ``True`` if the ponder gate fired and adaptation ran,
        ``False`` otherwise. When it runs, each ``layer.stiefel_query`` is
        replaced with the Stiefel-projected result of
        :func:`QASP.adaptation.matrix_qasp.matrix_qasp_update`.
        """

        cfg = qasp_config or QASPConfig()
        gate = PonderGate(
            entropy_threshold=cfg.entropy_threshold,
            confidence_threshold=cfg.confidence_threshold,
        )
        if not gate.should_adapt(logits):
            return False

        for idx, module in enumerate(self.layers):
            layer = cast(QASPLayer, module)
            current = cast(Tensor, layer.stiefel_query.data)

            def layer_loss(w: Tensor, _idx: int = idx) -> Tensor:
                return loss_fn_for_layer(_idx, w)

            updated = matrix_qasp_update(
                matrix=current,
                loss_fn=layer_loss,
                quality_scores=quality_scores,
                step_size=cfg.step_size,
                num_adapt_steps=cfg.num_adapt_steps,
                ns_iters=cfg.ns_iters,
                eps=cfg.epsilon,
            )
            cast(Tensor, layer.stiefel_query.data).copy_(updated)
        return True


def create_qasp_transformer(**kwargs: Any) -> QASPTransformer:
    """Factory helper for quick model creation in tests and experiments."""

    config = QASPTransformerConfig(**kwargs)
    return QASPTransformer(config)
