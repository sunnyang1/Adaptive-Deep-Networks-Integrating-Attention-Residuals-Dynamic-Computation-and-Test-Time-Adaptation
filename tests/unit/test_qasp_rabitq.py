"""Unit tests for the RaBitQ 1-bit KV codec (paper §5.5)."""

from __future__ import annotations

import math

import pytest
import torch

from QASP.inference.rabitq import RaBitQCodec


def test_rotation_buffer_is_orthonormal() -> None:
    codec = RaBitQCodec(dim=16, seed=0)
    q = codec.rotation
    identity = torch.eye(16)
    assert torch.allclose(q.transpose(-2, -1) @ q, identity, atol=1e-5)


def test_encode_preserves_norm_and_returns_int8_signs() -> None:
    torch.manual_seed(0)
    codec = RaBitQCodec(dim=32, seed=1)
    x = torch.randn(4, 32)

    signs, norms = codec.encode(x)

    assert signs.dtype == torch.int8
    assert set(signs.unique().tolist()).issubset({-1, 1})
    assert torch.allclose(norms, x.norm(dim=-1), atol=1e-5)


def test_decode_preserves_norm_and_direction() -> None:
    torch.manual_seed(0)
    codec = RaBitQCodec(dim=64, seed=2)
    x = torch.randn(8, 64) * 3.5

    x_hat = codec.quantize(x)

    assert torch.allclose(x_hat.norm(dim=-1), x.norm(dim=-1), atol=1e-4)

    cos = (x * x_hat).sum(dim=-1) / (x.norm(dim=-1) * x_hat.norm(dim=-1))
    assert (cos > 0.5).all()


def test_encode_rejects_wrong_last_dim() -> None:
    codec = RaBitQCodec(dim=8)
    with pytest.raises(ValueError):
        codec.encode(torch.randn(2, 7))


def test_decode_rejects_mismatched_shapes() -> None:
    codec = RaBitQCodec(dim=8)
    signs = torch.ones(2, 3, 8, dtype=torch.int8)
    bad_norms = torch.ones(2, 4)
    with pytest.raises(ValueError):
        codec.decode(signs, bad_norms)


def test_reconstruction_scale_matches_closed_form() -> None:
    """When signs perfectly agree with a rotated axis, decode must recover the axis."""

    codec = RaBitQCodec(dim=4, seed=5)
    rotated = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    x = rotated @ codec.rotation.transpose(0, 1)
    x_hat = codec.quantize(x)

    expected_norm = rotated.norm(dim=-1)
    assert torch.allclose(x_hat.norm(dim=-1), expected_norm, atol=1e-5)

    scale = expected_norm / math.sqrt(codec.dim)
    expected_rotated = torch.ones(1, 4) * scale
    assert torch.allclose((x_hat @ codec.rotation), expected_rotated, atol=1e-5)
