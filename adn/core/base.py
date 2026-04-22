"""
Base components for Adaptive Deep Networks.

Includes RMSNorm, SwiGLU activation, and BaseModule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm normalizes inputs by their root mean square instead of mean and variance,
    providing better stability for deep networks without the mean-centering overhead
    of LayerNorm.

    Formula:
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Args:
        dim: Dimension to normalize over
        eps: Small constant for numerical stability (default: 1e-6)

    Attributes:
        weight: Learnable scale parameter of shape [dim]

    Example:
        >>> import torch
        >>> norm = RMSNorm(dim=512)
        >>> x = torch.randn(2, 10, 512)
        >>> normalized = norm(x)
        >>> normalized.shape
        torch.Size([2, 10, 512])

    References:
        Zhang and Sennrich (2019): "Root Mean Square Layer Normalization"
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.

        Computes in float32 for numerical stability and casts back to the
        input dtype. This is important for mixed-precision training.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor of shape [..., dim]
        """
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.

    SwiGLU is a gated activation function that combines SiLU (Swish)
    with a gating mechanism. It uses three linear projections:
    - gate_proj: Gating projection
    - up_proj: Value projection
    - down_proj: Output projection

    Formula:
        SwiGLU(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    Args:
        hidden_dim: Input/output dimension
        mlp_ratio: Expansion ratio for MLP hidden dimension (default: 4)
        dropout: Dropout probability (default: 0.0)

    Example:
        >>> import torch
        >>> swiglu = SwiGLU(hidden_dim=512, mlp_ratio=4)
        >>> x = torch.randn(2, 10, 512)
        >>> out = swiglu(x)
        >>> out.shape
        torch.Size([2, 10, 512])

    References:
        Shazeer (2020): "GLU Variants Improve Transformer"
    """

    def __init__(self, hidden_dim: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        mlp_dim = hidden_dim * mlp_ratio

        self.gate_proj = nn.Linear(hidden_dim, mlp_dim)
        self.up_proj = nn.Linear(hidden_dim, mlp_dim)
        self.down_proj = nn.Linear(mlp_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.

        Args:
            x: Input tensor [..., hidden_dim]

        Returns:
            Output tensor [..., hidden_dim]
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        out = self.down_proj(hidden)
        return out


class BaseModule(nn.Module):
    """
    Base module for ADN components.

    Provides common functionality for all ADN modules including:
    - Standard weight initialization
    - Parameter counting utilities
    - Device management helpers
    """

    def __init__(self):
        super().__init__()

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_module(self) -> dict:
        """Count parameters by sub-module."""
        counts = {}
        for name, module in self.named_children():
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return counts
