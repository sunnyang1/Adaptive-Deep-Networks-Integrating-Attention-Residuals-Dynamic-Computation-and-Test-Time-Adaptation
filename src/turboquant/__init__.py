"""
TurboQuant: Data-Oblivious Extreme Compression

Based on: TurboQuant (ICLR 2026)
Two-stage quantization pipeline:
1. PolarQuant: (b-1)-bit compression via random Hadamard + polar coordinates
2. QJL: 1-bit Johnson-Lindenstrauss residual correction

Key Features:
- Data-oblivious: No calibration data, no fine-tuning required
- 6×+ memory reduction with zero accuracy loss
- 8× throughput increase on Tensor Cores via 4-bit INT kernels
"""

from .polar_quant import PolarQuant, CartesianToPolar
from .qjl import QJLCompressor, QJLDecompressor
from .turbo_quant import TurboQuantPipeline, TurboQuantConfig
from .tensor_core import TensorCoreKernel, INT4Linear

# MNN-inspired improvements
from .mnn_improved import (
    MNNTurboQuantConfig,
    MNNTurboQuantCompressor,
    AttentionMode,
    KVQuantMode,
    LloydMaxQuantizer,
    create_mnn_turboquant,
    CONFIG_RECOMMENDATIONS,
)

__all__ = [
    # Original TurboQuant
    'PolarQuant',
    'CartesianToPolar',
    'QJLCompressor',
    'QJLDecompressor',
    'TurboQuantPipeline',
    'TurboQuantConfig',
    'TensorCoreKernel',
    'INT4Linear',
    # MNN-inspired improvements
    'MNNTurboQuantConfig',
    'MNNTurboQuantCompressor',
    'AttentionMode',
    'KVQuantMode',
    'LloydMaxQuantizer',
    'create_mnn_turboquant',
    'CONFIG_RECOMMENDATIONS',
]
