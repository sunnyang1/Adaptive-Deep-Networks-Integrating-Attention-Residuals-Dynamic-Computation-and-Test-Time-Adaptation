"""
TurboQuant V3: Refactored Implementation

Clean, modular implementation of TurboQuant V3 compression.
Based on community findings that QJL hurts attention quality.

Quick Start:
    >>> from turboquant import create_k4_v2
    
    >>> # Create compressor
    >>> tq = create_k4_v2(head_dim=64)
    >>> 
    >>> # Fit on sample data
    >>> tq.fit(sample_keys, sample_values)
    >>> 
    >>> # Compress/decompress
    >>> compressed = tq.compress(keys, values)
    >>> keys_dq, values_dq = tq.decompress(compressed)
    >>> 
    >>> # Use as HF cache
    >>> cache = tq.as_cache(residual_window=128)
    >>> model.generate(..., past_key_values=cache)

Recommended Configs:
    - create_k4_v2(): 4-bit K, 2-bit V (recommended, ~4.9x compression)
    - create_k3_v2(): 3-bit K, 2-bit V (~3.0x compression)
    - create_k2_v2(): 2-bit K, 2-bit V (~7.1x compression, max memory)
"""

# ============================================================================
# Core API (Recommended)
# ============================================================================

from .api import (
    TurboQuantV3,
    TurboQuantConfig,
    create_k4_v2,
    create_k3_v2,
    create_k2_v2,
    RECOMMENDED,
)

# ============================================================================
# Low-level Components
# ============================================================================

from .rotation import (
    RandomRotation,
    fwht,
    fwht_inverse,
)

from .quantizer import (
    LloydMaxQuantizer,
)

from .compressor import (
    MSECompressor,
    CompressorConfig,
    pack_bits,
    unpack_bits,
)

from .cache import (
    V3Cache,
    CacheConfig,
)

# ============================================================================
# Legacy Exports (Backward Compatibility)
# ============================================================================

# Keep old V3 imports working
from .v3_improved import (
    TurboQuantV3 as LegacyTurboQuantV3,
    TurboQuantV3Config as LegacyTurboQuantV3Config,
    create_v3_k4_v2 as legacy_create_v3_k4_v2,
    create_v3_k3_v2 as legacy_create_v3_k3_v2,
    MSECompressor as LegacyMSECompressor,
    MSECompressorConfig as LegacyMSECompressorConfig,
    LloydMaxQuantizerV3 as LegacyLloydMaxQuantizerV3,
    RandomRotation as LegacyRandomRotation,
    pack_bits as legacy_pack_bits,
    unpack_bits as legacy_unpack_bits,
)

# Keep old unified API working
from .core import (
    TurboQuant,
    TurboQuantConfig as LegacyTurboQuantConfig,
    QuantMode,
)

# ============================================================================
# Version
# ============================================================================

__version__ = '3.0.0'

__all__ = [
    # New API (Recommended)
    'TurboQuantV3',
    'TurboQuantConfig',
    'create_k4_v2',
    'create_k3_v2',
    'create_k2_v2',
    'RECOMMENDED',
    
    # Low-level Components
    'RandomRotation',
    'fwht',
    'fwht_inverse',
    'LloydMaxQuantizer',
    'MSECompressor',
    'CompressorConfig',
    'pack_bits',
    'unpack_bits',
    'V3Cache',
    'CacheConfig',
    
    # Legacy (Backward Compatibility)
    'TurboQuant',
    'LegacyTurboQuantConfig',
    'QuantMode',
    'LegacyTurboQuantV3',
    'LegacyTurboQuantV3Config',
    'legacy_create_v3_k4_v2',
    'legacy_create_v3_k3_v2',
]
