# RaBitQ Refactored - Unified API

## Overview

The RaBitQ module has been refactored to provide a **simple, unified API** while maintaining backward compatibility.

## Quick Start

### Simple Usage

```python
from src.rabitq import RaBitQ

# Create quantizer
quant = RaBitQ('tq4')  # 4-bit RaBitQ

# Fit on sample data (required for TQ modes)
quant.fit(sample_keys, sample_values)

# Compress/decompress
compressed = quant.compress_kv(keys, values)
keys_deq, values_deq = quant.decompress_kv(compressed)
```

### Context Manager

```python
with RaBitQ('tq4', head_dim=128) as quant:
    quant.fit(sample_keys, sample_values)
    compressed = quant.compress_kv(keys, values)
    keys_deq, values_deq = quant.decompress_kv(compressed)
```

## Modes

| Mode | Bits | Compression | Use Case |
|------|------|-------------|----------|
| `'fp16'` | 16 | 1x | No compression (default) |
| `'int8'` | 8 | 2x | Balanced accuracy/size |
| `'tq4'` | 4 | 3x | Fast inference (4B+ models) |
| `'tq3'` | 3 | 4x | Extreme compression (4B+ models) |

### FlashAttention Integration

Add `_flash` suffix to enable FlashAttention:

```python
quant = RaBitQ('tq4_flash')  # TQ4 + FlashAttention
```

## Configuration

### Basic Config

```python
from src.rabitq import RaBitQConfig

config = RaBitQConfig(
    mode='tq4',
    head_dim=128,
    lloyd_max_iterations=100,
    device='cuda'
)

quant = RaBitQ.from_config(config)
```

### Recommended Configs

```python
from src.rabitq import RECOMMENDED_CONFIGS

# Pre-defined configs
quant = RaBitQ(RECOMMENDED_CONFIGS['fast'])      # 'tq4'
quant = RaBitQ(RECOMMENDED_CONFIGS['balanced'])  # 'int8'
quant = RaBitQ(RECOMMENDED_CONFIGS['extreme'])   # 'tq3'
```

## API Reference

### RaBitQ Class

```python
class RaBitQ:
    def __init__(self, mode: str = 'fp16', **kwargs)
    def fit(self, keys, values) -> RaBitQ
    def fit_beta(self) -> RaBitQ
    def compress_kv(self, keys, values) -> Dict[str, Tensor]
    def decompress_kv(self, compressed) -> Tuple[Tensor, Tensor]
    def memory_stats(self, seq_len, batch_size, num_heads) -> Dict
```

### Parameters

- `mode`: Quantization mode (`'fp16'`, `'int8'`, `'tq3'`, `'tq4'`)
- `head_dim`: Head dimension (default: 128)
- `lloyd_max_iterations`: Iterations for codebook fitting (default: 100)
- `device`: Device (`'cpu'` or `'cuda'`)

## Memory Savings

```python
quant = RaBitQ('tq4')
stats = quant.memory_stats(seq_len=32768, batch_size=1, num_heads=32)

print(f"Original: {stats['original_mb']:.1f} MB")
print(f"Compressed: {stats['compressed_mb']:.1f} MB")
print(f"Saved: {stats['memory_saved']:.0f}%")
```

Output:
```
Original: 1024.0 MB
Compressed: 341.3 MB
Saved: 67%
```

## Comparison with Legacy API

### New API (Recommended)

```python
from src.rabitq import RaBitQ

quant = RaBitQ('tq4')
quant.fit(keys, values)
compressed = quant.compress_kv(keys, values)
```

### Legacy API (Still Supported)

```python
from src.rabitq import RaBitQPipeline, RaBitQConfig

config = RaBitQConfig(angle_bits=3, qjl_proj_dim=256)
pipeline = RaBitQPipeline(dim=4096, config=config)
r, theta, qjl, norm = pipeline.compress_vector(x)
```

## Migration Guide

### From Legacy to New API

| Legacy | New |
|--------|-----|
| `RaBitQPipeline` | `RaBitQ` |
| `RaBitQConfig(angle_bits=3)` | `RaBitQ('tq4')` |
| `compress_vector(x)` | `compress_kv(keys, values)` |
| Manual QJL handling | Automatic |

## Demo

Run the demo:

```bash
python scripts/experiments/rabitq_refactored_demo.py --all
```

## Testing

```bash
pytest tests/unit/test_rabitq_core.py -v
```

## Architecture

```
RaBitQ (Unified API)
├── RaBitQConfig (Configuration)
├── LloydMaxQuantizer (TQ3/TQ4)
├── INT8Quantizer (INT8)
└── FP16Quantizer (FP16)
```

## Backward Compatibility

All legacy classes are still available:

```python
from src.rabitq import (
    RaBitQPipeline,      # Legacy
    PolarQuant,              # Legacy
    QJLCompressor,           # Legacy
    RaBitQ,              # New (Recommended)
)
```
