---
title: "RaBitQ API"
description: "RaBitQ KV 缓存压缩 API 参考"
category: "reference"
last_updated: "2026-04-24"
---

# RaBitQ API 参考

## RaBitQ

RaBitQ 量化器主类。

### 类定义

```python
class RaBitQ:
    def __init__(
        self,
        config: RaBitQConfig
    )
```

### 方法

#### compress

```python
def compress(
    self,
    tensor: torch.Tensor
) -> CompressedTensor
```

**参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `tensor` | torch.Tensor | 输入张量，形状 `[..., head_dim]` |

**返回**: `CompressedTensor` 对象

#### decompress

```python
def decompress(
    self,
    compressed: CompressedTensor
) -> torch.Tensor
```

---

## RaBitQCache

RaBitQ KV 缓存管理器。

### 类定义

```python
class RaBitQCache:
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        compression: str = "k2"
    )
```

### 方法

#### store

```python
def store(
    self,
    layer_idx: int,
    keys: torch.Tensor,
    values: torch.Tensor
)
```

#### retrieve

```python
def retrieve(
    self,
    layer_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]
```

#### get_memory_usage

```python
def get_memory_usage(self) -> int
```

返回当前缓存使用的字节数。

---

## 工厂函数

### create_k1

```python
def create_k1(head_dim: int) -> RaBitQ
```

创建 1-bit 量化器 (~32x 压缩)。

### create_k2

```python
def create_k2(head_dim: int) -> RaBitQ
```

创建 2-bit 量化器 (~16x 压缩)。

### create_k3

```python
def create_k3(head_dim: int) -> RaBitQ
```

创建 3-bit 量化器 (~10.7x 压缩)。

---

## 使用示例

### 基础压缩

```python
from src.rabitq import create_k2

# 创建量化器
rq = create_k2(head_dim=128)

# 压缩
keys = torch.randn(1, 8, 2048, 128)
compressed = rq.compress(keys)

# 解压缩
decompressed = rq.decompress(compressed)
error = torch.norm(keys - decompressed)
print(f"重建误差: {error.item():.4f}")
```

### 缓存管理

```python
from src.rabitq import RaBitQCache

# 创建缓存
cache = RaBitQCache(
    num_layers=32,
    num_heads=8,
    head_dim=128,
    max_seq_len=32768,
    compression='k2'
)

# 存储
for layer_idx in range(32):
    keys = torch.randn(1, 8, 1024, 128)
    values = torch.randn(1, 8, 1024, 128)
    cache.store(layer_idx, keys, values)

# 检查内存
memory_mb = cache.get_memory_usage() / 1024 / 1024
print(f"缓存使用: {memory_mb:.2f} MB")
```

---

## 参考

- [源码](../../src/rabitq/)
- [教程 3: RaBitQ 压缩](../../tutorials/tutorial-03-rabitq.md)
