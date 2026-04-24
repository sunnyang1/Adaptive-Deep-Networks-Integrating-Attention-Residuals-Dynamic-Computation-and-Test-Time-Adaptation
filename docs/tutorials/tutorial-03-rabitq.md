---
title: "教程 3: RaBitQ 压缩"
description: "学习 RaBitQ KV 缓存压缩的原理和使用方法"
category: "tutorials"
difficulty: "beginner"
duration: "30分钟"
prerequisites: ["完成教程 1", "量化基础"]
last_updated: "2026-04-24"
---

# 教程 3: RaBitQ 压缩

本教程将带你了解 ADN 的 **RaBitQ (Randomized Binary Quantization)** KV 缓存压缩模块。你将学习它的压缩原理、配置选择，以及如何与模型集成。

---

## 你将学习

- [ ] RaBitQ 压缩原理
- [ ] 如何选择压缩配置 (k1/k2/k3)
- [ ] 压缩和解压缩 KV 缓存
- [ ] 与 HuggingFace 集成

---

## 1. 什么是 RaBitQ？

### 1.1 背景

KV 缓存是 Transformer 推理的内存瓶颈：

```
内存 = 2 × batch_size × num_layers × num_heads × seq_len × head_dim × sizeof(float32)

示例: batch=1, layers=32, heads=8, seq_len=32768, head_dim=128
      = 2 × 1 × 32 × 8 × 32768 × 128 × 4 bytes
      = 8.6 GB！
```

**RaBitQ 的解决方案**: 使用**随机旋转 + 二进制量化**实现高倍率压缩。

### 1.2 核心思想

RaBitQ 基于三个关键洞察：

1. **随机旋转**: 通过随机正交矩阵旋转，使向量各维度方差均匀化
2. **二进制量化**: 将旋转后的向量量化为 1-bit (符号)
3. **残差编码**: 保留少量高精度残差信息

```
原始向量 x ──> 随机旋转 ──> 二进制量化 ──> 压缩表示
     ↑                                      ↓
     └──────── 残差信息存储 ────────────────┘
```

### 1.3 压缩率

| 配置 | 位宽 | 压缩率 | 适用场景 |
|------|------|--------|----------|
| k1 | 1-bit | ~32x | 极致压缩，允许一定精度损失 |
| k2 | 2-bit | ~16x | 平衡压缩率和精度 |
| k3 | 3-bit | ~10.7x | 高精度，较小压缩 |

---

## 2. 压缩配置

### 2.1 k1 配置 (1-bit)

```python
from src.rabitq import create_k1

# 创建 1-bit 量化器
rq = create_k1(head_dim=64)

# 压缩
keys = torch.randn(1, 8, 1024, 64)  # [batch, heads, seq, head_dim]
compressed = rq.compress(keys)

print(f"原始大小: {keys.numel() * 4 / 1024 / 1024:.2f} MB")
print(f"压缩后大小: {compressed.size} / 1024 / 1024:.2f} MB")
print(f"压缩率: {keys.numel() * 4 / compressed.size:.1f}x")
```

### 2.2 k2 配置 (2-bit)

```python
from src.rabitq import create_k2

# 创建 2-bit 量化器
rq = create_k2(head_dim=64)

# 压缩
compressed = rq.compress(keys)

# 解压缩
decompressed = rq.decompress(compressed)
print(f"重建误差: {torch.norm(keys - decompressed).item():.4f}")
```

### 2.3 k3 配置 (3-bit)

```python
from src.rabitq import create_k3

# 创建 3-bit 量化器
rq = create_k3(head_dim=64)

# 压缩
compressed = rq.compress(keys)
```

---

## 3. 代码实践

### 3.1 基础使用

```python
from src.rabitq import RaBitQ, RaBitQConfig

# 创建配置
config = RaBitQConfig(
    head_dim=128,
    num_bits=2,  # 1, 2, or 3
    use_rotation=True,
    random_seed=42
)

# 创建量化器
rabitq = RaBitQ(config)

# 准备数据
keys = torch.randn(2, 8, 2048, 128)  # [batch, heads, seq, head_dim]
values = torch.randn(2, 8, 2048, 128)

# 压缩
compressed_keys = rabitq.compress(keys)
compressed_values = rabitq.compress(values)
```

### 3.2 缓存管理

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

# 存储 KV
cache.store(layer_idx=0, keys=keys, values=values)

# 检索 KV
retrieved_keys, retrieved_values = cache.retrieve(layer_idx=0)
```

### 3.3 与模型集成

```python
from src.models import AdaptiveTransformer, AttnResSmallConfig
from src.rabitq import enable_rabitq

# 创建模型
config = AttnResSmallConfig()
model = AdaptiveTransformer(config)

# 启用 RaBitQ
enable_rabitq(
    model,
    compression='k2',
    head_dim=config.hidden_dim // config.num_heads
)

# 现在模型的 KV 缓存会自动压缩
```

---

## 4. 完整示例

```python
#!/usr/bin/env python3
"""
RaBitQ 压缩示例
运行: python tutorial_03_rabitq_demo.py
"""

import torch
import time
from src.rabitq import create_k1, create_k2, create_k3, RaBitQCache

def demo_compression_rates():
    """演示不同配置的压缩率"""
    print("=" * 60)
    print("演示 1: 压缩率对比")
    print("=" * 60)

    # 模拟 KV 缓存
    batch_size = 1
    num_heads = 8
    seq_len = 8192
    head_dim = 128

    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    original_size = keys.numel() * 4  # float32

    configs = {
        'k1 (1-bit)': create_k1(head_dim),
        'k2 (2-bit)': create_k2(head_dim),
        'k3 (3-bit)': create_k3(head_dim),
    }

    print(f"\n原始数据: {original_size / 1024 / 1024:.2f} MB")
    print(f"\n{'配置':<15} {'压缩后':<15} {'压缩率':<10} {'重建误差':<15}")
    print("-" * 60)

    for name, rq in configs.items():
        # 压缩
        compressed = rq.compress(keys)
        compressed_size = compressed.size if hasattr(compressed, 'size') else compressed.numel()

        # 解压缩
        decompressed = rq.decompress(compressed)
        error = torch.norm(keys - decompressed).item() / keys.numel()

        ratio = original_size / compressed_size
        print(f"{name:<15} {compressed_size/1024/1024:<15.2f} {ratio:<10.1f}x {error:<15.6f}")

def demo_speed_benchmark():
    """演示压缩/解压缩速度"""
    print("\n" + "=" * 60)
    print("演示 2: 速度基准测试")
    print("=" * 60)

    keys = torch.randn(1, 8, 4096, 128)
    rq = create_k2(head_dim=128)

    # 预热
    for _ in range(10):
        compressed = rq.compress(keys)
        _ = rq.decompress(compressed)

    # 测试压缩速度
    num_iters = 100
    start = time.time()
    for _ in range(num_iters):
        compressed = rq.compress(keys)
    compress_time = (time.time() - start) / num_iters * 1000

    # 测试解压缩速度
    compressed = rq.compress(keys)
    start = time.time()
    for _ in range(num_iters):
        _ = rq.decompress(compressed)
    decompress_time = (time.time() - start) / num_iters * 1000

    print(f"\n序列长度: 4096, Head dim: 128")
    print(f"压缩时间: {compress_time:.2f} ms")
    print(f"解压缩时间: {decompress_time:.2f} ms")
    print(f"总延迟增加: {compress_time + decompress_time:.2f} ms/token")

def demo_cache_management():
    """演示缓存管理"""
    print("\n" + "=" * 60)
    print("演示 3: 缓存管理")
    print("=" * 60)

    # 创建缓存
    cache = RaBitQCache(
        num_layers=4,  # 简化
        num_heads=8,
        head_dim=128,
        max_seq_len=8192,
        compression='k2'
    )

    # 模拟多层存储
    for layer_idx in range(4):
        keys = torch.randn(1, 8, 2048, 128)
        values = torch.randn(1, 8, 2048, 128)
        cache.store(layer_idx, keys, values)
        print(f"✓ 存储 Layer {layer_idx}: {keys.shape}")

    # 计算总内存
    total_memory = cache.get_memory_usage()
    print(f"\n总内存使用: {total_memory / 1024 / 1024:.2f} MB")

    # 对比未压缩
    uncompressed = 4 * 4 * 8 * 2048 * 128 * 4 / 1024 / 1024  # 4 layers
    print(f"未压缩估计: {uncompressed:.2f} MB")
    print(f"节省: {(1 - total_memory / uncompressed / 1024 / 1024) * 100:.1f}%")

def demo_error_analysis():
    """分析不同序列位置的重建误差"""
    print("\n" + "=" * 60)
    print("演示 4: 重建误差分析")
    print("=" * 60)

    seq_lengths = [1024, 2048, 4096, 8192]
    head_dim = 128
    rq = create_k2(head_dim)

    print(f"\n{'序列长度':<12} {'k2 误差':<15}")
    print("-" * 30)

    for seq_len in seq_lengths:
        keys = torch.randn(1, 8, seq_len, head_dim)
        compressed = rq.compress(keys)
        decompressed = rq.decompress(compressed)

        # 计算相对误差
        relative_error = torch.norm(keys - decompressed) / torch.norm(keys)
        print(f"{seq_len:<12} {relative_error.item():<15.4f}")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RaBitQ 压缩教程")
    print("=" * 60 + "\n")

    demo_compression_rates()
    demo_speed_benchmark()
    demo_cache_management()
    demo_error_analysis()

    print("\n" + "=" * 60)
    print("教程完成！")
    print("=" * 60)
    print("\n下一步: 学习教程 4 - Engram 记忆")
```

---

## 5. 配置选择指南

### 5.1 根据场景选择

| 场景 | 推荐配置 | 理由 |
|------|----------|------|
| 边缘设备/手机 | k1 | 极致压缩，内存受限 |
| 实时推理 | k2 | 平衡压缩率和速度 |
| 高精度要求 | k3 | 最小精度损失 |
| 长上下文 (>64K) | k2/k3 | 需要更大缓存 |

### 5.2 与注意力机制配合

```python
# AttnRes + RaBitQ 组合
from src.attnres import BlockAttnRes
from src.rabitq import create_k2

# 创建 AttnRes 模块
attn_res = BlockAttnRes(hidden_dim=512, num_heads=8, num_blocks=8)

# 创建 RaBitQ 压缩器
rabitq = create_k2(head_dim=64)

# 在推理时自动压缩 KV 缓存
# AttnRes 减少计算量，RaBitQ 减少内存
```

---

## 6. 练习

### 练习 1: 对比不同 head_dim 的压缩效果

```python
for head_dim in [64, 128, 256]:
    rq = create_k2(head_dim)
    # 测试压缩率和重建误差
```

### 练习 2: 实现自定义压缩策略

```python
class CustomCompression:
    def __init__(self, head_dim, num_bits):
        self.rq = RaBitQConfig(head_dim, num_bits)

    def compress(self, tensor):
        # 自定义预处理
        processed = self.preprocess(tensor)
        return self.rq.compress(processed)
```

---

## 7. 常见问题

**Q: RaBitQ 与 TurboQuant 有什么区别？**
A: RaBitQ 是项目当前使用的压缩方案，TurboQuant 是旧版实现。RaBitQ 提供更好的压缩率和重建质量。

**Q: 压缩后的缓存可以跨设备使用吗？**
A: 可以，但需要注意随机旋转矩阵的一致性（通过设置相同的 random_seed）。

**Q: 如何处理动态增长的序列？**
A: 使用 `RaBitQCache` 类，它支持增量存储和动态扩展。

---

## 8. 下一步

完成本教程后，你应该能够：
- [x] 理解 RaBitQ 的压缩原理
- [x] 选择适合的压缩配置
- [x] 压缩和解压缩 KV 缓存
- [x] 与模型集成

**下一步**: [教程 4: Engram 记忆](tutorial-04-engram.md)

---

## 参考

- [RaBitQ API 文档](../reference/api/rabitq.md)
- [RaBitQ 指南](../guides/RABITQ_GUIDE.md)
- [源码](../../src/rabitq/)
