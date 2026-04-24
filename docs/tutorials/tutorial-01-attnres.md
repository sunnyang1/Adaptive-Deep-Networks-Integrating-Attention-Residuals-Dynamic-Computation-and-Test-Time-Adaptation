---
title: "教程 1: AttnRes 入门"
description: "学习块注意力残差 (Block Attention Residuals) 的基本概念和使用方法"
category: "tutorials"
difficulty: "beginner"
duration: "30分钟"
prerequisites: ["Python基础", "PyTorch基础"]
last_updated: "2026-04-24"
---

# 教程 1: AttnRes 入门

本教程将带你了解 ADN 的核心模块之一：**块注意力残差 (Block Attention Residuals, AttnRes)**。你将学习它的基本概念、工作原理，以及如何在代码中使用它。

---

## 你将学习

- [ ] 什么是块注意力残差 (AttnRes)
- [ ] 伪查询 (Pseudo-Queries) 的概念
- [ ] 如何创建和配置 AttnRes 模块
- [ ] 运行简单的注意力实验

---

## 1. 什么是 AttnRes？

### 1.1 背景

在标准 Transformer 中，注意力机制需要计算查询 (Query) 与所有键 (Key) 的点积，导致内存复杂度为 **O(L²)**，其中 L 是序列长度。

AttnRes 通过引入**块结构**来解决这个问题：

```
标准注意力: O(L²) 内存
AttnRes:     O(L × N) 内存 (N << L, 通常 N=8)
```

### 1.2 核心思想

AttnRes 将序列分成 **N 个块**，每个块维护一个**块表示 (Block Representation)**。注意力分为两个阶段：

**阶段 1: 块间注意力 (Inter-Block)**
- 查询与所有块表示进行注意力计算
- 并行计算，确定关注哪些块

**阶段 2: 块内注意力 (Intra-Block)**
- 查询与选中块内的所有 token 进行注意力
- 顺序计算，获取细粒度信息

```
┌─────────────────────────────────────────┐
│           阶段 1: 块间注意力            │
│  Query ──> [Block1][Block2][Block3]...  │
│              ↓      ↓       ↓           │
│           选择关注的块                  │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│           阶段 2: 块内注意力            │
│  Query ──> [Token1][Token2][Token3]...  │
│              ↓      ↓       ↓           │
│           获取具体信息                  │
└─────────────────────────────────────────┘
```

---

## 2. 伪查询 (Pseudo-Queries)

### 2.1 概念

**伪查询**是 AttnRes 的核心创新。它们是**可学习的参数**，用于：
1. 在阶段 1 中作为"代表"参与块间注意力
2. 帮助模型学习哪些块包含重要信息

```python
# 伪查询的维度
pseudo_query_dim = hidden_dim  # 与隐藏维度相同
num_pseudo_queries = num_blocks  # 每个块一个伪查询
```

### 2.2 初始化策略

伪查询使用**零初始化**，这是训练稳定的关键：

```python
# 零初始化确保训练开始时行为与标准注意力相似
self.pseudo_queries = nn.Parameter(torch.zeros(num_blocks, hidden_dim))
```

> **论文依据**: §5.3 指出零初始化对训练稳定性至关重要。

---

## 3. 代码实践

### 3.1 安装和导入

```python
import sys
sys.path.insert(0, '/path/to/adn')

import torch
from src.attnres import BlockAttnRes, TwoPhaseBlockAttnRes
```

### 3.2 创建 AttnRes 模块

```python
# 基础配置
hidden_dim = 512
num_heads = 8
num_blocks = 8

# 创建 AttnRes 模块
attn_res = BlockAttnRes(
    hidden_dim=hidden_dim,
    num_heads=num_heads,
    num_blocks=num_blocks
)

print(f"Created AttnRes module:")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Num heads: {num_heads}")
print(f"  Num blocks: {num_blocks}")
print(f"  Head dim: {hidden_dim // num_heads}")
```

### 3.3 前向传播

```python
# 创建示例输入
batch_size = 2
seq_len = 128

x = torch.randn(batch_size, seq_len, hidden_dim)

# 前向传播
output, stats = attn_res(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Stats: {stats}")
```

### 3.4 两阶段 AttnRes

对于更复杂的场景，使用 `TwoPhaseBlockAttnRes`：

```python
# 创建两阶段模块
two_phase = TwoPhaseBlockAttnRes(
    hidden_dim=hidden_dim,
    num_heads=num_heads,
    num_blocks=num_blocks,
    use_rmsnorm=True  # 论文推荐
)

# 前向传播
output, stats = two_phase(x)

# 查看阶段统计
print(f"Phase 1 (inter-block) attention shape: {stats.get('phase1_attn', 'N/A')}")
print(f"Phase 2 (intra-block) attention shape: {stats.get('phase2_attn', 'N/A')}")
```

---

## 4. 完整示例

创建一个完整的可运行脚本：

```python
#!/usr/bin/env python3
"""
AttnRes 入门示例
运行: python tutorial_01_attnres_demo.py
"""

import torch
import torch.nn as nn
from src.attnres import BlockAttnRes, TwoPhaseBlockAttnRes

def demo_basic_attnres():
    """演示基本的 AttnRes 使用"""
    print("=" * 60)
    print("演示 1: 基本 AttnRes")
    print("=" * 60)

    # 配置
    hidden_dim = 512
    num_heads = 8
    num_blocks = 8
    batch_size = 2
    seq_len = 128

    # 创建模块
    attn_res = BlockAttnRes(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_blocks=num_blocks
    )

    # 创建输入
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # 前向传播
    output, stats = attn_res(x)

    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 参数量: {sum(p.numel() for p in attn_res.parameters()):,}")
    print()

def demo_two_phase():
    """演示两阶段 AttnRes"""
    print("=" * 60)
    print("演示 2: 两阶段 AttnRes")
    print("=" * 60)

    # 配置
    config = {
        'hidden_dim': 768,
        'num_heads': 12,
        'num_blocks': 8,
        'use_rmsnorm': True,
    }

    attn_res = TwoPhaseBlockAttnRes(**config)

    # 测试不同序列长度
    batch_size = 2
    for seq_len in [64, 128, 256, 512]:
        x = torch.randn(batch_size, seq_len, config['hidden_dim'])
        output, stats = attn_res(x)

        # 计算内存节省
        standard_memory = seq_len * seq_len * batch_size * 4 / 1024 / 1024  # MB
        attnres_memory = seq_len * config['num_blocks'] * batch_size * 4 / 1024 / 1024  # MB
        savings = (1 - attnres_memory / standard_memory) * 100

        print(f"序列长度 {seq_len:4d}: 内存节省 {savings:.1f}%")

    print()

def demo_comparison():
    """对比标准注意力和 AttnRes"""
    print("=" * 60)
    print("演示 3: 标准注意力 vs AttnRes")
    print("=" * 60)

    hidden_dim = 512
    num_heads = 8
    batch_size = 2

    # 标准注意力 (简化版)
    class StandardAttention(nn.Module):
        def __init__(self, hidden_dim, num_heads):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x):
            B, L, D = x.shape
            q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, L, D)
            return self.o_proj(out)

    standard = StandardAttention(hidden_dim, num_heads)
    attn_res = BlockAttnRes(hidden_dim, num_heads, num_blocks=8)

    x = torch.randn(batch_size, 256, hidden_dim)

    # 对比
    import time

    # 标准注意力
    start = time.time()
    out_std = standard(x)
    time_std = time.time() - start

    # AttnRes
    start = time.time()
    out_ar, _ = attn_res(x)
    time_ar = time.time() - start

    print(f"标准注意力: {time_std*1000:.2f} ms")
    print(f"AttnRes:    {time_ar*1000:.2f} ms")
    print(f"速度比:     {time_std/time_ar:.2f}x")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AttnRes 入门教程")
    print("=" * 60 + "\n")

    demo_basic_attnres()
    demo_two_phase()
    demo_comparison()

    print("=" * 60)
    print("教程完成！")
    print("=" * 60)
    print("\n下一步: 学习教程 2 - qTTT 使用")
```

---

## 5. 关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_dim` | int | 必填 | 隐藏层维度 |
| `num_heads` | int | 必填 | 注意力头数 |
| `num_blocks` | int | 8 | 块数量，论文推荐 N=8 |
| `use_rmsnorm` | bool | False | 是否使用 RMSNorm (推荐) |
| `dropout` | float | 0.0 | Dropout 概率 |

### 最佳实践

1. **块数量选择**: N=8 是论文推荐的"甜点"，能恢复大部分 FullAttnRes 的效益
2. **使用 RMSNorm**: 在 keys 上使用 RMSNorm 对性能至关重要
3. **单头深度注意力**: 深度注意力使用单头即可，多头反而会降低性能

---

## 6. 练习

### 练习 1: 改变块数量

尝试不同的 `num_blocks` 值，观察对性能和内存的影响：

```python
for num_blocks in [4, 8, 16, 32]:
    attn_res = BlockAttnRes(hidden_dim=512, num_heads=8, num_blocks=num_blocks)
    # 测量内存和速度
```

### 练习 2: 可视化注意力模式

```python
import matplotlib.pyplot as plt

output, stats = attn_res(x)

# 假设 stats 包含注意力权重
if 'attention_weights' in stats:
    plt.imshow(stats['attention_weights'][0].detach().numpy())
    plt.colorbar()
    plt.title("AttnRes Attention Pattern")
    plt.show()
```

---

## 7. 常见问题

**Q: AttnRes 会损失多少精度？**
A: 论文显示 N=8 时，AttnRes 能恢复 FullAttnRes 95% 以上的性能。

**Q: 什么时候应该使用 TwoPhaseBlockAttnRes？**
A: 当你需要更细粒度的控制，或者需要与增量生成配合时。

**Q: 伪查询需要特殊的学习率吗？**
A: 论文使用与其他参数相同的学习率 (1.0x)。

---

## 8. 下一步

完成本教程后，你应该能够：
- [x] 理解 AttnRes 的基本概念
- [x] 创建和配置 AttnRes 模块
- [x] 运行基本的注意力实验

**下一步**: [教程 2: qTTT 使用](tutorial-02-qttt.md)

---

## 参考

- [AttnRes API 文档](../reference/api/attnres.md)
- [论文 §5.3](../papers/adn-paper.md#53-attention-residuals)
- [PROJECT_ORGANIZATION.md](../../PROJECT_ORGANIZATION.md)
