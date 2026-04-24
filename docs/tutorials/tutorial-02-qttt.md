---
title: "教程 2: qTTT 使用"
description: "学习查询时训练 (Query-Only Test-Time Training) 的基本概念和配置方法"
category: "tutorials"
difficulty: "beginner"
duration: "30分钟"
prerequisites: ["完成教程 1", "PyTorch优化基础"]
last_updated: "2026-04-24"
---

# 教程 2: qTTT 使用

本教程将带你了解 ADN 的**查询时训练 (Query-Only Test-Time Training, qTTT)** 模块。你将学习它的核心概念、配置方法，以及如何在推理时进行自适应优化。

---

## 你将学习

- [ ] 查询时训练 (qTTT) 的基本概念
- [ ] 冻结 KV 缓存的工作原理
- [ ] 如何配置 qTTT 参数
- [ ] 使用边际最大化损失

---

## 1. 什么是 qTTT？

### 1.1 背景

标准的大语言模型在推理时**冻结所有参数**，这导致：
- 无法适应特定查询的特点
- 长上下文中的"注意力稀释"问题
- 难以处理分布外样本

**qTTT 的核心思想**: 在推理时**只更新查询 (Query) 参数**，而冻结键 (Key) 和值 (Value)。

```
标准推理: 所有参数冻结
qTTT:      Query 可更新, KV 冻结

优势:
- 计算成本低 (只更新 ~25% 参数)
- 保持 KV 缓存一致性
- 针对特定查询优化
```

### 1.2 冻结 KV 缓存

这是 qTTT 的关键设计：

```python
# Prefill 阶段: 计算并缓存 KV
keys, values = compute_kv_cache(input_tokens)
kv_cache = (keys, values)  # 冻结！

# qTTT 阶段: 只更新 Query
for step in range(max_steps):
    # 使用相同的 KV，只优化 Query
    query = optimize_query(query, kv_cache)
    loss = margin_maximization_loss(query, kv_cache, target)
    query = query - lr * grad(loss, query)
```

> **论文依据**: qTTT 基于 Bansal et al. 的工作，但采用查询专用更新策略。

---

## 2. 边际最大化损失

### 2.1 为什么需要边际损失？

标准交叉熵损失只关注正确标签的概率，而边际损失**显式最大化正确标签与错误标签之间的差距**。

```python
# 标准交叉熵
loss = -log(p_correct)

# 边际最大化
margin = logit_correct - max(logit_incorrect)
loss = max(0, margin_target - margin)
```

### 2.2 对数边际要求

论文指出，长上下文需要**对数级别的边际**：

```
所需边际 ∝ log(上下文长度)
```

这是因为注意力分数会随序列长度稀释，需要更大的边际来保持区分度。

---

## 3. 代码实践

### 3.1 基础配置

```python
from src.qttt import qTTTConfig, QueryOnlyTTT

# 创建配置
config = qTTTConfig(
    max_steps=32,           # 最大优化步数
    learning_rate=0.005,    # 学习率
    span_length=128,        # 跨度长度
    use_margin_loss=True,   # 使用边际损失
    margin_target=2.0       # 目标边际
)

print(f"qTTT Config:")
print(f"  Max steps: {config.max_steps}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Span length: {config.span_length}")
```

### 3.2 创建 qTTT 适配器

```python
from src.qttt import AdaptiveInference

# 创建自适应推理模块
adapter = AdaptiveInference(
    model=your_transformer_model,
    config=config
)

# 使用适配器进行推理
output = adapter.generate(
    input_ids=input_tokens,
    max_new_tokens=100,
    use_qttt=True  # 启用 qTTT
)
```

### 3.3 批量适配

```python
from src.qttt import BatchAdapter

# 批量适配器
batch_adapter = BatchAdapter(config)

# 适配多个样本
results = batch_adapter.adapt_batch(
    queries=query_batch,
    kv_cache=shared_kv_cache,
    targets=target_batch
)
```

---

## 4. 完整示例

```python
#!/usr/bin/env python3
"""
qTTT 使用示例
运行: python tutorial_02_qttt_demo.py
"""

import torch
import torch.nn as nn
from src.qttt import qTTTConfig, QueryOnlyTTT, MarginMaximizationLoss

def demo_basic_qttt():
    """演示基本的 qTTT 配置"""
    print("=" * 60)
    print("演示 1: 基本 qTTT 配置")
    print("=" * 60)

    # 不同场景的配置
    configs = {
        'fast': qTTTConfig(max_steps=8, learning_rate=0.01),
        'balanced': qTTTConfig(max_steps=16, learning_rate=0.005),
        'quality': qTTTConfig(max_steps=32, learning_rate=0.003),
    }

    for name, config in configs.items():
        print(f"\n{name.upper()} 模式:")
        print(f"  步数: {config.max_steps}")
        print(f"  学习率: {config.learning_rate}")
        print(f"  预计时间: {config.max_steps * 0.5:.1f} ms/token")

def demo_margin_loss():
    """演示边际最大化损失"""
    print("\n" + "=" * 60)
    print("演示 2: 边际最大化损失")
    print("=" * 60)

    # 模拟 logits
    batch_size = 2
    vocab_size = 1000

    logits = torch.randn(batch_size, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size,))

    # 创建损失函数
    margin_loss = MarginMaximizationLoss(
        margin_target=2.0,
        hard_negative_weight=1.0
    )

    # 计算损失
    loss = margin_loss(logits, targets)

    print(f"\n模拟数据:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Targets: {targets.tolist()}")
    print(f"  Margin loss: {loss.item():.4f}")

    # 对比交叉熵
    ce_loss = nn.CrossEntropyLoss()(logits, targets)
    print(f"  Cross-entropy: {ce_loss.item():.4f}")

def demo_adaptation_loop():
    """演示适配循环"""
    print("\n" + "=" * 60)
    print("演示 3: 适配循环")
    print("=" * 60)

    # 模拟设置
    hidden_dim = 512
    num_heads = 8
    seq_len = 128

    # 模拟 KV 缓存 (已冻结)
    keys = torch.randn(1, num_heads, seq_len, hidden_dim // num_heads)
    values = torch.randn(1, num_heads, seq_len, hidden_dim // num_heads)
    kv_cache = (keys, values)

    # 初始化查询
    query = torch.randn(1, num_heads, 1, hidden_dim // num_heads)

    # 目标 token ID
    target_token = torch.tensor([42])

    # qTTT 配置
    config = qTTTConfig(max_steps=16, learning_rate=0.005)

    print(f"\n初始状态:")
    print(f"  Query shape: {query.shape}")
    print(f"  KV cache: {keys.shape}, {values.shape}")

    # 模拟适配步骤
    query.requires_grad = True
    optimizer = torch.optim.SGD([query], lr=config.learning_rate)

    losses = []
    for step in range(config.max_steps):
        optimizer.zero_grad()

        # 模拟注意力计算
        scores = torch.matmul(query, keys.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, values)

        # 模拟输出投影和损失
        logits = torch.randn(1, 1000)  # 简化
        loss = nn.CrossEntropyLoss()(logits, target_token)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 4 == 0:
            print(f"  Step {step:2d}: loss = {loss.item():.4f}")

    print(f"\n适配完成!")
    print(f"  初始损失: {losses[0]:.4f}")
    print(f"  最终损失: {losses[-1]:.4f}")
    print(f"  改善: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")

def demo_context_length_scaling():
    """演示上下文长度与边际的关系"""
    print("\n" + "=" * 60)
    print("演示 4: 上下文长度与边际要求")
    print("=" * 60)

    import math

    context_lengths = [1024, 4096, 16384, 32768, 65536, 131072]
    base_margin = 1.0

    print("\n上下文长度 vs 建议边际:")
    print(f"{'Context Length':<15} {'Suggested Margin':<20}")
    print("-" * 35)

    for length in context_lengths:
        # 对数缩放
        suggested_margin = base_margin * math.log2(length / 1024 + 1)
        print(f"{length:<15} {suggested_margin:<20.2f}")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("qTTT 使用教程")
    print("=" * 60 + "\n")

    demo_basic_qttt()
    demo_margin_loss()
    demo_adaptation_loop()
    demo_context_length_scaling()

    print("\n" + "=" * 60)
    print("教程完成！")
    print("=" * 60)
    print("\n下一步: 学习教程 3 - RaBitQ 压缩")
```

---

## 5. 配置参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_steps` | int | 32 | 最大优化步数 |
| `learning_rate` | float | 0.005 | 查询参数学习率 |
| `span_length` | int | 128 | 处理跨度长度 |
| `use_margin_loss` | bool | True | 是否使用边际损失 |
| `margin_target` | float | 2.0 | 目标边际值 |
| `hard_negative_weight` | float | 1.0 | 难负样本权重 |

### 预设配置模式

```python
# 快速模式 - 低延迟
FAST = qTTTConfig(max_steps=8, learning_rate=0.01)

# 平衡模式 - 默认推荐
BALANCED = qTTTConfig(max_steps=16, learning_rate=0.005)

# 质量模式 - 高精度
QUALITY = qTTTConfig(max_steps=32, learning_rate=0.003)
```

---

## 6. 练习

### 练习 1: 对比不同步数的效果

```python
for max_steps in [4, 8, 16, 32, 64]:
    config = qTTTConfig(max_steps=max_steps)
    # 运行适配并记录最终损失
```

### 练习 2: 学习率调优

```python
learning_rates = [0.001, 0.003, 0.005, 0.01, 0.02]
# 对每个学习率运行适配，绘制收敛曲线
```

---

## 7. 常见问题

**Q: qTTT 会增加多少推理延迟？**
A: 取决于 `max_steps`，通常增加 20-50% 的延迟，但显著提升质量。

**Q: 为什么只更新 Query 而不更新 KV？**
A: 1) 保持 KV 缓存一致性；2) 降低计算成本；3) 避免破坏预训练知识。

**Q: 边际目标值如何选择？**
A: 短上下文 (1K-4K): 1.0-1.5；中上下文 (8K-32K): 2.0-2.5；长上下文 (64K+): 3.0+。

---

## 8. 下一步

完成本教程后，你应该能够：
- [x] 理解 qTTT 的核心概念
- [x] 配置 qTTT 参数
- [x] 使用边际最大化损失

**下一步**: [教程 3: RaBitQ 压缩](tutorial-03-rabitq.md)

---

## 参考

- [qTTT API 文档](../reference/api/qttt.md)
- [论文 §5.2](../papers/adn-paper.md#52-test-time-training)
- [MarginMaximizationLoss 源码](../../src/qttt/margin_loss.py)
