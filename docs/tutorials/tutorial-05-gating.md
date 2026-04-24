---
title: "教程 5: 动态门控"
description: "学习动态计算门控的概念和实现方法"
category: "tutorials"
difficulty: "intermediate"
duration: "45分钟"
prerequisites: ["完成教程 1-4", "概率基础"]
last_updated: "2026-04-24"
---

# 教程 5: 动态门控

本教程将带你了解 ADN 的**动态计算门控 (Dynamic Gating)** 机制。你将学习如何使用重构损失作为难度信号、阈值校准策略，以及 Ponder Gate 的实现。

---

## 你将学习

- [ ] 动态计算门控的概念
- [ ] 重构损失作为难度信号
- [ ] 阈值校准策略 (EMA, Target Rate)
- [ ] Ponder Gate 的使用

---

## 1. 什么是动态门控？

### 1.1 背景

在推理时，不同 token 的计算需求不同：
- **简单 token**: 标准前向传播即可
- **困难 token**: 需要额外的 qTTT 适配

**动态门控的目标**: 根据输入难度**自适应地决定**是否执行额外计算。

```
标准推理: 所有 token 相同计算
动态门控: 简单 token ──> 快速路径
          困难 token ──> 深度路径 (qTTT)
```

### 1.2 核心思想

使用**重构损失**作为难度指标：

```python
# 重构损失 = 模型对自身输出的预测误差
reconstruction_loss = compute_reconstruction_loss(model_output)

# 损失高 -> 模型不确定 -> 需要额外计算
if reconstruction_loss > threshold:
    apply_qttt_adaptation()
else:
    use_standard_output()
```

---

## 2. 重构损失

### 2.1 为什么重构损失有效？

重构损失反映了模型的**自我一致性**：
- **低损失**: 模型对预测很有信心
- **高损失**: 模型处于不确定状态

```python
def compute_reconstruction_loss(logits, targets):
    """
    计算重构损失

    这里的技巧是：用模型的输出作为目标
    如果模型不能很好地"解释"自己的输出，说明它不确定
    """
    probs = F.softmax(logits, dim=-1)
    # 使用模型自己的预测作为软目标
    soft_targets = probs.detach()
    loss = F.kl_div(F.log_softmax(logits, dim=-1), soft_targets, reduction='batchmean')
    return loss
```

### 2.2 与困惑度的关系

```
困惑度 (Perplexity) = exp(交叉熵损失)
重构损失 ≈ 困惑度的变体

高困惑度 -> 高重构损失 -> 触发深度计算
```

---

## 3. 阈值校准策略

### 3.1 EMA 阈值

使用指数移动平均动态调整阈值：

```python
from src.gating import EMAThreshold

# 创建 EMA 阈值控制器
threshold = EMAThreshold(
    target_rate=0.3,      # 目标: 30% 的 token 触发深度计算
    ema_decay=0.99        # EMA 衰减率
)

# 在推理循环中使用
for token in sequence:
    loss = compute_reconstruction_loss(output)
    should_adapt = threshold.should_adapt(loss)

    if should_adapt:
        output = apply_qttt(output)

    # 更新阈值
    threshold.update(loss, should_adapt)
```

### 3.2 目标率阈值

直接控制适应率：

```python
from src.gating import TargetRateThreshold

# 创建目标率控制器
threshold = TargetRateThreshold(target_rate=0.3)

# 使用
should_adapt = threshold.decide(loss)
```

---

## 4. Ponder Gate

### 4.1 什么是 Ponder Gate？

Ponder Gate 是一个**元控制器**，决定：
1. 是否执行 qTTT
2. 执行多少步 qTTT

```python
from src.gating import PonderGate

# 创建 Ponder Gate
ponder = PonderGate(mode='balanced')

# 三种模式
# 'strict':   只有高置信度才跳过 qTTT
# 'balanced': 默认平衡策略
# 'lenient':  更容易跳过 qTTT
```

### 4.2 决策逻辑

```python
def decide(self, entropy, max_prob, reconstruction_loss):
    """
    基于多个信号做决策

    Signals:
    - entropy: 输出分布的熵 (高 = 不确定)
    - max_prob: 最大概率 (低 = 不确定)
    - reconstruction_loss: 重构损失 (高 = 不确定)
    """
    uncertainty_score = (
        self.weights['entropy'] * normalize(entropy) +
        self.weights['max_prob'] * (1 - max_prob) +
        self.weights['reconstruction'] * normalize(reconstruction_loss)
    )

    return uncertainty_score > self.threshold
```

---

## 5. 代码实践

### 5.1 完整示例

```python
#!/usr/bin/env python3
"""
动态门控示例
运行: python tutorial_05_gating_demo.py
"""

import torch
import torch.nn.functional as F
from src.gating import EMAThreshold, TargetRateThreshold, PonderGate

def compute_reconstruction_loss(logits):
    """计算重构损失"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    # KL 散度作为重构损失
    loss = F.kl_div(log_probs, probs.detach(), reduction='batchmean')
    return loss

def demo_ema_threshold():
    """演示 EMA 阈值"""
    print("=" * 60)
    print("演示 1: EMA 阈值校准")
    print("=" * 60)

    threshold = EMAThreshold(target_rate=0.3, ema_decay=0.95)

    # 模拟推理过程
    torch.manual_seed(42)

    adaptation_history = []
    threshold_history = []

    for step in range(100):
        # 模拟重构损失 (有些 token 难，有些容易)
        loss = torch.rand(1).item() * 2.0  # 0-2 范围

        # 决策
        should_adapt = threshold.should_adapt(loss)
        adaptation_history.append(should_adapt)
        threshold_history.append(threshold.current_threshold)

        # 更新阈值
        threshold.update(loss, should_adapt)

    # 统计
    actual_rate = sum(adaptation_history) / len(adaptation_history)

    print(f"\n目标适应率: 30%")
    print(f"实际适应率: {actual_rate*100:.1f}%")
    print(f"最终阈值: {threshold_history[-1]:.4f}")
    print(f"阈值变化: {threshold_history[0]:.4f} -> {threshold_history[-1]:.4f}")

def demo_ponder_gate():
    """演示 Ponder Gate"""
    print("\n" + "=" * 60)
    print("演示 2: Ponder Gate")
    print("=" * 60)

    modes = ['strict', 'balanced', 'lenient']

    for mode in modes:
        ponder = PonderGate(mode=mode)

        # 模拟不同难度的输入
        test_cases = [
            {'entropy': 0.5, 'max_prob': 0.9, 'loss': 0.1},   # 简单
            {'entropy': 1.5, 'max_prob': 0.5, 'loss': 0.5},   # 中等
            {'entropy': 2.5, 'max_prob': 0.2, 'loss': 1.0},   # 困难
        ]

        decisions = []
        for case in test_cases:
            should_ponder = ponder.decide(
                case['entropy'],
                case['max_prob'],
                case['loss']
            )
            decisions.append(should_ponder)

        print(f"\n{mode.upper()} 模式:")
        print(f"  简单输入: {'深度计算' if decisions[0] else '快速路径'}")
        print(f"  中等输入: {'深度计算' if decisions[1] else '快速路径'}")
        print(f"  困难输入: {'深度计算' if decisions[2] else '快速路径'}")

def demo_adaptive_pipeline():
    """演示完整的自适应流程"""
    print("\n" + "=" * 60)
    print("演示 3: 完整自适应流程")
    print("=" * 60)

    # 模拟模型输出
    vocab_size = 1000
    batch_size = 1
    seq_len = 20

    # 创建控制器
    threshold = EMAThreshold(target_rate=0.3)
    ponder = PonderGate(mode='balanced')

    print("\n模拟推理过程:\n")
    print(f"{'Token':<8} {'Loss':<10} {'Entropy':<10} {'MaxProb':<10} {'决策':<15}")
    print("-" * 60)

    stats = {'fast': 0, 'deep': 0}

    for i in range(seq_len):
        # 模拟模型输出
        logits = torch.randn(batch_size, vocab_size)

        # 计算信号
        loss = compute_reconstruction_loss(logits).item()
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        max_prob = probs.max().item()

        # 决策
        should_adapt = threshold.should_adapt(loss)
        should_ponder = ponder.decide(entropy, max_prob, loss)

        # 最终决策 (两者都同意才深度计算)
        final_decision = should_adapt and should_ponder

        decision_str = '深度计算 (qTTT)' if final_decision else '快速路径'
        stats['deep' if final_decision else 'fast'] += 1

        print(f"{i:<8} {loss:<10.4f} {entropy:<10.4f} {max_prob:<10.4f} {decision_str:<15}")

        # 更新阈值
        threshold.update(loss, should_adapt)

    print("\n统计:")
    print(f"  快速路径: {stats['fast']} ({stats['fast']/seq_len*100:.1f}%)")
    print(f"  深度计算: {stats['deep']} ({stats['deep']/seq_len*100:.1f}%)")

def demo_tradeoff_analysis():
    """分析不同目标率的权衡"""
    print("\n" + "=" * 60)
    print("演示 4: 目标率权衡分析")
    print("=" * 60)

    target_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    print(f"\n{'目标率':<10} {'估计延迟':<15} {'估计质量':<15}")
    print("-" * 40)

    for rate in target_rates:
        # 假设深度计算比快速路径慢 3 倍
        base_latency = 1.0
        deep_latency = 3.0

        expected_latency = (1 - rate) * base_latency + rate * deep_latency

        # 假设深度计算提升 20% 质量
        base_quality = 1.0
        deep_quality = 1.2

        expected_quality = (1 - rate) * base_quality + rate * deep_quality

        print(f"{rate*100:.0f}%{'':<7} {expected_latency:<15.2f}x {expected_quality:<15.2f}x")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("动态门控教程")
    print("=" * 60 + "\n")

    demo_ema_threshold()
    demo_ponder_gate()
    demo_adaptive_pipeline()
    demo_tradeoff_analysis()

    print("\n" + "=" * 60)
    print("教程完成！")
    print("=" * 60)
    print("\n下一步: 学习教程 6 - 端到端训练")
```

---

## 6. 配置参数详解

### EMAThreshold

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_rate` | float | 0.3 | 目标适应率 |
| `ema_decay` | float | 0.99 | EMA 衰减系数 |
| `initial_threshold` | float | 0.5 | 初始阈值 |

### PonderGate

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | str | 'balanced' | 模式: strict/balanced/lenient |
| `entropy_weight` | float | 0.4 | 熵信号权重 |
| `maxprob_weight` | float | 0.3 | 最大概率权重 |
| `reconstruction_weight` | float | 0.3 | 重构损失权重 |

---

## 7. 练习

### 练习 1: 自定义门控策略

```python
class CustomGating:
    def __init__(self):
        self.uncertainty_history = []

    def should_adapt(self, loss, entropy):
        # 基于历史不确定性做决策
        self.uncertainty_history.append((loss, entropy))
        # 实现你的逻辑
```

### 练习 2: 可视化阈值变化

```python
import matplotlib.pyplot as plt

threshold_history = []
# ... 运行推理 ...

plt.plot(threshold_history)
plt.xlabel('Step')
plt.ylabel('Threshold')
plt.title('EMA Threshold Adaptation')
plt.show()
```

---

## 8. 常见问题

**Q: 如何确定目标适应率？**
A: 根据延迟预算和质量要求权衡。通常 20-30% 是较好的起点。

**Q: EMA 衰减率如何影响适应？**
A: 高衰减 (0.99): 平滑但响应慢；低衰减 (0.9): 响应快但波动大。

**Q: Ponder Gate 与简单阈值有什么区别？**
A: Ponder Gate 综合多个信号，比单一阈值更鲁棒。

---

## 9. 下一步

完成本教程后，你应该能够：
- [x] 理解动态门控的概念
- [x] 使用重构损失作为难度信号
- [x] 配置阈值校准策略
- [x] 使用 Ponder Gate

**下一步**: [教程 6: 端到端训练](tutorial-06-end-to-end.md)

---

## 参考

- [Gating API 文档](../reference/api/gating.md)
- [论文 §5.6](../papers/adn-paper.md#56-dynamic-gating)
- [源码](../../src/gating/)
