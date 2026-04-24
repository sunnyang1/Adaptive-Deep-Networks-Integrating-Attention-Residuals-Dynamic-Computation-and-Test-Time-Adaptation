---
title: "术语表"
description: "ADN 专业术语解释"
category: "reference"
difficulty: "beginner"
last_updated: "2026-04-24"
---

# 术语表

本文档解释 ADN 项目中使用的专业术语。

---

## A

### Adaptive Deep Networks (ADN)
自适应深度网络。本项目研究的模块化 Transformer 架构，集成注意力残差、动态计算和测试时训练。

### Attention Residuals (AttnRes)
注意力残差。一种块级注意力机制，通过伪查询在块间传递信息，减少内存复杂度。

---

## B

### Block
块。AttnRes 中的基本单元，包含若干层 Transformer。N 个块组成完整模型。

### Block Attention Residuals
块注意力残差。见 [AttnRes](#attention-residuals-attnres)。

---

## D

### Depth Attention
深度注意力。AttnRes 中的注意力机制，在块内沿深度方向计算注意力。

### Dynamic Computation
动态计算。根据输入难度自适应调整计算量的机制，通过门控实现。

---

## E

### EMA Threshold
指数移动平均阈值。一种动态阈值校准策略，使用 EMA 平滑阈值变化。

### Engram
记忆印迹。显式的 n-gram 记忆机制，用于增强长距离依赖建模。

---

## F

### Frozen KV Cache
冻结 KV 缓存。qTTT 的核心设计，预填充后 Key 和 Value 不再更新。

---

## G

### Gating
门控。动态控制计算量的机制，基于重构损失决定何时进行适配。

---

## H

### Head Dimension (head_dim)
头维度。每个注意力头的维度，等于 hidden_dim / num_heads。

### Hidden Dimension (hidden_dim)
隐藏层维度。Transformer 层的隐藏状态维度。

---

## I

### Inter-block Attention
块间注意力。AttnRes 第一阶段，在块之间计算注意力。

### Intra-block Attention
块内注意力。AttnRes 第二阶段，在块内沿深度方向计算注意力。

---

## K

### KV Cache
Key-Value 缓存。Transformer 推理时缓存的 Key 和 Value，避免重复计算。

---

## L

### Layer
层。Transformer 的基本单元，包含注意力层和 MLP 层。

---

## M

### Margin Maximization Loss
边际最大化损失。qTTT 使用的损失函数，显式最大化正确 token 与其他 token 的 logit 差距。

### MLP
多层感知机。Transformer 中的前馈网络，通常使用 SwiGLU 激活。

---

## N

### Needle-in-Haystack
针在干草堆。长上下文评估基准，测试模型在极长文本中检索特定信息的能力。

### N-gram
N 元组。连续的 N 个 token，Engram 使用多粒度 n-gram 增强记忆。

### Num Blocks (num_blocks)
块数量。AttnRes 中的块数，论文推荐 N=8。

### Num Heads (num_heads)
头数量。注意力机制中的并行头数。

### Num Layers (num_layers)
层数量。Transformer 的总层数。

---

## P

### Phase 1 / Phase 2
阶段 1 / 阶段 2。AttnRes 的两阶段计算：Phase 1 块间并行，Phase 2 块内顺序。

### Ponder Gate
思考门控。基于不确定性的条件 qTTT 触发机制。

### Pseudo-Query
伪查询。AttnRes 中可学习的深度检索向量，用于块间信息传递。

---

## Q

### qTTT (Query-only Test-Time Training)
仅查询测试时训练。推理时仅更新查询参数的训练方法，冻结 KV 缓存。

### Query Adaptation
查询适配。qTTT 的核心操作，根据当前输入调整查询参数。

---

## R

### RaBitQ
Rapid and Accurate Bit-level Quantization。快速精确的位级量化，用于 KV 缓存压缩。

### Reconstruction Loss
重构损失。门控机制使用的信号，衡量模型重构输入的难度。

### RMSNorm
Root Mean Square Layer Normalization。AttnRes 使用的归一化方法。

---

## S

### Score Dilution
分数稀释。长上下文中注意力分数被稀释的现象，qTTT 旨在缓解此问题。

### SwiGLU
Swish-Gated Linear Unit。ADN 使用的 MLP 激活函数。

---

## T

### Target Rate
目标率。门控机制的目标适配比例，默认 30%。

### Test-Time Training
测试时训练。在推理阶段进行训练的技术，qTTT 是其特例。

### Threshold Calibration
阈值校准。动态调整门控阈值以维持目标适配率。

### Two-Phase
两阶段。AttnRes 的计算模式，分为 Phase 1 (块间) 和 Phase 2 (块内)。

---

## V

### Vocabulary Size (vocab_size)
词表大小。模型支持的 token 数量。

---

## 常用缩写

| 缩写 | 全称 | 中文 |
|------|------|------|
| ADN | Adaptive Deep Networks | 自适应深度网络 |
| AttnRes | Attention Residuals | 注意力残差 |
| qTTT | Query-only Test-Time Training | 仅查询测试时训练 |
| KV | Key-Value | 键值 |
| MLP | Multi-Layer Perceptron | 多层感知机 |
| EMA | Exponential Moving Average | 指数移动平均 |
| FLOP | Floating Point Operation | 浮点运算 |
| SRAM | Static Random Access Memory | 静态随机存取存储器 |
| HBM | High Bandwidth Memory | 高带宽内存 |

---

## 中英文对照

| 英文 | 中文 | 备注 |
|------|------|------|
| Attention Residuals | 注意力残差 | AttnRes |
| Block | 块 | AttnRes 中的块结构 |
| Depth Attention | 深度注意力 | 沿层方向的注意力 |
| Dynamic Computation | 动态计算 | 自适应计算量 |
| Engram | 记忆印迹 | n-gram 记忆机制 |
| Frozen KV Cache | 冻结 KV 缓存 | qTTT 的核心设计 |
| Gating | 门控 | 动态控制机制 |
| Margin Maximization | 边际最大化 | 损失函数类型 |
| Pseudo-Query | 伪查询 | 可学习的查询向量 |
| Query Adaptation | 查询适配 | qTTT 的操作 |
| Reconstruction Loss | 重构损失 | 门控信号 |
| Score Dilution | 分数稀释 | 长上下文问题 |
| Test-Time Training | 测试时训练 | 推理时训练 |
| Threshold Calibration | 阈值校准 | 门控参数调整 |

---

## 相关资源

- [技术文档](../TECHNICAL_DOCUMENTATION.md) - 综合技术文档
- [API 参考](api/) - 详细 API 文档
- [论文](../papers/) - 原始论文

---

*术语表持续更新中。如有遗漏，请提交 Issue 补充。*
