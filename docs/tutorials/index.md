---
title: "教程系列"
description: "ADN 手把手教程，从入门到精通"
category: "tutorials"
difficulty: "beginner"
last_updated: "2026-04-24"
---

# 教程系列

欢迎来到 ADN 教程系列！这里提供循序渐进的学习路径，帮助你掌握 ADN 的各个模块。

---

## 🎯 学习路径

```
教程 1 ──> 教程 2 ──> 教程 3 ──> 教程 4 ──> 教程 5 ──> 教程 6
AttnRes    qTTT      RaBitQ    Engram    Gating    端到端
(30分钟)   (30分钟)  (30分钟)  (45分钟)  (45分钟)  (60分钟)
```

---

## 📚 教程列表

### [教程 1: AttnRes 入门](tutorial-01-attnres.md)
**难度**: ⭐
**时间**: 30分钟
**前置**: Python, PyTorch 基础

**你将学习**:
- 什么是块注意力残差 (AttnRes)
- 如何创建和配置 AttnRes 模块
- 理解伪查询 (Pseudo-Queries) 的概念
- 运行简单的注意力实验

**代码示例**:
```python
from src.attnres import BlockAttnRes

# 创建 AttnRes 模块
attn_res = BlockAttnRes(
    hidden_dim=512,
    num_heads=8,
    num_blocks=8
)
```

---

### [教程 2: qTTT 使用](tutorial-02-qttt.md)
**难度**: ⭐
**时间**: 30分钟
**前置**: 完成教程 1

**你将学习**:
- 查询时训练 (qTTT) 的基本概念
- 如何配置 qTTT 参数
- 冻结 KV 缓存的工作原理
- 使用边际最大化损失

**代码示例**:
```python
from src.qttt import QueryOnlyTTT, qTTTConfig

# 配置 qTTT
config = qTTTConfig(max_steps=32, learning_rate=0.005)
ttt = QueryOnlyTTT(config)
```

---

### [教程 3: RaBitQ 压缩](tutorial-03-rabitq.md)
**难度**: ⭐
**时间**: 30分钟
**前置**: 完成教程 1

**你将学习**:
- RaBitQ 压缩原理
- 如何选择压缩配置 (k1/k2/k3)
- 压缩和解压缩 KV 缓存
- 与 HuggingFace 集成

**代码示例**:
```python
from src.rabitq import create_k1

# 创建 1-bit 量化器 (~32x 压缩)
rq = create_k1(head_dim=64)
compressed = rq.compress(keys, values)
```

---

### [教程 4: Engram 记忆](tutorial-04-engram.md)
**难度**: ⭐⭐
**时间**: 45分钟
**前置**: 完成教程 1-3

**你将学习**:
- n-gram 记忆机制原理
- Engram 模块的配置和使用
- 多粒度哈希映射
- 与 Transformer 集成

**代码示例**:
```python
from src.engram import Engram, EngramConfig

# 创建 Engram 模块
config = EngramConfig(ngram_sizes=[2, 3, 4])
engram = Engram(config)
```

---

### [教程 5: 动态门控](tutorial-05-gating.md)
**难度**: ⭐⭐
**时间**: 45分钟
**前置**: 完成教程 1-3

**你将学习**:
- 动态计算门控的概念
- 重构损失作为难度信号
- 阈值校准策略 (EMA, Target Rate)
- Ponder Gate 的使用

**代码示例**:
```python
from src.gating import EMAThreshold, PonderGate

# 创建门控控制器
threshold = EMAThreshold(target_rate=0.3)
ponder = PonderGate(mode='balanced')
```

---

### [教程 6: 端到端训练](tutorial-06-end-to-end.md)
**难度**: ⭐⭐
**时间**: 60分钟
**前置**: 完成教程 1-5

**你将学习**:
- 整合所有模块的训练流程
- 配置完整的 ADN 模型
- 训练循环的实现
- 监控和调试技巧

**代码示例**:
```python
from src.models import AdaptiveTransformer, AttnResSmallConfig

# 创建完整模型
config = AttnResSmallConfig()
model = AdaptiveTransformer(config)

# 训练循环
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

---

## 🎓 学习建议

### 初学者路径

如果你是第一次接触 ADN：

1. **按顺序完成教程 1-3** (约 90 分钟)
   - 掌握三个核心模块的基本使用

2. **选择兴趣方向**:
   - 对记忆机制感兴趣 → 教程 4 (Engram)
   - 对动态计算感兴趣 → 教程 5 (Gating)

3. **完成教程 6** (约 60 分钟)
   - 整合所有知识

### 研究者路径

如果你是研究人员，关注特定模块：

| 研究兴趣 | 推荐教程 | 补充阅读 |
|----------|----------|----------|
| 注意力机制 | 教程 1 | [AttnRes 论文](../papers/) |
| 测试时适应 | 教程 2 | [QTTT 论文](../papers/) |
| 模型压缩 | 教程 3 | [RaBitQ 指南](../guides/RABITQ_GUIDE.md) |
| 长上下文 | 教程 4 | [Engram 文档](../reference/api/engram.md) |
| 动态计算 | 教程 5 | [Gating 文档](../reference/api/gating.md) |

### 工程师路径

如果你是工程开发者，关注实际应用：

1. **快速浏览教程 1-3** (了解概念)
2. **重点学习教程 6** (端到端训练)
3. **查看 [操作指南](../how-to/)** (解决实际问题)

---

## ✅ 完成检查清单

完成所有教程后，你应该能够：

- [ ] 独立使用 AttnRes 模块
- [ ] 配置和运行 qTTT 适配
- [ ] 使用 RaBitQ 压缩 KV 缓存
- [ ] 集成 Engram 记忆机制
- [ ] 实现动态门控策略
- [ ] 搭建完整的训练流程

---

## 📖 相关资源

- [入门系列](../getting-started/) - 环境设置和基础概念
- [操作指南](../how-to/) - 解决特定问题的步骤
- [API 参考](../reference/api/) - 详细的 API 文档
- [解释文档](../explanation/) - 深入理解原理

---

准备好开始了吗？点击 [教程 1: AttnRes 入门](tutorial-01-attnres.md) 开始学习！
