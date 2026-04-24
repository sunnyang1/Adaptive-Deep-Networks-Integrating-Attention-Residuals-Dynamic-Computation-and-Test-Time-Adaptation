# Adaptive Deep Networks (ADN) 技术文档

> **版本**: 0.2.0
> **最后更新**: 2026-04-23
> **Python**: >=3.12
> **PyTorch**: >=2.0.0

---

## 目录

1. [项目概览](#1-项目概览)
2. [架构设计](#2-架构设计)
3. [核心模块](#3-核心模块)
   - 3.1 [AttnRes (块注意力残差)](#31-attnres-块注意力残差)
   - 3.2 [qTTT (查询时训练)](#32-qttt-查询时训练)
   - 3.3 [RaBitQ (KV缓存压缩)](#33-rabitq-kv缓存压缩)
   - 3.4 [Engram (n-gram记忆)](#34-engram-n-gram记忆)
   - 3.5 [Gating (动态门控)](#35-gating-动态门控)
4. [模型配置](#4-模型配置)
5. [训练指南](#5-训练指南)
6. [实验与评估](#6-实验与评估)
7. [API参考](#7-api参考)
8. [故障排除](#8-故障排除)

---

## 1. 项目概览

Adaptive Deep Networks (ADN) 是一个用于高效长上下文推理的模块化Transformer架构研究代码库。本项目实现了论文 *"Adaptive Deep Networks: Integrating Attention Residuals, Dynamic Computation, and Test-Time Adaptation"* 中提出的核心方法。

### 1.1 核心特性

| 特性 | 描述 | 论文章节 |
|------|------|----------|
| **AttnRes** | 块级注意力残差，将内存从 O(Ld) 降至 O(Nd) | §3-4 |
| **qTTT** | 仅查询参数的测试时训练，冻结KV缓存 | §5 |
| **RaBitQ** | 快速精确的位级KV缓存量化 (~32x压缩) | §6 |
| **Engram** | 显式n-gram记忆机制 | §7 |
| **Gating** | 基于重构损失的动态计算门控 | §8 |

### 1.2 技术栈

```
Python >= 3.12
PyTorch >= 2.0.0
Transformers >= 4.35.0
NumPy >= 1.24.0
```

### 1.3 快速开始

```bash
# 安装
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v --tb=short --ignore=tests/legacy

# 训练小模型
python3 scripts/training/train_model.py --model-size small --output-dir results/small

# 运行实验
make quick    # 快速验证
make full     # 完整实验套件
make validate # 验证实验
```

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Adaptive Transformer                      │
├─────────────────────────────────────────────────────────────┤
│  Input Embeddings + Positional Encoding                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Layer 1    │→ │  Layer 2    │→ │  Layer N    │         │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │         │
│  │ │ AttnRes │ │  │ │ AttnRes │ │  │ │ AttnRes │ │         │
│  │ │ + qTTT  │ │  │ │ + qTTT  │ │  │ │ + qTTT  │ │         │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │         │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │         │
│  │ │  MLP    │ │  │ │  MLP    │ │  │ │  MLP    │ │         │
│  │ │ (SwiGLU)│ │  │ │ (SwiGLU)│ │  │ │ (SwiGLU)│ │         │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  Output Projection (LM Head)                                │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模块交互

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  AttnRes │←────│  qTTT    │←────│  Gating  │
│  (核心)   │     │ (适配)   │     │ (控制)   │
└────┬─────┘     └──────────┘     └──────────┘
     │
     ↓
┌──────────┐     ┌──────────┐
│  RaBitQ  │←────│  Engram  │
│ (压缩)   │     │ (记忆)   │
└──────────┘     └──────────┘
```

---

## 3. 核心模块

### 3.1 AttnRes (块注意力残差)

**位置**: `src/attnres/`

AttnRes通过块级表示的两阶段注意力机制减少内存复杂度。

#### 3.1.1 核心概念

- **块结构**: 将L层分为N个块，每块S=L/N层
- **两阶段注意力**:
  - Phase 1: 块间并行注意力 (inter-block)
  - Phase 2: 块内顺序注意力 (intra-block)
- **伪查询 (Pseudo-Queries)**: 可学习的深度检索向量

#### 3.1.2 关键设计决策

| 设计选择 | 决策 | 理由 |
|----------|------|------|
| 块数量 N | 8 | 论文§5.3: N≈8恢复大部分FullAttnRes收益 |
| 伪查询初始化 | 零初始化 | 训练稳定性 |
| Key归一化 | RMSNorm | 无此: +0.006/+0.004 loss |
| 深度注意力头数 | 单头 | 多头性能下降(1.752 vs 1.746) |
| 归一化函数 | Softmax | 强制更尖锐的选择 |

#### 3.1.3 API使用

```python
from src.attnres import BlockAttnRes, PseudoQueryManager

# 创建块注意力残差模块
attn_res = BlockAttnRes(
    hidden_dim=1408,
    num_heads=8,
    num_blocks=8,
    num_layers=32
)

# 前向传播
output = attn_res(hidden_states, layer_idx=0)
```

#### 3.1.4 配置参数

```python
@dataclass
class AttnResConfig:
    num_blocks: int = 8          # 块数量
    use_rmsnorm_keys: bool = True # Key RMSNorm
    single_head_depth: bool = True # 单头深度注意力
    use_softmax: bool = True     # Softmax vs Sigmoid
```

---

### 3.2 qTTT (查询时训练)

**位置**: `src/qttt/`

Query-only Test-Time Training: 在推理时仅更新查询参数，冻结KV缓存。

#### 3.2.1 核心概念

- **冻结KV**: 预填充阶段的Key和Value永不改变
- **仅查询更新**: 只有查询参数被更新
- **边际最大化损失**: 显式最大化logit边际

#### 3.2.2 关键设计决策

| 特性 | 实现 | 说明 |
|------|------|------|
| 冻结KV缓存 | ✓ | 预填充后KV固定 |
| 查询自适应 | ✓ | 仅更新查询投影 |
| 边际损失 | ✓ | 显式logit边际最大化 |
| 极坐标适配 | ✓ | 50%参数减少的变体 |

#### 3.2.3 API使用

```python
from src.qttt import QueryOnlyTTT, qTTTConfig, MarginMaximizationLoss

# 配置
config = qTTTConfig(
    max_steps=32,
    learning_rate=0.005,
    span_length=128
)

# 创建qTTT适配器
qttt = QueryOnlyTTT(config)

# 适配步骤
adapted_query = qttt.adapt(query, kv_cache, target_positions)
```

#### 3.2.4 边际最大化损失

```python
from src.qttt import MarginMaximizationLoss

# 创建损失函数
margin_loss = MarginMaximizationLoss(
    margin=0.5,           # 边际阈值
    temperature=1.0,      # 温度系数
    hard_negative_weight=1.0  # 困难负样本权重
)

# 计算损失
loss = margin_loss(logits, target_positions, distractor_positions)
```

---

### 3.3 RaBitQ (KV缓存压缩)

**位置**: `src/rabitq/`

RaBitQ (Rapid and Accurate Bit-level Quantization) 实现SIGMOD 2024/2025论文中的KV缓存压缩方法。

#### 3.3.1 核心概念

- **随机正交旋转**: FWHT-Kac或QR-based旋转
- **1-bit二进制量化**: 可选扩展位精化
- **每向量因子计算**: 无偏内积估计
- **Popcount非对称距离估计**

#### 3.3.2 推荐配置

| 函数 | 位宽 | 压缩比 | 适用场景 |
|------|------|--------|----------|
| `create_k1()` | 1-bit | ~32x | 最大速度 ⭐ |
| `create_k2()` | 2-bit | ~16x | 平衡速度/精度 |
| `create_k3()` | 3-bit | ~10x | 高精度需求 |

#### 3.3.3 API使用

```python
from src.rabitq import create_k1, RaBitQCache

# 创建RaBitQ量化器 (1-bit, ~32x压缩)
rq = create_k1(head_dim=64)

# 拟合样本数据
rq.fit(sample_keys, sample_values)

# 压缩
compressed = rq.compress(keys, values)

# 解压缩
keys_dq, values_dq = rq.decompress(compressed)

# HuggingFace兼容缓存
cache = rq.as_cache(residual_window=128)
```

#### 3.3.4 低层组件

```python
from src.rabitq import (
    FhtKacRotator,      # FWHT-Kac旋转
    MatrixRotator,      # 矩阵旋转
    QuantizedVector,    # 量化向量
    RaBitQCache,        # 压缩缓存
)
```

---

### 3.4 Engram (n-gram记忆)

**位置**: `src/engram/`

显式n-gram记忆机制，增强长距离依赖建模。

#### 3.4.1 核心概念

- **n-gram哈希**: 高效的多粒度哈希映射
- **多头嵌入**: 多头部n-gram表示
- **短卷积**: 局部特征提取
- **压缩分词器**: 与tokenizer对接

#### 3.4.2 配置预设

```python
from src.engram import EngramSmallConfig, EngramMediumConfig, EngramLargeConfig

# 小模型配置
engram_config = EngramSmallConfig()

# 中模型配置
engram_config = EngramMediumConfig()

# 大模型配置
engram_config = EngramLargeConfig()
```

#### 3.4.3 API使用

```python
from src.engram import Engram, EngramConfig

# 创建Engram模块
config = EngramConfig(
    vocab_size=32000,
    embedding_dim=1408,
    ngram_sizes=[2, 3, 4],
    num_heads=8
)

engram = Engram(config)

# 前向传播
output = engram(input_ids, hidden_states)
```

---

### 3.5 Gating (动态门控)

**位置**: `src/gating/`

基于重构损失的动态计算门控系统。

#### 3.5.1 核心概念

- **信号**: 重构损失作为难度代理
- **校准**: EMA或目标率阈值
- **Ponder Gate**: 基于不确定性的条件qTTT触发
- **深度优先**: RaBitQ感知的深度优先控制器

#### 3.5.2 阈值校准策略

| 类 | 策略 | 说明 |
|----|------|------|
| `EMAThreshold` | 指数移动平均 | 自适应阈值 |
| `TargetRateThreshold` | 目标率维持 | 保持目标适配率 |
| `HybridThreshold` | 混合策略 | 结合EMA和目标率 |

#### 3.5.3 API使用

```python
from src.gating import (
    EMAThreshold,
    TargetRateThreshold,
    GatingController,
    PonderGate
)

# EMA阈值
ema = EMAThreshold(
    target_rate=0.3,
    ema_decay=0.99,
    adjustment_rate=0.01
)

# Ponder Gate
ponder = PonderGate(
    mode='balanced',  # 'strict', 'balanced', 'lenient'
    entropy_threshold=1.0,
    max_prob_threshold=0.9
)

# 门控决策
should_adapt = ponder.should_adapt(logits, hidden_states)
```

---

## 4. 模型配置

### 4.1 预定义配置

| 配置 | 参数量 | 层数 | 隐藏层 | 头数 | 块数 | d_model/L_b | H/L_b |
|------|--------|------|--------|------|------|-------------|-------|
| `AttnResSmallConfig` | 1.1B | 32 | 1408 | 8 | 8 | 44.0 | 0.25 |
| `AttnResMediumConfig` | 5.7B | 56 | 2496 | 16 | 8 | 44.6 | 0.29 |
| `AttnResLargeConfig` | 23.0B | 88 | 4032 | 18 | 11 | 45.8 | 0.20 |
| `AttnResT4Config` | ~125M | 14 | 640 | 4 | 7 | 45.7 | 0.286 |

### 4.2 架构优化原则

基于论文§5.4.1，AttnRes的最优架构参数：

- **d_model/L_b ≈ 45**: AttnRes将最优值从~60移至~45
- **H/L_b ≈ 0.3**: 头数与层数比
- **更深的网络**: 偏好更深、更窄的网络

### 4.3 使用配置

```python
from src.models import ModelConfig, AttnResSmallConfig

# 使用预定义配置
config = AttnResSmallConfig()

# 或自定义
config = ModelConfig(
    num_layers=32,
    hidden_dim=1408,
    num_heads=8,
    num_blocks=8,
    max_qttt_steps=16,
    gating_target_rate=0.3
)
```

---

## 5. 训练指南

### 5.1 训练配置

```python
from src.models.configs import TrainingConfig

training_config = TrainingConfig(
    batch_size_tokens=4_000_000,  # 4M tokens
    learning_rate=3e-4,
    lr_schedule='cosine',
    warmup_steps=2000,
    weight_decay=0.1,
    gradient_clipping=1.0,
    mixed_precision='bf16'
)
```

### 5.2 训练命令

```bash
# 基础训练
python3 scripts/training/train_model.py \
    --model-size small \
    --output-dir results/small

# 特定模型大小
python3 scripts/training/train_small.py --output-dir results/small
python3 scripts/training/train_medium.py --output-dir results/medium
python3 scripts/training/train_large.py --output-dir results/large

# 使用DeepSpeed
python3 scripts/training/train_h20.py --output-dir results/h20

# 流式训练
python3 scripts/training/train_streaming.py --output-dir results/streaming
```

### 5.3 Makefile训练目标

```bash
# 论文对齐的训练 (自动检查对齐)
make train-paper-small OUTPUT_DIR=results/small_paper
make train-paper-medium OUTPUT_DIR=results/medium_paper
make train-paper-large OUTPUT_DIR=results/large_paper
```

---

## 6. 实验与评估

### 6.1 实验框架

```bash
# 列出所有实验
make list

# 快速模式 (减少样本数)
make quick

# 完整实验
make full

# 核心实验
make core

# 验证实验
make validate

# 论文指标
make paper-metrics
```

### 6.2 核心实验

| 实验 | 描述 | 论文对应 |
|------|------|----------|
| 表示埋葬 | 验证表示埋葬问题 | §3 |
| 边际分析 | 查询边际分布分析 | §4 |
| 梯度流 | AttnRes梯度流分析 | §5 |
| FLOP等效 | FLOP等效性验证 | §6 |
| 协同效应 | 组件协同效应 | §7 |
| 辅助头 | 辅助头分析 | §8 |

### 6.3 验证实验 (Table 1-9)

```bash
# 运行所有验证
python3 experiments/validation/run_all_validations.py

# 单独验证
python3 experiments/validation/table1_representation_burial.py
python3 experiments/validation/table2_gradient_flow.py
python3 experiments/validation/table3_rabitq_space_accuracy.py
python3 experiments/validation/table4_needle_haystack.py
python3 experiments/validation/table5_query_margin.py
python3 experiments/validation/table6_math.py
python3 experiments/validation/table7_synergy.py
python3 experiments/validation/table8_sram_allocation.py
python3 experiments/validation/table9_coupling_effect.py
```

### 6.4 基准测试

```bash
# Needle-in-Haystack
python3 src/benchmarks/needle_haystack.py

# MATH评估
python3 src/benchmarks/math_eval.py

# FLOP分析
python3 src/benchmarks/flop_analysis.py
```

---

## 7. API参考

### 7.1 主要入口点

```python
# 模型
from src.models import ModelConfig
from src.models.adaptive_transformer import AdaptiveTransformer

# AttnRes
from src.attnres import BlockAttnRes, PseudoQueryManager

# qTTT
from src.qttt import QueryOnlyTTT, qTTTConfig, MarginMaximizationLoss

# RaBitQ
from src.rabitq import create_k1, create_k2, create_k3, RaBitQCache

# Engram
from src.engram import Engram, EngramConfig

# Gating
from src.gating import EMAThreshold, PonderGate, GatingController
```

### 7.2 增量生成

```python
from src.models.incremental_generator import IncrementalGenerator
from src.models.incremental_state import IncrementalState

# 创建生成器
generator = IncrementalGenerator(model, config)

# 预填充
state = generator.prefill(input_ids)

# 逐步生成
for _ in range(max_new_tokens):
    token, state = generator.step(state)
```

### 7.3 模型配置注册表

```python
from src.models.configs import get_config, CONFIGS

# 获取配置
config = get_config('small')   # AttnResSmallConfig
config = get_config('medium')  # AttnResMediumConfig
config = get_config('large')   # AttnResLargeConfig
config = get_config('t4')      # AttnResT4Config

# 查看可用配置
print(CONFIGS.keys())  # dict_keys(['t4', 'small', 'medium', 'large'])
```

---

## 8. 故障排除

### 8.1 常见问题

#### 测试失败

```bash
# 忽略遗留测试 (推荐)
pytest tests/ -v --tb=short --ignore=tests/legacy

# 仅单元测试
pytest tests/unit/ -v --tb=short
```

#### 导入错误

```python
# 确保repo根目录在PYTHONPATH
import sys
sys.path.insert(0, '/path/to/Adaptive-Deep-Networks')

# 然后导入
from src.attnres import BlockAttnRes
```

#### 内存不足

```python
# 使用梯度检查点
config.use_gradient_checkpointing = True

# 减少批次大小
config.batch_size_tokens = 2_000_000

# 使用T4配置
config = AttnResT4Config()
```

### 8.2 已知问题

1. **`tests/legacy/` 已弃用** — 始终传递 `--ignore=tests/legacy` 给pytest
2. **Mypy预存在错误** — `mypy src/` 可能报告模块名重复问题
3. **统一实验运行器CLI不匹配** — 某些实验脚本期望 `--output_dir` 而非 `--output-dir`
4. **使用 `python3` 而非 `python`** — 后者默认未链接

### 8.3 环境检查

```bash
# 运行环境检查
python3 scripts/setup/check_env.py

# 或快速检查
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## 9. 参考

- **论文草稿**: `ADN_paper.md` (根目录), `matdo-e_paper.md` (根目录)
- **项目结构**: `PROJECT_ORGANIZATION.md`
- **架构图**: `docs/ARCHITECTURE.md`
- **Agent指南**: `AGENTS.md`

---

## 10. 许可证

MIT License - 详见 `LICENSE` 文件。

---

*本文档由技术文档工程师Agent生成，基于项目实际代码结构和AGENTS.md指南。*
