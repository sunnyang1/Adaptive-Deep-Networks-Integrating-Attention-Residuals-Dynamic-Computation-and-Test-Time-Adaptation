---
title: "配置参考"
description: "ADN 所有配置类和参数的完整参考"
category: "reference"
last_updated: "2026-04-24"
---

# 配置参考

本文档提供 ADN 所有配置类和参数的完整参考。

---

## 模型配置

### ModelConfig

基础模型配置。

```python
@dataclass
class ModelConfig:
    # 基础架构
    num_layers: int = 32
    hidden_dim: int = 4096
    num_heads: int = 32
    mlp_ratio: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 32768

    # AttnRes 配置
    num_blocks: int = 8

    # qTTT 配置
    max_qttt_steps: int = 32
    qttt_span_length: int = 128
    qttt_learning_rate: float = 0.005

    # Gating 配置
    gating_target_rate: float = 0.3

    # Engram 配置
    use_engram: bool = False
    engram_config: Optional[EngramConfig] = None

    # 训练配置
    dropout: float = 0.0
    attention_dropout: float = 0.0
```

### 预定义配置

| 配置类 | 参数量 | 层数 | 隐藏维度 | 头数 | 块数 |
|--------|--------|------|----------|------|------|
| `AttnResT4Config` | ~125M | 14 | 640 | 4 | 7 |
| `AttnResSmallConfig` | ~1.1B | 32 | 1408 | 8 | 8 |
| `AttnResMediumConfig` | ~5.7B | 56 | 2496 | 16 | 8 |
| `AttnResLargeConfig` | ~23B | 88 | 4032 | 18 | 11 |

---

## 训练配置

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    # 优化器
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    # 调度器
    lr_schedule: str = "cosine"  # "cosine", "linear", "constant"
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1

    # 训练
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_steps: int = 100000
    max_epochs: int = 3

    # 正则化
    dropout: float = 0.0
    attention_dropout: float = 0.0
    gradient_clipping: float = 1.0

    # 效率
    mixed_precision: str = "bf16"  # "fp16", "bf16", "fp32"
    gradient_checkpointing: bool = False
    compile: bool = False

    # 日志
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000

    # 路径
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
```

---

## qTTT 配置

### qTTTConfig

```python
@dataclass
class qTTTConfig:
    # 优化参数
    max_steps: int = 32
    learning_rate: float = 0.005
    span_length: int = 128

    # 损失函数
    use_margin_loss: bool = True
    margin_target: float = 2.0
    hard_negative_weight: float = 1.0

    # 早停
    use_early_stopping: bool = False
    patience: int = 5
    min_delta: float = 0.001

    # 预设模式
    @classmethod
    def fast(cls) -> "qTTTConfig":
        return cls(max_steps=8, learning_rate=0.01)

    @classmethod
    def balanced(cls) -> "qTTTConfig":
        return cls(max_steps=16, learning_rate=0.005)

    @classmethod
    def quality(cls) -> "qTTTConfig":
        return cls(max_steps=32, learning_rate=0.003)
```

---

## RaBitQ 配置

### RaBitQConfig

```python
@dataclass
class RaBitQConfig:
    head_dim: int = 128
    num_bits: int = 2  # 1, 2, or 3
    use_rotation: bool = True
    random_seed: int = 42

    # 预设工厂方法
    @classmethod
    def k1(cls, head_dim: int) -> "RaBitQConfig":
        """1-bit 配置 (~32x 压缩)"""
        return cls(head_dim=head_dim, num_bits=1)

    @classmethod
    def k2(cls, head_dim: int) -> "RaBitQConfig":
        """2-bit 配置 (~16x 压缩)"""
        return cls(head_dim=head_dim, num_bits=2)

    @classmethod
    def k3(cls, head_dim: int) -> "RaBitQConfig":
        """3-bit 配置 (~10.7x 压缩)"""
        return cls(head_dim=head_dim, num_bits=3)
```

---

## Engram 配置

### EngramConfig

```python
@dataclass
class EngramConfig:
    ngram_sizes: List[int] = field(default_factory=lambda: [2, 3, 4])
    embedding_dim: int = 512
    memory_size: int = 50000
    num_hashes: int = 3
    use_compression: bool = True
    compression_ratio: float = 0.5
```

---

## Gating 配置

### GatingConfig

```python
@dataclass
class GatingConfig:
    # 阈值策略
    strategy: str = "ema"  # "ema", "target_rate", "fixed"
    target_rate: float = 0.3

    # EMA 参数
    ema_decay: float = 0.99
    ema_initial_threshold: float = 0.5

    # Ponder Gate
    ponder_mode: str = "balanced"  # "strict", "balanced", "lenient"
    entropy_weight: float = 0.4
    maxprob_weight: float = 0.3
    reconstruction_weight: float = 0.3
```

---

## 验证配置

### ValidationConfig

```python
@dataclass
class ValidationConfig:
    # Needle-in-Haystack
    nih_context_lengths: List[int] = field(
        default_factory=lambda: [1024, 4096, 16384, 32768, 65536, 131072, 262144]
    )
    nih_depths_per_length: int = 10
    nih_num_trials: int = 5

    # MATH
    math_difficulty_levels: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    math_max_samples: Optional[int] = None

    # 效率
    measure_flops: bool = True
    measure_latency: bool = True
    measure_memory: bool = True

    # 输出
    output_dir: str = "./validation_results"
    save_attention_maps: bool = False
    save_detailed_logs: bool = True
```

---

## 配置示例

### 完整训练配置

```yaml
# config.yaml
model:
  name: small
  vocab_size: 32000

training:
  batch_size: 4
  learning_rate: 3e-4
  warmup_steps: 2000
  max_steps: 10000

qttt:
  max_steps: 16
  learning_rate: 0.005
  use_margin_loss: true

rabitq:
  num_bits: 2
  use_rotation: true

engram:
  enabled: true
  ngram_sizes: [2, 3, 4]
  memory_size: 50000

gating:
  strategy: ema
  target_rate: 0.3
```

### 代码中使用

```python
from src.models import get_config

# 加载配置
config = get_config("small")

# 修改配置
config.max_qttt_steps = 24
config.use_engram = True

# 创建模型
from src.models import AdaptiveTransformer
model = AdaptiveTransformer(config)
```

---

## 参考

- [模型 API](../api/models.md)
- [训练指南](../../how-to/train-model.md)
- [源码](../../src/models/configs.py)
