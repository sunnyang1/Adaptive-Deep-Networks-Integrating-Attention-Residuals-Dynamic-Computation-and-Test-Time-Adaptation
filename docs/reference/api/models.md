---
title: "Models API"
description: "ADN 模型类和配置 API 参考"
category: "reference"
last_updated: "2026-04-24"
---

# Models API 参考

## AdaptiveTransformer

主模型类，整合所有 ADN 模块。

### 类定义

```python
class AdaptiveTransformer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        vocab_size: Optional[int] = None
    )
```

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | ModelConfig | 模型配置 |
| `vocab_size` | Optional[int] | 词汇表大小，覆盖 config 中的值 |

### 方法

#### forward

```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    use_cache: bool = False
) -> TransformerOutput
```

**参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `input_ids` | torch.Tensor | 输入 token IDs，形状 `[batch, seq_len]` |
| `attention_mask` | Optional[torch.Tensor] | 注意力掩码 |
| `labels` | Optional[torch.Tensor] | 用于计算损失的标签 |
| `output_attentions` | bool | 是否返回注意力权重 |
| `output_hidden_states` | bool | 是否返回隐藏状态 |
| `use_cache` | bool | 是否使用 KV 缓存 |

**返回**: `TransformerOutput` 对象，包含:
- `logits`: 预测 logits
- `loss`: 如果提供了 labels
- `attentions`: 如果 `output_attentions=True`
- `hidden_states`: 如果 `output_hidden_states=True`

#### generate

```python
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    use_qttt: Union[bool, str] = False,
    **kwargs
) -> torch.Tensor
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_ids` | torch.Tensor | - | 输入 token IDs |
| `max_new_tokens` | int | 100 | 最大生成 token 数 |
| `temperature` | float | 1.0 | 采样温度 |
| `top_p` | float | 1.0 | nucleus sampling 阈值 |
| `top_k` | int | 0 | top-k sampling (0=禁用) |
| `use_qttt` | bool/str | False | 是否使用 qTTT，或 'adaptive' |

---

## ModelConfig

基础模型配置类。

### 类定义

```python
@dataclass
class ModelConfig:
    # 架构
    num_layers: int = 32
    hidden_dim: int = 4096
    num_heads: int = 32
    mlp_ratio: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 32768

    # AttnRes
    num_blocks: int = 8

    # qTTT
    max_qttt_steps: int = 32
    qttt_span_length: int = 128
    qttt_learning_rate: float = 0.005

    # Gating
    gating_target_rate: float = 0.3

    # Engram
    use_engram: bool = False
    engram_config: Optional[EngramConfig] = None

    # Training
    dropout: float = 0.0
    attention_dropout: float = 0.0
```

### 预定义配置

#### AttnResSmallConfig

1.1B 参数模型配置。

```python
config = AttnResSmallConfig()
# 等效于:
# num_layers=32, hidden_dim=1408, num_heads=8, num_blocks=8
```

#### AttnResMediumConfig

5.7B 参数模型配置。

```python
config = AttnResMediumConfig()
# 等效于:
# num_layers=56, hidden_dim=2496, num_heads=16, num_blocks=8
```

#### AttnResLargeConfig

23B 参数模型配置。

```python
config = AttnResLargeConfig()
# 等效于:
# num_layers=88, hidden_dim=4032, num_heads=18, num_blocks=11
```

#### AttnResT4Config

T4 GPU 友好的小模型配置 (~125M 参数)。

```python
config = AttnResT4Config()
# 等效于:
# num_layers=14, hidden_dim=640, num_heads=4, num_blocks=7
```

---

## AdaptiveLayer

单个 Transformer 层，包含注意力、MLP 和 AttnRes。

### 类定义

```python
class AdaptiveLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int
    )
```

### 方法

#### forward

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    use_cache: bool = False
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]
```

---

## AdaptiveAttention

自适应注意力模块，支持 AttnRes。

### 类定义

```python
class AdaptiveAttention(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int
    )
```

### 方法

#### forward

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    use_cache: bool = False,
    output_attentions: bool = False
) -> Tuple[torch.Tensor, ...]
```

---

## IncrementalGenerator

增量生成器，用于高效自回归生成。

### 类定义

```python
class IncrementalGenerator:
    def __init__(
        self,
        model: AdaptiveTransformer,
        config: Optional[GenerationConfig] = None
    )
```

### 方法

#### prefill

```python
def prefill(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor
```

初始化生成状态，处理提示词。

#### step

```python
def step(
    self,
    input_id: torch.Tensor,
    use_qttt: bool = False
) -> torch.Tensor
```

单步生成一个 token。

#### generate

```python
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    **kwargs
) -> torch.Tensor
```

完整生成序列。

---

## 使用示例

### 创建模型

```python
from src.models import AdaptiveTransformer, AttnResSmallConfig

config = AttnResSmallConfig()
model = AdaptiveTransformer(config)
```

### 前向传播

```python
import torch

input_ids = torch.randint(0, 32000, (2, 128))
outputs = model(input_ids)
logits = outputs.logits
```

### 生成文本

```python
# 基础生成
output = model.generate(input_ids, max_new_tokens=50)

# 启用 qTTT
output = model.generate(
    input_ids,
    max_new_tokens=50,
    use_qttt='adaptive'
)

# 使用采样
output = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.95
)
```

### 使用增量生成器

```python
from src.models import IncrementalGenerator

generator = IncrementalGenerator(model)

# Prefill
logits = generator.prefill(input_ids)

# 逐步生成
for _ in range(50):
    next_token_logits = generator.step(next_token)
    next_token = sample(next_token_logits)
```

---

## 参考

- [源码](../../src/models/)
- [配置详解](../config/models.md)
- [教程 6: 端到端训练](../../tutorials/tutorial-06-end-to-end.md)
