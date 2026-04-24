---
title: "AttnRes API"
description: "块注意力残差 (Block Attention Residuals) API 参考"
category: "reference"
last_updated: "2026-04-24"
---

# AttnRes API 参考

## BlockAttnRes

基础的块注意力残差模块。

### 类定义

```python
class BlockAttnRes(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int = 8,
        dropout: float = 0.0,
        use_bias: bool = False
    )
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_dim` | int | 必填 | 隐藏层维度 |
| `num_heads` | int | 必填 | 注意力头数 |
| `num_blocks` | int | 8 | 块数量 |
| `dropout` | float | 0.0 | Dropout 概率 |
| `use_bias` | bool | False | 是否使用偏置 |

### 方法

#### forward

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False
) -> Tuple[torch.Tensor, Optional[Dict]]
```

**返回**:
- `hidden_states`: 输出隐藏状态
- `stats`: 包含注意力统计信息的字典

---

## TwoPhaseBlockAttnRes

两阶段块注意力残差模块，分离块间和块内注意力。

### 类定义

```python
class TwoPhaseBlockAttnRes(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int = 8,
        use_rmsnorm: bool = True,
        dropout: float = 0.0
    )
```

### 方法

#### forward

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False
) -> Tuple[torch.Tensor, Dict]
```

**返回的 stats 包含**:
- `phase1_attn`: 阶段 1 (块间) 注意力权重
- `phase2_attn`: 阶段 2 (块内) 注意力权重
- `block_usage`: 各块的使用频率

---

## PseudoQueryGenerator

伪查询生成器。

### 类定义

```python
class PseudoQueryGenerator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_blocks: int,
        init_strategy: str = "zero"
    )
```

### 方法

#### forward

```python
def forward(
    self,
    block_representations: torch.Tensor
) -> torch.Tensor
```

---

## 使用示例

### 基础使用

```python
from src.attnres import BlockAttnRes

# 创建模块
attn_res = BlockAttnRes(
    hidden_dim=512,
    num_heads=8,
    num_blocks=8
)

# 前向传播
x = torch.randn(2, 128, 512)
output, stats = attn_res(x)
```

### 两阶段使用

```python
from src.attnres import TwoPhaseBlockAttnRes

attn_res = TwoPhaseBlockAttnRes(
    hidden_dim=768,
    num_heads=12,
    num_blocks=8,
    use_rmsnorm=True
)

output, stats = attn_res(x)
print(f"块间注意力: {stats['phase1_attn'].shape}")
print(f"块内注意力: {stats['phase2_attn'].shape}")
```

---

## 参考

- [源码](../../src/attnres/)
- [教程 1: AttnRes 入门](../../tutorials/tutorial-01-attnres.md)
