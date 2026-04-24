---
title: "qTTT API"
description: "查询时训练 (Query-Only Test-Time Training) API 参考"
category: "reference"
last_updated: "2026-04-24"
---

# qTTT API 参考

## QueryOnlyTTT

查询时训练主类。

### 类定义

```python
class QueryOnlyTTT(nn.Module):
    def __init__(
        self,
        config: qTTTConfig
    )
```

### 方法

#### adapt

```python
def adapt(
    self,
    query: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor],
    target: Optional[torch.Tensor] = None,
    max_steps: Optional[int] = None
) -> Tuple[torch.Tensor, Dict]
```

**参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `query` | torch.Tensor | 初始查询，形状 `[batch, heads, 1, head_dim]` |
| `kv_cache` | Tuple | (keys, values)，形状 `[batch, heads, seq_len, head_dim]` |
| `target` | Optional[torch.Tensor] | 目标 token ID |
| `max_steps` | Optional[int] | 最大优化步数，覆盖配置 |

**返回**:
- `adapted_query`: 适配后的查询
- `info`: 包含适配过程信息的字典

---

## MarginMaximizationLoss

边际最大化损失函数。

### 类定义

```python
class MarginMaximizationLoss(nn.Module):
    def __init__(
        self,
        margin_target: float = 2.0,
        hard_negative_weight: float = 1.0
    )
```

### 方法

#### forward

```python
def forward(
    self,
    logits: torch.Tensor,
    targets: torch.Tensor,
    hard_negatives: Optional[torch.Tensor] = None
) -> torch.Tensor
```

---

## AdaptiveInference

自适应推理包装器。

### 类定义

```python
class AdaptiveInference:
    def __init__(
        self,
        model: nn.Module,
        config: qTTTConfig
    )
```

### 方法

#### generate

```python
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    use_qttt: Union[bool, str] = True,
    **kwargs
) -> torch.Tensor
```

---

## 使用示例

### 基础适配

```python
from src.qttt import qTTTConfig, QueryOnlyTTT

# 创建配置
config = qTTTConfig(max_steps=16, learning_rate=0.005)
ttt = QueryOnlyTTT(config)

# 准备输入
query = torch.randn(1, 8, 1, 64)
keys = torch.randn(1, 8, 128, 64)
values = torch.randn(1, 8, 128, 64)

# 适配
adapted_query, info = ttt.adapt(query, (keys, values))
print(f"适配步数: {info.get('steps', 'N/A')}")
print(f"最终损失: {info.get('final_loss', 'N/A')}")
```

### 使用边际损失

```python
from src.qttt import MarginMaximizationLoss

margin_loss = MarginMaximizationLoss(margin_target=2.0)
loss = margin_loss(logits, targets)
```

---

## 参考

- [源码](../../src/qttt/)
- [教程 2: qTTT 使用](../../tutorials/tutorial-02-qttt.md)
