# 模型权重更新摘要

## 更新日期
2026-03-25

## 更新原因
根据实际代码实现精确计算模型参数量，统一更新所有文档。

## 参数量对照表

| 模型   | 原始标注 | 实际计算 | 最终采用 |
|--------|---------|---------|---------|
| Small  | 1.5B    | 2.21B   | **2.2B** |
| Medium | 7B      | 8.72B   | **8.7B** |
| Large  | 50B     | 27.0B   | **27B**  |

## 已更新的文件

### 核心文档
- [x] `README.md` - 模型配置表、MATH 基准标注
- [x] `AGENTS.md` - 模型配置表、MATH 基准标注
- [x] `MODEL_PARAMS_REPORT.md` - 完整参数量报告

### 任务文档
- [x] `tasks/product-brief.md` - 模型规模描述
- [x] `tasks/prd-adaptive-deep-networks-validation.md` - 参数表、基准标注

### 论文文档（已正确）
- [x] `Adaptive_Deep_Networks_Final.md` - 已使用 2.2B/8.7B/27B
- [x] `LARGE_MODEL_BUILD.md` - 已正确标注 27B vs 50B 差异
- [x] `PAPER_UPDATES.md` - 已使用正确参数

## 配置详情

### Small (2.2B)
```python
num_layers=32
hidden_dim=2048
num_heads=32
mlp_ratio=4
vocab_size=32000
```

### Medium (8.7B)
```python
num_layers=32
hidden_dim=4096
num_heads=32
mlp_ratio=4
vocab_size=32000
```

### Large (27B)
```python
num_layers=64
hidden_dim=5120
num_heads=40
mlp_ratio=4
vocab_size=32000
```

## 显卡适配建议（更新版）

| 显卡 | Small (2.2B) | Medium (8.7B) | Large (27B) |
|------|-------------|--------------|-------------|
| RTX 3090/4090 24GB | ✅ 可行 | ❌ 不可行 | ❌ 不可行 |
| A100 40GB | ✅ 轻松 | ❌ 不可行 | ❌ 不可行 |
| A100 80GB | ✅ 轻松 | ⚠️ 需优化 | ❌ 不可行 |
| **A800 80GB** | ✅ 轻松 | ⚠️ 需优化 | ❌ 不可行 |
| **H20 96GB** | ✅ 非常轻松 | ✅ 可行 | ⚠️ 需多卡 |
| **RTX PRO 6000 96GB** | ✅ 非常轻松 | ✅ 可行 | ❌ 不可行 |
| 4×A100 80GB | ✅ - | ✅ - | ✅ 可行 |
| 4×H20 96GB | ✅ - | ✅ - | ✅ 轻松 |

## 验证命令

```bash
python calculate_params.py
```

输出应显示：
- Small: 2,213,414,912 参数 (2.21B)
- Medium: 8,721,797,120 参数 (8.72B)
- Large: 27,009,356,800 参数 (27.01B)
