# Adaptive_Deep_Networks_Final.md 模型权重更新

## 更新日期
2026-03-25

## 更新内容

将论文中剩余的旧标注 (50B) 更新为实际计算值 (27B)。

### 具体修改

| 行号 | 修改前 | 修改后 |
|------|-------|-------|
| 499 | `AttnRes-L \| 50B \| 64 \| ...` | `AttnRes-L \| 27B \| 64 \| ...` |
| 798 | `Large \| 50B \| 48.2% \| ...` | `Large \| 27B \| 48.2% \| ...` |
| 977 | `Large (50B)` | `Large (27B)` |
| 1119 | `8.7B, 50B variants` | `2.2B, 8.7B, 27B variants` |

## 当前论文中的模型参数统一情况

### 摘要部分
- ✅ "86.9% average accuracy on needle-in-haystack retrieval with 8.7B parameters"
- ✅ "89.4% with only 2.2B parameters"
- ✅ "52.3% on MATH with 8.7B parameters"
- ✅ "56.1% with 2.2B parameters"
- ✅ "matching 50B static baselines" (保留 - 指对比其他50B基线模型)

### 表格配置 (Table A1)
| 模型 | 参数量 | 层数 | Hidden | Heads |
|------|-------|------|--------|-------|
| AttnRes-S | **2.2B** | 32 | 2048 | 32 |
| AttnRes-M | **8.7B** | 32 | 4096 | 32 |
| AttnRes-L | **27B** | 64 | 5120 | 40 |

### 超参数表 (Table A7)
| 参数 | Small | Medium | Large |
|------|-------|--------|-------|
| 参数量 | **2.2B** | **8.7B** | **27B** |

### 模型检查点
- ✅ "HuggingFace (2.2B, 8.7B, 27B variants)"

## 未修改的保留内容

以下"50B"引用保留不变，因为它们指的是其他 baseline 模型或训练token数量：

1. **第39行**: "matching 50B static baselines" - 对比其他50B参数模型
2. **第831行**: "50B" (Training Tokens) - 训练使用的500亿token
