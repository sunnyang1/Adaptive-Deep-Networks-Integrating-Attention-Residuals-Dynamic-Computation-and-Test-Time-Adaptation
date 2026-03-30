# 论文数据集测试指标汇总

本目录包含 Adaptive Deep Networks 论文中所有关键数据集的测试指标汇总。

## 文件说明

```
results/paper_metrics/
├── README.md                          # 本文件
├── paper_metrics_summary.json         # 完整指标数据 (JSON)
└── paper_metrics_report.txt           # 可读报告 (文本)
```

## 测试数据集与指标

### 1. Needle-in-Haystack (Table 4)

长上下文检索能力测试

| 上下文长度 | Transformer | TTT-Linear | AttnRes | ADB+TurboQuant |
|-----------|-------------|------------|---------|----------------|
| 4K        | 87.5%       | 94.2%      | 96.8%   | **98.5%**      |
| 32K       | 22.1%       | 65.3%      | 75.6%   | **91.3%**      |
| 64K       | 8.7%        | 48.7%      | 58.9%   | **85.5%**      |
| 128K      | 3.2%        | 32.1%      | 42.3%   | **78.2%**      |
| 256K      | 1.5%        | 18.5%      | 28.7%   | **68.2%**      |
| **平均**   | **38.2%**   | **62.3%**  | **69.9%** | **86.9%**   |

**关键发现**:
- 256K 上下文: 68.2% vs 1.5% 基线 (**45× 提升**)
- 优势随长度增加: +11.1% (4K) → +53.6% (256K)

### 2. MATH Dataset (Table 6)

数学推理能力测试 (8.7B 模型)

| 方法                    | Level 1-2 | Level 3-4 | Level 5 | Overall |
|------------------------|-----------|-----------|---------|---------|
| Transformer            | 60.4%     | 31.6%     | 12.1%   | 35.2%   |
| CoT (5 samples)        | 65.5%     | 38.7%     | 18.5%   | 41.5%   |
| TTT-Linear             | 70.0%     | 46.8%     | 28.7%   | 48.9%   |
| **AttnRes + qTTT**     | **71.5%** | **51.3%** | **34.5%** | **52.3%** |
| AttnRes + qTTT (max)   | 74.9%     | 58.6%     | 42.1%   | 58.9%   |

**关键发现**:
- 8.7B 参数达到 **52.3%** 整体准确率
- 匹配 **50B** 静态基线性能
- Level 5 (最难): 34.5% vs 12.1% (+22.4%)

### 3. 消融研究 (Table 7)

组件贡献分析 (8.7B, LongBench-v2)

| 配置                  | 平均分数 | Δ vs Full |
|----------------------|---------|-----------|
| **完整系统**          | **56.8%** | —         |
| w/o qTTT             | 50.1%   | -6.7%     |
| w/o Gating           | 53.2%   | -3.6%     |
| w/o AttnRes          | 48.9%   | -7.9%     |
| w/o TurboQuant       | 51.5%   | -5.3%     |
| 标准 Transformer      | 39.7%   | -17.1%    |

**协同系数**: 1.18 (超加性交互)

**组件重要性** (按移除影响排序):
1. AttnRes: -7.9%
2. qTTT: -6.7%
3. TurboQuant: -5.3%
4. Gating: -3.6%

### 4. 计算效率 (Table 8)

准确率-计算量 Pareto 前沿 (MATH 数据集)

| 配置                    | Avg FLOP (×10^14) | 准确率  | Acc/FLOP |
|------------------------|-------------------|--------|----------|
| Standard 32L           | 1.0               | 35.2%  | 35.2     |
| AttnRes 32L (static)   | 1.05              | 41.8%  | 39.8     |
| AttnRes + qTTT (uniform)| 1.45             | 47.5%  | 32.8     |
| **AttnRes + qTTT (gated)** | **1.28**     | **52.3%** | **40.9** |
| AttnRes + qTTT (oracle)| 1.15              | 54.8%  | 47.7     |

**关键发现**:
- Gated adaptation: 最佳准确率下的最低 FLOP
- Acc/FLOP: 40.9 (gated) vs 35.2 (standard)
- **40%** 计算量减少 vs FLOP-匹配替代方案

### 5. Logit Margin 分析 (Table 5)

| 上下文 | 理论最小 | Vanilla Attention | qTTT After | 改进  |
|--------|---------|-------------------|------------|------|
| 1K     | ~7.0    | 8.2               | 12.5       | +4.3 |
| 16K    | ~9.8    | 6.1               | 11.8       | +5.7 |
| 64K    | ~11.2   | 4.3               | 10.9       | +6.6 |
| 128K   | ~12.5   | 3.2               | 10.2       | +7.0 |
| 256K   | ~13.8   | 2.1               | 9.4        | +7.3 |

**关键发现**:
- Vanilla margins 随长度衰减 (8.2 → 2.1)
- qTTT 保持稳定 margins (12.5 → 9.4)
- 显式 margin 最大化通过梯度优化

## 顶级成就汇总

| 指标 | 数值 | 对比基线 |
|------|------|----------|
| Needle-in-Haystack 平均 | **86.9%** | 38.2% (2.3×) |
| MATH 整体 | **52.3%** | 35.2% (8.7B vs 50B) |
| 推理吞吐量 | **110 tok/s** | 45 tok/s (2.4×) |
| KV Cache 减少 | **5.7×** | 16GB → 2.8GB |
| 计算效率 | **40%** 减少 | vs FLOP-匹配方案 |

## 模型规格

| 模型 | 参数量 | 层数 | 隐藏维度 | 头数 | AttnRes 块数 |
|------|--------|------|----------|------|--------------|
| AttnRes-S | 2.2B | 32 | 2048 | 32 | 8 |
| AttnRes-M | 8.7B | 32 | 4096 | 32 | 8 |
| AttnRes-L | 27B | 64 | 5120 | 40 | 16 |

## TurboQuant 压缩指标

| 指标 | 数值 |
|------|------|
| 内存减少 | **6×+** (零准确率损失) |
| KV Cache 减少 | **5.7×** (16GB → 2.8GB) |
| 吞吐量提升 | **8×** (Tensor Core INT4) |
| 深度缩放成本 | **8×** 减少 |

## 如何使用这些指标

### 评估新模型
```python
# 对比论文指标评估你的模型
from scripts.paper_metrics_summary import needle_haystack_metrics

paper_results = needle_haystack_metrics()
# 运行你的模型测试
your_results = test_your_model()
# 对比分析
compare_results(paper_results, your_results)
```

### 复现论文结果
1. 使用提供的配置文件 (`configs/`)
2. 在相同硬件环境 (NVIDIA H100)
3. 使用相同数据集和评估协议
4. 对比上述指标验证复现

### 引用论文
```bibtex
@article{adaptive_deep_networks_2026,
  title={Adaptive Deep Networks: Integrating Attention 
         Residuals, TurboQuant Compression, and 
         Test-Time Adaptation},
  author={[Authors]},
  journal={arXiv preprint},
  year={2026}
}
```

## 相关脚本

```
scripts/
├── paper_metrics_summary.py       # 生成本报告
├── test_small_model_datasets.py   # 实际测试框架 (需预训练权重)
└── run_small_model_experiments_fast.py  # Small Model 实验
```

## 注意事项

1. **预训练权重**: 实际评估需要预训练模型权重
2. **硬件要求**: 论文使用 NVIDIA H100 80GB
3. **完整评估**: 需要运行完整训练流程
4. **数据集**: MATH 数据集需从 HuggingFace 下载

---

*Generated: 2026-03-30*  
*Based on: Adaptive Deep Networks TurboQuant Paper*
