---
title: "训练你的第一个模型"
description: "从零开始训练你的第一个ADN模型"
category: "getting-started"
difficulty: "beginner"
time: "30分钟"
prerequisites: ["完成安装", "了解PyTorch基础"]
last_updated: "2026-04-24"
---

# 训练你的第一个模型

本指南将引导你完成第一个 ADN 模型的完整训练流程。

---

## 🎯 目标

完成本指南后，你将：
- 理解 ADN 的训练流程
- 成功训练一个小模型
- 学会监控训练过程
- 能够评估模型性能

---

## 📋 准备工作

### 硬件要求

| 配置 | 显存需求 | 训练时间 |
|------|----------|----------|
| T4 配置 | 16GB | ~2小时 |
| Small 配置 | 24GB | ~4小时 |
| 使用 CPU | 32GB RAM | ~10小时 |

### 数据准备

ADN 支持多种数据集：

```bash
# 自动下载 (首次运行)
python3 scripts/data/download_datasets.sh

# 或使用 HuggingFace 数据集
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像
```

---

## 🚀 快速训练

### 方式一: 使用 Makefile (推荐)

```bash
# 训练小模型
make train-paper-small OUTPUT_DIR=results/my_first_model

# 或训练 T4 配置 (适合小显存)
make train-paper-t4 OUTPUT_DIR=results/my_first_model
```

### 方式二: 使用训练脚本

```bash
# 基础训练命令
python3 scripts/training/train_model.py \
    --model-size small \
    --output-dir results/my_first_model \
    --num-steps 1000 \
    --batch-size 4
```

### 方式三: Python API

```python
# train_first_model.py
from src.models.configs import AttnResSmallConfig, TrainingConfig
from src.models.adaptive_transformer import AdaptiveTransformer
import torch

# 1. 配置
model_config = AttnResSmallConfig()
training_config = TrainingConfig(
    batch_size_tokens=100000,  # 小批次适合快速测试
    learning_rate=3e-4,
    num_steps=1000
)

print(f"🚀 Training {model_config.num_layers}L/{model_config.hidden_dim}H model")

# 2. 创建模型
model = AdaptiveTransformer(model_config)
print(f"✅ Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# 3. 训练 (简化示例)
optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)

for step in range(training_config.num_steps):
    # 这里应该是真实的数据加载和前向传播
    # 简化示例仅展示结构

    optimizer.step()
    optimizer.zero_grad()

    if step % 100 == 0:
        print(f"Step {step}/{training_config.num_steps}")

print("🎉 Training complete!")
```

---

## 📊 监控训练

### 命令行监控

训练脚本会自动输出：
```
Step 100/1000 | Loss: 2.345 | LR: 2.8e-4 | Time: 1.2s/step
Step 200/1000 | Loss: 2.123 | LR: 2.5e-4 | Time: 1.1s/step
...
```

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir results/my_first_model/logs

# 访问 http://localhost:6006
```

### Weights & Biases

```bash
# 登录 wandb
wandb login

# 训练时自动记录
python3 scripts/training/train_model.py \
    --model-size small \
    --output-dir results/my_first_model \
    --use-wandb
```

---

## 🔧 配置详解

### 模型配置

```python
from src.models.configs import AttnResSmallConfig, AttnResT4Config

# 小模型 (1.1B 参数)
small_config = AttnResSmallConfig()

# T4 配置 (~125M 参数，适合小显存)
t4_config = AttnResT4Config()

# 自定义配置
from src.models.configs import ModelConfig

custom_config = ModelConfig(
    num_layers=16,
    hidden_dim=1024,
    num_heads=8,
    num_blocks=4,
    max_seq_len=8192
)
```

### 训练配置

```python
from src.models.configs import TrainingConfig

training_config = TrainingConfig(
    # 批次设置
    batch_size_tokens=4_000_000,  # 4M tokens

    # 优化器设置
    learning_rate=3e-4,
    lr_schedule='cosine',
    warmup_steps=2000,
    weight_decay=0.1,

    # 训练设置
    gradient_clipping=1.0,
    use_gradient_checkpointing=True,
    mixed_precision='bf16',

    # 日志设置
    log_every=100,
    eval_every=1000,
    save_every=5000
)
```

---

## 🎛️ 高级训练选项

### 分布式训练

```bash
# 使用 DeepSpeed
python3 scripts/training/train_h20.py \
    --model-size medium \
    --output-dir results/distributed_model \
    --deepspeed configs/ds_config_h20.json

# 或使用 torchrun
torchrun --nproc_per_node=4 scripts/training/train_model.py \
    --model-size medium \
    --output-dir results/distributed_model
```

### 流式训练

```bash
# 大数据集流式训练
python3 scripts/training/train_streaming.py \
    --model-size small \
    --dataset-path /path/to/large/dataset \
    --output-dir results/streaming_model
```

### 从检查点恢复

```bash
python3 scripts/training/train_model.py \
    --model-size small \
    --output-dir results/my_first_model \
    --resume-from results/my_first_model/checkpoint_5000.pt
```

---

## 📈 评估模型

### 训练中评估

训练脚本会自动在验证集上评估：
```
Eval @ Step 1000 | Loss: 2.012 | Perplexity: 7.48
```

### 独立评估

```bash
# 运行验证实验
python3 scripts/evaluation/validate_models.py \
    --model-path results/my_first_model/checkpoint_final.pt \
    --output-dir results/evaluation

# Needle-in-Haystack 测试
python3 src/benchmarks/needle_haystack.py \
    --model-path results/my_first_model/checkpoint_final.pt \
    --context-lengths 1024 4096 16384
```

### Python API 评估

```python
# evaluate_model.py
import torch
from src.models.adaptive_transformer import AdaptiveTransformer
from src.models.configs import AttnResSmallConfig

# 加载模型
config = AttnResSmallConfig()
model = AdaptiveTransformer(config)

checkpoint = torch.load('results/my_first_model/checkpoint_final.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# 评估
model.eval()
with torch.no_grad():
    # 你的评估代码
    pass
```

---

## 💾 保存和加载

### 检查点结构

```
results/my_first_model/
├── checkpoint_1000.pt      # 中间检查点
├── checkpoint_5000.pt      # 中间检查点
├── checkpoint_final.pt     # 最终模型
├── config.json             # 模型配置
├── training_state.json     # 训练状态
└── logs/                   # 训练日志
    ├── events.out.tfevents...  # TensorBoard
    └── wandb/              # W&B 日志
```

### 导出模型

```python
# 导出为 HuggingFace 格式
python3 scripts/evaluation/export_model.py \
    --checkpoint results/my_first_model/checkpoint_final.pt \
    --output-dir exported_model/
```

---

## 🐛 故障排除

### 显存不足 (OOM)

```bash
# 解决方案 1: 使用梯度检查点
python3 scripts/training/train_model.py \
    --model-size small \
    --use-gradient-checkpointing

# 解决方案 2: 减小批次大小
python3 scripts/training/train_model.py \
    --model-size small \
    --batch-size 2

# 解决方案 3: 使用 T4 配置
python3 scripts/training/train_model.py \
    --model-size t4
```

### 训练发散

```bash
# 降低学习率
python3 scripts/training/train_model.py \
    --model-size small \
    --learning-rate 1e-4

# 增加 warmup
python3 scripts/training/train_model.py \
    --model-size small \
    --warmup-steps 5000
```

### 训练太慢

```bash
# 使用混合精度
python3 scripts/training/train_model.py \
    --model-size small \
    --mixed-precision bf16

# 使用 Flash Attention (Linux only)
pip install flash-attn
python3 scripts/training/train_model.py \
    --model-size small \
    --use-flash-attention
```

---

## ✅ 检查清单

完成第一个模型训练后，确认：

- [ ] 训练成功完成，没有错误
- [ ] 损失曲线正常下降
- [ ] 检查点文件已保存
- [ ] 评估指标合理
- [ ] 能够从检查点恢复训练

---

## 🎓 下一步

训练完成第一个模型后：

1. **深入理解**: 阅读 [教程系列](../tutorials/) 理解各模块原理
2. **优化性能**: 查看 [性能优化指南](../explanation/performance-optimization.md)
3. **复现论文**: 按照 [复现论文结果](../how-to/reproduce-paper.md) 验证模型
4. **扩展开发**: 学习 [添加新模块](../how-to/add-new-module.md)

---

## 📚 相关文档

- [训练配置参考](../reference/config/training-configs.md)
- [模型配置参考](../reference/config/model-configs.md)
- [调试训练](../how-to/debug-training.md)
- [内存分析](../how-to/profile-memory.md)

---

*训练遇到问题？查看 [故障排除](troubleshooting.md) 或 [调试训练](../how-to/debug-training.md)。*
