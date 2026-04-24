---
title: "如何训练模型"
description: "从零开始训练 ADN 模型的完整步骤"
category: "how-to"
difficulty: "intermediate"
duration: "30分钟"
last_updated: "2026-04-24"
---

# 如何训练模型

本指南将带你完成从零开始训练 ADN 模型的完整流程。

---

## 前置条件

- [ ] 完成 [安装指南](../getting-started/installation.md)
- [ ] 至少 16GB GPU 显存 (用于 small 模型)
- [ ] 准备好训练数据

---

## 快速开始

### 1. 使用预定义配置训练

```bash
# 训练 small 模型 (1.1B 参数)
python scripts/training/train_model.py \
    --model-size small \
    --output-dir ./outputs/small \
    --epochs 3 \
    --batch-size 4

# 训练 medium 模型 (5.7B 参数)
python scripts/training/train_model.py \
    --model-size medium \
    --output-dir ./outputs/medium \
    --epochs 3 \
    --batch-size 2
```

### 2. 使用 Makefile

```bash
# 快速训练 (用于测试)
make train-small

# 完整训练
make train-medium

# 使用 DeepSpeed
make train-h20
```

---

## 详细步骤

### 步骤 1: 准备数据

#### 选项 A: 使用 HuggingFace 数据集

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 或使用自定义数据集
dataset = load_dataset("text", data_files={
    "train": "path/to/train.txt",
    "validation": "path/to/valid.txt"
})
```

#### 选项 B: 使用本地数据

```bash
# 准备数据目录
mkdir -p data/train data/valid

# 放置你的文本文件
cp your_training_data.txt data/train/
cp your_validation_data.txt data/valid/
```

### 步骤 2: 配置训练参数

创建配置文件 `configs/my_training.yaml`:

```yaml
# 模型配置
model:
  size: small  # small, medium, large

# 训练配置
training:
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 3.0e-4
  warmup_steps: 2000
  max_steps: 10000
  weight_decay: 0.1
  max_grad_norm: 1.0

# 优化器
optimizer:
  type: adamw
  beta1: 0.9
  beta2: 0.95

# 学习率调度
lr_scheduler:
  type: cosine
  min_lr_ratio: 0.1

# 日志
logging:
  log_every: 100
  eval_every: 1000
  save_every: 5000

# 混合精度
mixed_precision: bf16
```

### 步骤 3: 启动训练

```bash
python scripts/training/train_model.py \
    --config configs/my_training.yaml \
    --output-dir ./outputs/my_experiment
```

### 步骤 4: 监控训练

#### 使用 TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir ./outputs/my_experiment/logs

# 访问 http://localhost:6006
```

#### 使用 Weights & Biases

```bash
# 设置 API key
export WANDB_API_KEY=your_key

# 训练时会自动记录
python scripts/training/train_model.py \
    --use-wandb \
    --wandb-project adn-training
```

---

## 高级配置

### 多 GPU 训练

```bash
# 使用 DataParallel
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/training/train_model.py \
    --model-size small

# 或使用 DeepSpeed
deepspeed scripts/training/train_model.py \
    --deepspeed configs/ds_config.json
```

### 恢复训练

```bash
# 从检查点恢复
python scripts/training/train_model.py \
    --resume-from ./outputs/my_experiment/checkpoint-5000 \
    --output-dir ./outputs/my_experiment
```

### 微调预训练模型

```bash
# 加载预训练权重
python scripts/training/train_model.py \
    --model-size small \
    --pretrained-path ./pretrained/model.pt \
    --output-dir ./outputs/finetuned
```

---

## 故障排除

### 显存不足 (OOM)

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 1. 减小 batch size
--batch-size 2

# 2. 启用梯度累积
--gradient-accumulation-steps 8

# 3. 启用梯度检查点
--gradient-checkpointing

# 4. 使用更小的模型
--model-size t4
```

### 训练不稳定

**症状**: 损失 NaN 或剧烈波动

**解决方案**:
```yaml
# 降低学习率
learning_rate: 1.0e-4

# 增加 warmup
warmup_steps: 5000

# 使用更保守的梯度裁剪
max_grad_norm: 0.5
```

### 训练速度慢

**优化建议**:
```bash
# 1. 使用混合精度
--mixed-precision bf16

# 2. 增加 DataLoader workers
--num-workers 8

# 3. 使用 Flash Attention (如果可用)
--use-flash-attention
```

---

## 最佳实践

1. **从小模型开始**: 先用 small 模型验证配置，再扩展到更大模型
2. **频繁保存**: 设置 `--save-every 1000` 防止训练中断丢失进度
3. **监控梯度**: 关注梯度范数，确保训练稳定
4. **验证集评估**: 定期在验证集上评估，防止过拟合

---

## 参考

- [训练配置详解](../reference/config/training.md)
- [分布式训练指南](../how-to/distributed-training.md)
- [教程 6: 端到端训练](../tutorials/tutorial-06-end-to-end.md)
