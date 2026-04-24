---
title: "教程 6: 端到端训练"
description: "整合所有模块，搭建完整的 ADN 训练和推理流程"
category: "tutorials"
difficulty: "intermediate"
duration: "60分钟"
prerequisites: ["完成教程 1-5", "深度学习训练基础"]
last_updated: "2026-04-24"
---

# 教程 6: 端到端训练

本教程将整合前面学到的所有模块，带你完成一个完整的 ADN 模型训练和推理流程。

---

## 你将学习

- [ ] 整合所有模块的训练流程
- [ ] 配置完整的 ADN 模型
- [ ] 训练循环的实现
- [ ] 监控和调试技巧

---

## 1. 完整模型配置

### 1.1 使用预定义配置

```python
from src.models import AdaptiveTransformer, AttnResSmallConfig

# 使用预定义的小模型配置
config = AttnResSmallConfig()

# 创建模型
model = AdaptiveTransformer(config)

print(f"模型配置:")
print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"  层数: {config.num_layers}")
print(f"  隐藏维度: {config.hidden_dim}")
print(f"  AttnRes 块数: {config.num_blocks}")
```

### 1.2 自定义配置

```python
from src.models import ModelConfig
from src.engram import EngramConfig

# 创建自定义配置
config = ModelConfig(
    # 基础架构
    num_layers=24,
    hidden_dim=2048,
    num_heads=16,

    # AttnRes
    num_blocks=8,

    # qTTT
    max_qttt_steps=24,
    qttt_span_length=128,
    qttt_learning_rate=0.005,

    # Gating
    gating_target_rate=0.3,

    # Engram (可选)
    use_engram=True,
    engram_config=EngramConfig(
        ngram_sizes=[2, 3, 4],
        embedding_dim=2048,
        memory_size=50000
    )
)

model = AdaptiveTransformer(config)
```

---

## 2. 数据准备

### 2.1 使用 HuggingFace 数据集

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# 加载数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 预处理
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)
```

### 2.2 创建 DataLoader

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_dataset["train"],
    batch_size=4,
    shuffle=True,
    collate_fn=lambda x: {
        'input_ids': torch.stack([torch.tensor(d['input_ids']) for d in x]),
        'attention_mask': torch.stack([torch.tensor(d['attention_mask']) for d in x])
    }
)
```

---

## 3. 训练循环

### 3.1 基础训练循环

```python
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        # 准备输入
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # 前向传播
        optimizer.zero_grad()

        with autocast():  # 混合精度
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = outputs.loss

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)
```

### 3.2 完整训练脚本

```python
#!/usr/bin/env python3
"""
端到端训练示例
运行: python tutorial_06_training.py
"""

import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup

from src.models import AdaptiveTransformer, AttnResSmallConfig

def setup_training(model, learning_rate=3e-4):
    """设置训练组件"""
    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    # 学习率调度
    num_training_steps = 10000
    num_warmup_steps = 2000
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 混合精度
    scaler = GradScaler()

    return optimizer, scheduler, scaler

def train_step(model, batch, optimizer, scaler, device):
    """单步训练"""
    model.train()

    input_ids = batch['input_ids'].to(device)
    labels = batch.get('labels', input_ids)

    optimizer.zero_grad()

    with autocast():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item()

def validate(model, dataloader, device):
    """验证"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch.get('labels', input_ids)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs

            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    config = AttnResSmallConfig()
    model = AdaptiveTransformer(config).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # 设置训练
    optimizer, scheduler, scaler = setup_training(model)

    # 模拟训练 (实际使用时替换为真实数据)
    print("\n开始训练...")
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}/3")

        # 模拟一个 batch
        batch = {
            'input_ids': torch.randint(0, 32000, (2, 128)).to(device),
            'labels': torch.randint(0, 32000, (2, 128)).to(device)
        }

        loss = train_step(model, batch, optimizer, scaler, device)
        scheduler.step()

        print(f"  Loss: {loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    print("\n训练完成!")

if __name__ == "__main__":
    main()
```

---

## 4. 推理与生成

### 4.1 基础生成

```python
from src.models import IncrementalGenerator

# 创建生成器
generator = IncrementalGenerator(model)

# 生成文本
prompt = "The future of artificial intelligence"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# 生成
output = generator.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.95
)

generated_text = tokenizer.decode(output[0])
print(generated_text)
```

### 4.2 启用 qTTT 的生成

```python
# 启用自适应 qTTT
output = generator.generate(
    input_ids,
    max_new_tokens=50,
    use_qttt='adaptive',  # 启用自适应 qTTT
    qttt_config={
        'max_steps': 16,
        'learning_rate': 0.005
    }
)
```

---

## 5. 监控和调试

### 5.1 训练监控

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/adn_experiment')

# 在训练循环中
writer.add_scalar('Loss/train', loss, global_step)
writer.add_scalar('LearningRate', lr, global_step)

# 监控各模块
writer.add_scalar('AttnRes/BlockUsage', avg_block_usage, global_step)
writer.add_scalar('Gating/AdaptationRate', adaptation_rate, global_step)
writer.add_scalar('qTTT/AvgSteps', avg_qttt_steps, global_step)
```

### 5.2 调试技巧

```python
# 1. 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm().item()}")

# 2. 检查激活值
outputs = model(input_ids, output_attentions=True)
print(f"Attention weights: {outputs.attentions[0].shape}")

# 3. 内存监控
import torch.cuda as cuda
print(f"GPU memory: {cuda.memory_allocated() / 1024**3:.2f} GB")
```

---

## 6. 完整示例：训练脚本

```python
#!/usr/bin/env python3
"""
完整的端到端训练脚本
"""

import argparse
import torch
from pathlib import Path

from src.models import (
    AdaptiveTransformer,
    AttnResSmallConfig,
    AttnResMediumConfig,
    AttnResLargeConfig
)
from src.training import Trainer, TrainingConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-size', type=str, default='small',
                       choices=['small', 'medium', 'large'])
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    return parser.parse_args()

def get_model_config(size):
    configs = {
        'small': AttnResSmallConfig,
        'medium': AttnResMediumConfig,
        'large': AttnResLargeConfig
    }
    return configs[size]()

def main():
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_model_config(args.model_size)

    # 创建模型
    print(f"创建 {args.model_size} 模型...")
    model = AdaptiveTransformer(config)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"参数量: {param_count / 1e6:.1f}M")

    # 训练配置
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        output_dir=str(output_dir)
    )

    # 创建训练器
    trainer = Trainer(model, train_config)

    # 准备数据 (简化版)
    print("准备数据...")
    # 实际使用时加载真实数据集

    # 训练
    print("开始训练...")
    trainer.train()

    # 保存模型
    checkpoint_path = output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'args': vars(args)
    }, checkpoint_path)

    print(f"模型已保存到: {checkpoint_path}")

if __name__ == '__main__':
    main()
```

---

## 7. 练习

### 练习 1: 实现自定义回调

```python
class CustomCallback:
    def on_train_begin(self):
        pass

    def on_epoch_end(self, epoch, metrics):
        print(f"Epoch {epoch}: {metrics}")

    def on_train_end(self):
        pass
```

### 练习 2: 分布式训练

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# 初始化
dist.init_process_group(backend='nccl')

# 包装模型
model = DistributedDataParallel(model)
```

---

## 8. 常见问题

**Q: 训练时显存不足怎么办？**
A: 1) 减小 batch_size；2) 启用梯度检查点；3) 使用更小的模型配置。

**Q: 如何恢复训练？**
A: 保存 optimizer 和 scheduler 状态，加载时一并恢复。

**Q: 伪查询的学习率需要调整吗？**
A: 通常使用与其他参数相同的学习率即可。

---

## 9. 恭喜完成！

完成所有教程后，你应该能够：
- [x] 配置完整的 ADN 模型
- [x] 实现训练循环
- [x] 进行推理和生成
- [x] 监控和调试训练过程

### 下一步

- 查看 [操作指南](../how-to/) 解决具体问题
- 阅读 [API 参考](../reference/api/) 深入了解接口
- 探索 [解释文档](../explanation/) 理解设计原理

---

## 参考

- [训练脚本](../../scripts/training/)
- [API 文档](../reference/api/models.md)
- [论文 §6](../papers/adn-paper.md#6-experiments)
