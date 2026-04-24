---
title: "如何调试模型"
description: "ADN 模型调试技巧和工具"
category: "how-to"
difficulty: "intermediate"
duration: "20分钟"
last_updated: "2026-04-24"
---

# 如何调试模型

本指南提供 ADN 模型调试的实用技巧和工具。

---

## 常见调试场景

### 场景 1: 检查模型输出

```python
import torch
from src.models import AdaptiveTransformer, AttnResSmallConfig

# 创建模型
config = AttnResSmallConfig()
model = AdaptiveTransformer(config)
model.eval()

# 准备输入
input_ids = torch.randint(0, 32000, (1, 128))

# 前向传播
with torch.no_grad():
    outputs = model(input_ids)

# 检查输出
print(f"输出形状: {outputs.logits.shape}")
print(f"输出范围: [{outputs.logits.min():.2f}, {outputs.logits.max():.2f}]")
print(f"输出均值: {outputs.logits.mean():.2f}")
print(f"输出标准差: {outputs.logits.std():.2f}")

# 检查是否有 NaN
if torch.isnan(outputs.logits).any():
    print("警告: 输出包含 NaN!")
```

### 场景 2: 检查梯度

```python
# 训练模式
model.train()

# 前向 + 反向
outputs = model(input_ids, labels=input_ids)
loss = outputs.loss
loss.backward()

# 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: grad_norm = {grad_norm:.4f}")

        # 检查异常梯度
        if torch.isnan(param.grad).any():
            print(f"  警告: {name} 梯度包含 NaN!")
        if grad_norm > 100:
            print(f"  警告: {name} 梯度爆炸!")
```

### 场景 3: 检查注意力权重

```python
# 获取注意力权重
outputs = model(input_ids, output_attentions=True)

# 检查每层注意力
for layer_idx, attn in enumerate(outputs.attentions):
    print(f"Layer {layer_idx}:")
    print(f"  形状: {attn.shape}")
    print(f"  注意力权重和: {attn.sum(dim=-1).mean():.4f}")  # 应该 ≈ 1.0
    print(f"  最大注意力: {attn.max():.4f}")
    print(f"  最小注意力: {attn.min():.4f}")
```

---

## 调试工具

### 使用 Python Debugger (pdb)

```python
import pdb

# 在代码中设置断点
def forward(self, x):
    pdb.set_trace()  # 执行到这里会进入调试器
    x = self.layer1(x)
    return x
```

常用 pdb 命令:
- `n` (next): 执行下一行
- `s` (step): 进入函数
- `c` (continue): 继续执行
- `p variable`: 打印变量
- `l`: 显示当前代码
- `q`: 退出调试器

### 使用 PyTorch 调试工具

```python
import torch.autograd.profiler as profiler

# 性能分析
with profiler.profile(record_shapes=True) as prof:
    outputs = model(input_ids)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 内存调试

```python
import torch.cuda as cuda

# 打印内存使用情况
def print_memory_usage():
    print(f"已分配: {cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"已缓存: {cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"最大分配: {cuda.max_memory_allocated() / 1024**3:.2f} GB")

# 在关键位置调用
print_memory_usage()
```

---

## 模块特定调试

### 调试 AttnRes

```python
from src.attnres import BlockAttnRes

# 创建模块
attn_res = BlockAttnRes(hidden_dim=512, num_heads=8, num_blocks=8)

# 测试输入
x = torch.randn(2, 128, 512)

# 前向传播
output, stats = attn_res(x)

# 检查统计信息
print("AttnRes 统计:")
for key, value in stats.items():
    print(f"  {key}: {value}")
```

### 调试 qTTT

```python
from src.qttt import QueryOnlyTTT, qTTTConfig

# 创建配置
config = qTTTConfig(max_steps=16, learning_rate=0.005)
ttt = QueryOnlyTTT(config)

# 测试适配
query = torch.randn(1, 8, 1, 64)
kv_cache = (torch.randn(1, 8, 128, 64), torch.randn(1, 8, 128, 64))

# 运行适配
adapted_query, info = ttt.adapt(query, kv_cache)

print("qTTT 信息:")
print(f"  初始损失: {info.get('initial_loss', 'N/A')}")
print(f"  最终损失: {info.get('final_loss', 'N/A')}")
print(f"  实际步数: {info.get('steps', 'N/A')}")
```

### 调试 Gating

```python
from src.gating import EMAThreshold, compute_reconstruction_loss

# 创建阈值控制器
threshold = EMAThreshold(target_rate=0.3)

# 测试不同损失值
test_losses = [0.1, 0.5, 1.0, 2.0]
for loss in test_losses:
    should_adapt = threshold.should_adapt(loss)
    print(f"损失 {loss:.2f}: {'适应' if should_adapt else '跳过'}")
```

---

## 可视化调试

### 可视化注意力热力图

```python
import matplotlib.pyplot as plt

# 获取注意力权重
outputs = model(input_ids, output_attentions=True)
attn = outputs.attentions[0][0, 0].detach().cpu()  # 第一层，第一个头

# 绘制热力图
plt.figure(figsize=(10, 8))
plt.imshow(attn, cmap='viridis')
plt.colorbar()
plt.title('Attention Heatmap (Layer 0, Head 0)')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.savefig('attention_heatmap.png')
```

### 可视化梯度流

```python
import matplotlib.pyplot as plt

# 收集梯度
gradients = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        gradients[name] = param.grad.norm().item()

# 绘制
plt.figure(figsize=(12, 6))
plt.bar(range(len(gradients)), list(gradients.values()))
plt.xticks(range(len(gradients)), list(gradients.keys()), rotation=90)
plt.ylabel('Gradient Norm')
plt.title('Gradient Flow')
plt.tight_layout()
plt.savefig('gradient_flow.png')
```

---

## 日志记录

### 配置详细日志

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 在代码中添加日志
logger.debug(f"输入形状: {input_ids.shape}")
logger.info(f"当前损失: {loss.item()}")
logger.warning(f"梯度范数过大: {grad_norm}")
```

---

## 参考

- [PyTorch 调试文档](https://pytorch.org/docs/stable/debugging.html)
- [Python pdb 文档](https://docs.python.org/3/library/pdb.html)
- [教程 6: 端到端训练](../tutorials/tutorial-06-end-to-end.md)
