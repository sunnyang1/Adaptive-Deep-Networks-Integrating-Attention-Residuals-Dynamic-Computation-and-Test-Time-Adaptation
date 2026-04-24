---
title: "5分钟快速开始"
description: "5分钟内运行你的第一个ADN示例"
category: "getting-started"
difficulty: "beginner"
time: "5分钟"
prerequisites: ["完成安装"]
last_updated: "2026-04-24"
---

# 5分钟快速开始

本指南将帮助你在5分钟内运行第一个ADN示例。

---

## 🚀 一键启动

如果你使用 A100 80G，最简单的方式：

```bash
bash scripts/setup/QUICKSTART.sh
```

你会看到：
```
==========================================
Quick Start Complete!
==========================================

You can now:
1. Run validation experiments: make validate
2. Train a small model: make train-paper-small OUTPUT_DIR=results/small
3. Run quick experiments: make quick
```

---

## 📝 手动快速开始

### 步骤 1: 验证环境

```bash
# 检查 Python
python3 --version
# Python 3.12.x

# 检查 PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 检查 ADN
python3 -c "import src; print('ADN ready!')"
```

### 步骤 2: 运行快速实验

```bash
# 运行快速实验 (约 2-3 分钟)
make quick
```

这会运行：
- 表示埋葬实验
- 梯度流实验
- FLOP 等效性验证

### 步骤 3: 查看结果

```bash
# 结果保存在
ls results/

# 查看实验报告
cat results/quick_experiments_summary.md
```

---

## 🎯 你的第一个代码

创建一个简单的测试脚本：

```python
# first_adn.py
import torch
from src.models.configs import AttnResSmallConfig
from src.attnres import BlockAttnRes

print("🚀 ADN Quick Start Demo")
print("=" * 50)

# 1. 创建配置
config = AttnResSmallConfig()
print(f"✅ Config created: {config.num_layers} layers, {config.hidden_dim} hidden dim")

# 2. 创建 AttnRes 模块
attn_res = BlockAttnRes(
    hidden_dim=config.hidden_dim,
    num_heads=config.num_heads,
    num_blocks=config.num_blocks,
    num_layers=config.num_layers
)
print(f"✅ AttnRes module created with {config.num_blocks} blocks")

# 3. 创建模拟输入
batch_size = 2
seq_len = 128
hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
print(f"✅ Input shape: {hidden_states.shape}")

# 4. 前向传播
with torch.no_grad():
    output = attn_res(hidden_states, layer_idx=0)
print(f"✅ Output shape: {output.shape}")

print("=" * 50)
print("🎉 Success! ADN is working correctly.")
```

运行：
```bash
python3 first_adn.py
```

---

## 📊 运行基准测试

### Needle-in-Haystack 测试

```bash
# 快速测试 (短序列)
python3 src/benchmarks/needle_haystack.py \
    --context-lengths 1024 4096 \
    --num-trials 3
```

### MATH 评估

```bash
# 快速评估 (少量样本)
python3 src/benchmarks/math_eval.py \
    --max-samples 10 \
    --difficulty 1 2
```

---

## 🔍 验证安装

运行完整的验证套件：

```bash
# 单元测试 (约 1 分钟)
pytest tests/unit/ -v --tb=short

# 集成测试 (约 2 分钟)
pytest tests/integration/ -v --tb=short
```

---

## 🎓 下一步

完成快速开始后，你可以选择：

### 学习路径

| 路径 | 适合 | 下一步 |
|------|------|--------|
| **研究者** | 想理解原理 | [教程 1: AttnRes](../tutorials/tutorial-01-attnres.md) |
| **工程师** | 想训练模型 | [训练第一个模型](first-model.md) |
| **开发者** | 想扩展代码 | [添加新模块](../how-to/add-new-module.md) |

### 推荐顺序

1. **理解概念**: 阅读 [技术文档](../TECHNICAL_DOCUMENTATION.md) 的项目概览
2. **动手实践**: 完成 [教程系列](../tutorials/)
3. **解决实际问题**: 查看 [操作指南](../how-to/)

---

## 💡 常见问题

### Q: 运行很慢？
A: 首次运行会下载模型和数据集，后续会快很多。

### Q: 出现 CUDA 错误？
A: 检查 GPU 可用性：
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Q: 测试失败？
A: 确保使用 `--ignore=tests/legacy`：
```bash
pytest tests/ -v --tb=short --ignore=tests/legacy
```

---

## ✅ 检查清单

确认你已完成：

- [ ] 环境验证通过
- [ ] 快速实验运行成功
- [ ] 第一个代码脚本运行成功
- [ ] 单元测试通过

---

## 🎉 恭喜！

你已经成功运行了第一个 ADN 示例！

现在可以：
- 🎓 [学习教程](../tutorials/)
- 🏋️ [训练模型](first-model.md)
- 📚 [阅读文档](../)

---

*遇到问题？查看 [故障排除](troubleshooting.md) 或 [安装指南](installation.md)。*
