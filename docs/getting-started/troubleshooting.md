---
title: "常见问题与故障排除"
description: "ADN 常见问题解答和故障排除指南"
category: "getting-started"
difficulty: "beginner"
last_updated: "2026-04-24"
---

# 常见问题与故障排除

本文档汇总了 ADN 使用过程中的常见问题和解决方案。

---

## 🔍 问题分类

| 类别 | 描述 | 快速跳转 |
|------|------|----------|
| **安装问题** | 环境配置、依赖安装 | [#安装问题](#安装问题) |
| **训练问题** | 训练过程中的错误 | [#训练问题](#训练问题) |
| **性能问题** | 速度慢、显存不足 | [#性能问题](#性能问题) |
| **测试问题** | 测试失败 | [#测试问题](#测试问题) |
| **其他问题** | 其他常见问题 | [#其他问题](#其他问题) |

---

## 安装问题

### Q1: Python 版本不匹配

**症状**:
```
ERROR: requires Python >= 3.12
```

**解决方案**:
```bash
# 检查 Python 版本
python3 --version

# 安装 Python 3.12
# Ubuntu/Debian
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# macOS
brew install python@3.12

# 创建虚拟环境
python3.12 -m venv venv
source venv/bin/activate
```

---

### Q2: PyTorch 安装失败

**症状**:
```
ERROR: Could not find a version that satisfies the requirement torch
```

**解决方案**:
```bash
# CPU 版本
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121
```

---

### Q3: 导入错误 `No module named 'src'`

**症状**:
```python
ModuleNotFoundError: No module named 'src'
```

**解决方案**:
```python
# 在代码开头添加
import sys
sys.path.insert(0, '/path/to/Adaptive-Deep-Networks')

# 或者设置 PYTHONPATH
export PYTHONPATH=/path/to/Adaptive-Deep-Networks:$PYTHONPATH
```

---

### Q4: HuggingFace 下载失败

**症状**:
```
Connection error downloading model
```

**解决方案**:
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或在代码中设置
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

---

## 训练问题

### Q5: CUDA Out of Memory (OOM)

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:

1. **使用梯度检查点**:
```bash
python3 scripts/training/train_model.py \
    --model-size small \
    --use-gradient-checkpointing
```

2. **减小批次大小**:
```bash
python3 scripts/training/train_model.py \
    --model-size small \
    --batch-size 2
```

3. **使用更小的模型配置**:
```bash
python3 scripts/training/train_model.py \
    --model-size t4  # T4 配置更小
```

4. **减少序列长度**:
```python
config.max_seq_len = 4096  # 默认可能是 32768
```

---

### Q6: 训练发散 (Loss 为 NaN 或无限增大)

**症状**:
```
Step 100 | Loss: nan
Step 200 | Loss: inf
```

**解决方案**:

1. **降低学习率**:
```bash
python3 scripts/training/train_model.py \
    --learning-rate 1e-4  # 默认是 3e-4
```

2. **增加 warmup 步数**:
```bash
python3 scripts/training/train_model.py \
    --warmup-steps 5000  # 默认是 2000
```

3. **检查数据**:
```python
# 确保输入数据没有 nan 或 inf
assert not torch.isnan(input).any()
assert not torch.isinf(input).any()
```

4. **使用梯度裁剪**:
```bash
python3 scripts/training/train_model.py \
    --gradient-clipping 0.5  # 默认是 1.0
```

---

### Q7: 训练卡住/无响应

**症状**:
训练进程卡住，没有输出

**解决方案**:

1. **检查数据加载**:
```python
# 确保数据加载器工作正常
for batch in dataloader:
    print(batch.shape)
    break
```

2. **减少工作进程**:
```bash
python3 scripts/training/train_model.py \
    --num-workers 0  # 禁用多进程数据加载
```

3. **检查 GPU 状态**:
```bash
nvidia-smi
# 查看是否有僵尸进程
```

---

### Q8: 检查点加载失败

**症状**:
```
RuntimeError: Error loading checkpoint
```

**解决方案**:

1. **检查检查点文件**:
```bash
ls -lh results/my_model/checkpoint_*.pt
```

2. **使用兼容的配置**:
```python
# 确保配置与训练时一致
config = AttnResSmallConfig()  # 或其他配置
```

3. **部分加载**:
```python
checkpoint = torch.load('checkpoint.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

---

## 性能问题

### Q9: 训练速度太慢

**症状**:
训练速度远低于预期

**解决方案**:

1. **使用混合精度**:
```bash
python3 scripts/training/train_model.py \
    --mixed-precision bf16
```

2. **启用 Flash Attention** (Linux only):
```bash
pip install flash-attn
python3 scripts/training/train_model.py \
    --use-flash-attention
```

3. **优化数据加载**:
```bash
python3 scripts/training/train_model.py \
    --num-workers 4 \
    --pin-memory
```

4. **使用更大的批次**:
```bash
python3 scripts/training/train_model.py \
    --batch-size 8  # 如果显存允许
```

---

### Q10: GPU 利用率低

**症状**:
`nvidia-smi` 显示 GPU 利用率 < 50%

**解决方案**:

1. **增加数据加载工作进程**:
```bash
python3 scripts/training/train_model.py \
    --num-workers 4
```

2. **使用预取**:
```bash
python3 scripts/training/train_model.py \
    --prefetch-factor 2
```

3. **检查 CPU 瓶颈**:
```bash
htop  # 查看 CPU 使用率
```

---

## 测试问题

### Q11: 测试失败 `tests/legacy`

**症状**:
```
FAILED tests/legacy/test_xxx.py
```

**解决方案**:
```bash
# 忽略遗留测试 (推荐)
pytest tests/ -v --tb=short --ignore=tests/legacy

# 或仅运行单元测试
pytest tests/unit/ -v --tb=short
```

**说明**: `tests/legacy/` 中的测试针对已弃用的 API，应该被忽略。

---

### Q12: 导入错误在测试中

**症状**:
```
ImportError: cannot import name 'XXX' from 'src.xxx'
```

**解决方案**:
```bash
# 确保在正确的目录运行
cd /path/to/Adaptive-Deep-Networks

# 检查 conftest.py 是否存在
ls tests/conftest.py

# 重新安装
pip install -e ".[dev]"
```

---

## 其他问题

### Q13: 随机种子不固定

**症状**:
每次运行结果不同

**解决方案**:
```python
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

### Q14: 日志太多/太少

**症状**:
日志输出不符合预期

**解决方案**:
```bash
# 增加日志详细度
python3 scripts/training/train_model.py \
    --log-level DEBUG

# 减少日志
python3 scripts/training/train_model.py \
    --log-level WARNING
```

---

### Q15: 无法保存大模型

**症状**:
```
RuntimeError: Unable to save checkpoint
```

**解决方案**:

1. **检查磁盘空间**:
```bash
df -h
```

2. **使用分片保存**:
```python
torch.save(
    checkpoint,
    'checkpoint.pt',
    _use_new_zipfile_serialization=True
)
```

3. **仅保存模型权重**:
```python
torch.save(model.state_dict(), 'model_weights.pt')
```

---

## 🛠️ 诊断工具

### 环境检查脚本

```bash
# 运行环境诊断
python3 scripts/setup/check_env.py
```

### 内存分析

```bash
# 分析内存使用
python3 -m memory_profiler scripts/training/train_model.py
```

### 性能分析

```bash
# PyTorch 分析器
python3 scripts/training/train_model.py \
    --profile \
    --profile-steps 10
```

---

## 📞 获取帮助

如果以上方案无法解决你的问题：

1. **查看日志**: 收集完整的错误日志
2. **搜索 Issues**: 在 [GitHub Issues](https://github.com/your-org/Adaptive-Deep-Networks/issues) 搜索
3. **提交 Issue**: 创建新 Issue，包含：
   - 问题描述
   - 复现步骤
   - 环境信息 (OS, Python, PyTorch 版本)
   - 完整错误日志
   - 已尝试的解决方案

---

## 📚 相关文档

- [安装指南](installation.md)
- [调试训练](../how-to/debug-training.md)
- [内存分析](../how-to/profile-memory.md)
- [AGENTS.md](../../AGENTS.md) - 已知问题列表

---

*最后更新: 2026-04-24*
