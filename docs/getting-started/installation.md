---
title: "安装指南"
description: "ADN 完整安装指南，包括环境配置和依赖安装"
category: "getting-started"
difficulty: "beginner"
time: "15分钟"
prerequisites: ["Python 3.10+", "CUDA (可选)"]
last_updated: "2026-04-24"
---

# 安装指南

本指南将帮助你完成 ADN 的安装和配置。

---

## 📋 系统要求

### 必需

| 项目 | 版本 | 说明 |
|------|------|------|
| Python | >= 3.12 | 推荐使用 3.12 |
| PyTorch | >= 2.0.0 | 深度学习框架 |
| NumPy | >= 1.24.0 | 数值计算 |
| Git | 任意 | 代码管理 |

### 可选 (推荐)

| 项目 | 版本 | 说明 |
|------|------|------|
| CUDA | >= 11.8 | GPU 加速 |
| cuDNN | >= 8.6 | 深度神经网络库 |

### 硬件建议

| 场景 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 开发/测试 | 16GB RAM, CPU | 32GB RAM, T4 GPU |
| 小模型训练 | 32GB RAM, T4 GPU | 64GB RAM, V100 GPU |
| 中模型训练 | 64GB RAM, V100 GPU | 128GB RAM, A100 40G |
| 大模型训练 | 128GB RAM, A100 40G | 256GB RAM, A100 80G |

---

## 🚀 快速安装

### 方式一: 一键安装 (推荐 A100 用户)

```bash
# 克隆仓库
git clone https://github.com/your-org/Adaptive-Deep-Networks.git
cd Adaptive-Deep-Networks

# 运行一键安装脚本
bash scripts/setup/QUICKSTART.sh
```

脚本会自动：
- 检查系统环境
- 安装 Python 依赖
- 验证安装

### 方式二: 手动安装

#### 步骤 1: 克隆仓库

```bash
git clone https://github.com/your-org/Adaptive-Deep-Networks.git
cd Adaptive-Deep-Networks
```

#### 步骤 2: 创建虚拟环境 (推荐)

```bash
# 使用 venv
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 或使用 conda
conda create -n adn python=3.12
conda activate adn
```

#### 步骤 3: 安装依赖

```bash
# 基础安装
pip install -e ".[dev]"

# 或使用 requirements.txt
pip install -r requirements.txt
```

#### 步骤 4: 验证安装

```bash
# 运行环境检查
python3 scripts/setup/check_env.py

# 运行简单测试
python3 -c "import src; print('ADN imported successfully')"
```

---

## 🔧 详细配置

### Python 环境

#### 检查 Python 版本

```bash
python3 --version
# 应显示 Python 3.12.x 或更高
```

如果版本过低，请升级：

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# macOS
brew install python@3.12
```

### PyTorch 安装

#### CPU 版本

```bash
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
```

#### GPU 版本 (CUDA 12.1)

```bash
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121
```

#### 验证 PyTorch

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 可选依赖

#### 开发工具

```bash
pip install black ruff mypy pytest pytest-cov
```

#### 可视化

```bash
pip install matplotlib seaborn pandas
```

#### 日志追踪

```bash
pip install wandb tqdm
```

#### 性能优化 (Linux only)

```bash
pip install flash-attn>=2.3.0 triton>=2.1.0
```

---

## 🌐 环境特定配置

### A100 80G

```bash
# 使用 A100 专用设置脚本
bash scripts/setup/a100_setup.sh
```

### H20 GPU

```bash
# 使用 H20 设置脚本
bash scripts/setup/autodl_h20_setup.sh
```

### Lambda Cloud

```bash
# Lambda 云环境
bash scripts/setup/lambda_setup.sh
```

### AutoDL

```bash
# AutoDL 平台
bash scripts/setup/autodl_setup.sh
```

---

## ✅ 安装验证

### 1. 基础导入测试

```python
# test_import.py
import torch
import transformers
import src
from src.attnres import BlockAttnRes
from src.qttt import QueryOnlyTTT
from src.rabitq import create_k1
from src.models import ModelConfig

print("✅ All imports successful!")
```

运行：
```bash
python3 test_import.py
```

### 2. 运行单元测试

```bash
# 快速测试
pytest tests/unit/ -v --tb=short

# 完整测试 (排除遗留测试)
pytest tests/ -v --tb=short --ignore=tests/legacy
```

### 3. 运行示例实验

```bash
# 快速实验
make quick

# 或手动运行
python3 experiments/run_experiments_unified.py --all --quick
```

---

## 🐛 常见问题

### 问题 1: Python 版本不匹配

**错误**: `requires Python >= 3.12`

**解决**:
```bash
# 检查版本
python3 --version

# 如果低于 3.12，安装新版本
# 然后使用新版本创建虚拟环境
python3.12 -m venv venv
```

### 问题 2: CUDA 版本不匹配

**错误**: `CUDA error: no kernel image is available`

**解决**:
```bash
# 检查 CUDA 版本
nvidia-smi

# 安装匹配的 PyTorch
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 问题 3: 内存不足

**错误**: `RuntimeError: CUDA out of memory`

**解决**:
- 使用更小的模型配置 (T4 或 Small)
- 减少批次大小
- 启用梯度检查点

### 问题 4: 导入错误

**错误**: `ModuleNotFoundError: No module named 'src'`

**解决**:
```python
# 在代码开头添加
import sys
sys.path.insert(0, '/path/to/Adaptive-Deep-Networks')
```

### 问题 5: HuggingFace 下载失败

**错误**: `Connection error` 下载模型时

**解决**:
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或在代码中设置
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

---

## 📝 安装检查清单

完成安装后，确认以下检查项：

- [ ] Python 版本 >= 3.12
- [ ] PyTorch 安装成功
- [ ] ADN 可以导入
- [ ] 单元测试通过
- [ ] 示例实验可以运行
- [ ] GPU 可用 (如果使用 GPU)

---

## 🎉 下一步

安装完成！接下来：

1. [5分钟快速开始](quickstart.md) - 运行第一个示例
2. [训练第一个模型](first-model.md) - 开始训练
3. [查看教程](../tutorials/) - 深入学习

---

## 📞 获取帮助

如果安装过程中遇到问题：

1. 查看 [常见问题](troubleshooting.md)
2. 运行诊断脚本: `python3 scripts/setup/check_env.py`
3. 在 GitHub Issues 中搜索类似问题
4. 提交新的 Issue，附上：
   - 操作系统版本
   - Python 版本
   - 错误日志
   - 已尝试的解决方案

---

*安装有问题？查看 [故障排除](troubleshooting.md) 获取更多帮助。*
