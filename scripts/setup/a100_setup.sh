#!/bin/bash
# =============================================================================
# A100 80G 新手一键设置脚本
# 使用方法: bash scripts/setup/a100_setup.sh
# =============================================================================

set -e  # 遇到错误立即退出

echo "============================================================"
echo "  A100 80G 环境自动设置脚本"
echo "============================================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在项目目录
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}错误: 请在项目根目录运行此脚本${NC}"
    echo "请先执行: cd ~/adaptive-deep-networks"
    exit 1
fi

echo -e "${GREEN}✓${NC} 项目目录检查通过"
echo ""

# =============================================================================
# Step 1: 系统检查
# =============================================================================
echo "[1/7] 系统检查..."
echo "------------------------------------------------------------"

# 检查 Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $PYTHON_VERSION"

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} nvidia-smi 已安装"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${RED}✗${NC} nvidia-smi 未找到，请检查 NVIDIA 驱动"
    exit 1
fi

# 检查磁盘空间
DISK_AVAIL=$(df -h . | awk 'NR==2 {print $4}')
echo "可用磁盘空间: $DISK_AVAIL"
echo ""

# =============================================================================
# Step 2: 创建虚拟环境
# =============================================================================
echo "[2/7] 创建 Python 虚拟环境..."
echo "------------------------------------------------------------"

if [ -d "venv" ]; then
    echo -e "${YELLOW}!${NC} 虚拟环境已存在，跳过创建"
else
    echo "创建虚拟环境..."
    python3.10 -m venv venv 2>/dev/null || python3 -m venv venv
    echo -e "${GREEN}✓${NC} 虚拟环境创建成功"
fi

echo "激活虚拟环境..."
source venv/bin/activate
echo -e "${GREEN}✓${NC} 虚拟环境已激活"
echo ""

# =============================================================================
# Step 3: 升级 pip
# =============================================================================
echo "[3/7] 升级 pip..."
echo "------------------------------------------------------------"
pip install --upgrade pip -q
echo -e "${GREEN}✓${NC} pip 升级完成: $(pip --version)"
echo ""

# =============================================================================
# Step 4: 安装 PyTorch (CUDA 12.1)
# =============================================================================
echo "[4/7] 安装 PyTorch + CUDA..."
echo "------------------------------------------------------------"

# 检查是否已安装 PyTorch
if python -c "import torch; exit(0 if torch.__version__.startswith('2.1') else 1)" 2>/dev/null; then
    echo -e "${YELLOW}!${NC} PyTorch 2.1 已安装，跳过"
else
    echo "安装 PyTorch 2.1.0 + CUDA 12.1..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu121
    echo -e "${GREEN}✓${NC} PyTorch 安装完成"
fi

# 验证 PyTorch CUDA
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'CUDA 版本: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
echo ""

# =============================================================================
# Step 5: 安装项目依赖
# =============================================================================
echo "[5/7] 安装项目依赖..."
echo "------------------------------------------------------------"

echo "安装基础依赖（约需 2-3 分钟）..."
pip install -q \
    transformers==4.35.0 \
    datasets==2.14.0 \
    accelerate==0.24.0 \
    numpy==1.24.0 \
    scipy==1.11.0 \
    matplotlib==3.8.0 \
    seaborn==0.13.0 \
    pandas==2.0.0 \
    tqdm==4.66.0 \
    wandb==0.15.0 \
    pytest==7.4.0

echo -e "${GREEN}✓${NC} 基础依赖安装完成"

# 可选：安装 Flash Attention
echo ""
echo "安装 Flash Attention（A100 专用加速，约需 5-10 分钟）..."
echo -e "${YELLOW}!${NC} 按 Ctrl+C 跳过此步骤"
sleep 3

pip install flash-attn==2.3.0 --no-build-isolation || echo -e "${YELLOW}!${NC} Flash Attention 安装失败，继续..."

echo ""

# =============================================================================
# Step 6: 验证安装
# =============================================================================
echo "[6/7] 验证安装..."
echo "------------------------------------------------------------"

python -c "
import sys
sys.path.insert(0, 'src')

try:
    import torch
    import transformers
    from models.configs import AttnResMediumConfig
    print(f'✓ PyTorch: {torch.__version__}')
    print(f'✓ Transformers: {transformers.__version__}')
    print(f'✓ CUDA 可用: {torch.cuda.is_available()}')
    print(f'✓ 模型配置可导入')
    print('')
    print('所有核心组件验证通过！')
except Exception as e:
    print(f'✗ 验证失败: {e}')
    sys.exit(1)
"

echo ""

# =============================================================================
# Step 7: 完成提示
# =============================================================================
echo "[7/7] 设置完成！"
echo "============================================================"
echo ""
echo -e "${GREEN}✓${NC} 环境设置完成！"
echo ""
echo "下一步操作："
echo "------------------------------------------------------------"
echo "1. 激活环境（每次新开终端都需要执行）："
echo "   source venv/bin/activate"
echo ""
echo "2. 开始训练 Medium 模型（5.7B 参数，推荐）："
echo "   tmux new-session -s training"
echo "   python scripts/training/train_medium.py \\"
echo "       --output-dir results/medium_model \\"
echo "       --epochs 3 \\"
echo "       --batch-size 4"
echo ""
echo "3. 查看详细指南："
echo "   cat docs/A100_80G_COMPLETE_GUIDE.md"
echo ""
echo "4. 运行基准测试："
echo "   python scripts/evaluation/run_benchmarks.py \\"
echo "       --model-size medium \\"
echo "       --benchmarks all"
echo ""
echo "------------------------------------------------------------"
echo "常用命令："
echo "  nvidia-smi          # 查看 GPU 状态"
echo "  tmux attach -t training  # 重新连接到训练会话"
echo "  htop                # 查看 CPU/内存使用"
echo "------------------------------------------------------------"
echo ""
echo -e "${GREEN}祝训练顺利！🎉${NC}"
echo ""
