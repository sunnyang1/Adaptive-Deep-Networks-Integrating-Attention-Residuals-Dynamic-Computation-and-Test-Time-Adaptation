#!/bin/bash
# Git 提交脚本
# 分批提交更改，避免一次性提交过多文件

set -e

echo "======================================"
echo "Git 提交助手"
echo "======================================"
echo ""

# 检查是否在 git 仓库
if [ ! -d .git ]; then
    echo "错误: 当前目录不是 git 仓库"
    exit 1
fi

echo "当前分支: $(git branch --show-current)"
echo "远程仓库: $(git remote get-url origin)"
echo ""

# 显示变更摘要
echo "变更摘要:"
echo "  修改的文件: $(git diff --name-only | wc -l)"
echo "  新增的文件: $(git ls-files --others --exclude-standard | wc -l)"
echo ""

# 第一步：提交修改的文件
echo "【步骤 1/4】提交修改的文件..."
git add AGENTS.md README.md tasks/product-brief.md tasks/prd-adaptive-deep-networks-validation.md
git add Adaptive_Deep_Networks_Final.md
git add src/attnres/pseudo_query.py src/models/adaptive_transformer.py

git commit -m "docs: update model parameters to actual calculated values (2.2B/8.7B/27B)

- Update README.md model config table
- Update AGENTS.md with correct parameter counts
- Update paper document with 2.2B/8.7B/27B
- Fix model weight references in all docs"

echo "✓ 步骤 1 完成"
echo ""

# 第二步：提交新增的计算和报告脚本
echo "【步骤 2/4】提交计算和验证脚本..."
git add calculate_params.py calculate_params_v2.py calculate_training_time.py
git add MODEL_PARAMS_REPORT.md PARAMS_UPDATE_SUMMARY.md PAPER_PARAMS_UPDATE.md

git commit -m "feat: add parameter calculation and validation tools

- Add accurate parameter count calculation scripts
- Add training time estimator for H20
- Add model parameter reports and update summaries"

echo "✓ 步骤 2 完成"
echo ""

# 第三步：提交 H20 训练脚本
echo "【步骤 3/4】提交 H20 训练配置..."
git add scripts/autodl_h20_setup.sh scripts/ds_config_h20.json
.git add scripts/train_h20.py scripts/quick_start_h20.sh
git add H20_4CARD_SUMMARY.md

git commit -m "feat: add H20-NVLink 4-card training support

- Add H20-specific setup script with CUDA 12.1 support
- Add DeepSpeed config for 4x H20 96GB
- Add training scripts optimized for H20
- Add 4-card configuration summary"

echo "✓ 步骤 3 完成"
echo ""

# 第四步：提交流式加载功能
echo "【步骤 4/4】提交流式加载功能..."
git add scripts/train_streaming.py
git add STREAMING_TRAINING_GUIDE.md STREAMING_UPDATE_SUMMARY.md

git commit -m "feat: add streaming dataset loading for limited disk space

- Add train_streaming.py for zero-local-storage training
- Support FineWeb, SlimPajama, OpenWebText streaming
- Support distributed training with streaming
- Add comprehensive streaming training guide"

echo "✓ 步骤 4 完成"
echo ""

# 推送
echo "======================================"
echo "所有更改已提交到本地仓库"
echo "======================================"
echo ""
echo "是否要推送到 GitHub? (y/n)"
read -r response
if [[ $response =~ ^[Yy]$ ]]; then
    echo "推送到 origin/main..."
    git push origin main
    echo ""
    echo "✓ 推送完成!"
else
    echo "已跳过推送"
    echo "稍后手动推送: git push origin main"
fi

echo ""
echo "提交历史:"
git log --oneline -5
