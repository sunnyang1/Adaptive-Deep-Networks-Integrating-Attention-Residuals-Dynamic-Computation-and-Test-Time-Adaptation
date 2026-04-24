# Adaptive Deep Networks 文档中心

欢迎来到 ADN 文档中心！这里汇集了项目的所有技术文档，按照 Divio 文档四象限模型组织，帮助你快速找到所需信息。

---

## 🚀 我是新手，从这里开始

**完全不了解 ADN？** 从 [新手指南](guides/README_FOR_BEGINNERS.md) 开始。

**有 A100 80G 机器？** 直接阅读 [A100 完整指南](guides/A100_80G_COMPLETE_GUIDE.md)。

**想快速体验？** 执行以下命令：

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/Adaptive-Deep-Networks.git
cd Adaptive-Deep-Networks

# 2. 一键启动
bash scripts/setup/QUICKSTART.sh

# 3. 运行第一个实验
make quick
```

---

## 📚 文档导航

### 🎓 入门系列 (Getting Started)

适合第一次接触 ADN 的开发者：

| 文档 | 内容 | 预计时间 |
|------|------|----------|
| [安装指南](getting-started/installation.md) | 环境配置、依赖安装 | 15分钟 |
| [5分钟快速开始](getting-started/quickstart.md) | 运行第一个示例 | 5分钟 |
| [训练第一个模型](getting-started/first-model.md) | 端到端训练流程 | 30分钟 |
| [常见问题](getting-started/troubleshooting.md) | 遇到问题？看这里 | - |

### 📖 教程系列 (Tutorials)

手把手教学，从零到掌握：

| 教程 | 主题 | 难度 | 时间 |
|------|------|------|------|
| [教程 1: AttnRes 入门](tutorials/tutorial-01-attnres.md) | 块注意力残差基础 | ⭐ | 30分钟 |
| [教程 2: qTTT 使用](tutorials/tutorial-02-qttt.md) | 查询时训练 | ⭐ | 30分钟 |
| [教程 3: RaBitQ 压缩](tutorials/tutorial-03-rabitq.md) | KV缓存压缩 | ⭐ | 30分钟 |
| [教程 4: Engram 记忆](tutorials/tutorial-04-engram.md) | n-gram记忆机制 | ⭐⭐ | 45分钟 |
| [教程 5: 动态门控](tutorials/tutorial-05-gating.md) | 自适应计算控制 | ⭐⭐ | 45分钟 |
| [教程 6: 端到端训练](tutorials/tutorial-06-end-to-end.md) | 完整训练流程 | ⭐⭐ | 60分钟 |

### 📋 操作指南 (How-To Guides)

解决特定问题的步骤指南：

| 指南 | 场景 | 难度 |
|------|------|------|
| [训练小模型](how-to/train-small-model.md) | 资源受限环境训练 | ⭐⭐ |
| [运行实验](how-to/run-experiments.md) | 复现论文实验 | ⭐⭐ |
| [评估模型](how-to/evaluate-model.md) | 模型性能评估 | ⭐⭐ |
| [调试训练](how-to/debug-training.md) | 训练问题排查 | ⭐⭐⭐ |
| [内存分析](how-to/profile-memory.md) | 性能优化 | ⭐⭐⭐ |
| [添加新模块](how-to/add-new-module.md) | 扩展开发 | ⭐⭐⭐⭐ |
| [复现论文结果](how-to/reproduce-paper.md) | 论文对齐验证 | ⭐⭐⭐ |

### 📖 参考文档 (Reference)

精确的技术信息，便于查阅：

#### API 参考
- [API 概览](reference/api/index.md)
- [AttnRes API](reference/api/attnres.md) - 块注意力残差
- [qTTT API](reference/api/qttt.md) - 查询时训练
- [RaBitQ API](reference/api/rabitq.md) - KV缓存压缩
- [Engram API](reference/api/engram.md) - n-gram记忆
- [Gating API](reference/api/gating.md) - 动态门控
- [Models API](reference/api/models.md) - 模型定义

#### 配置参考
- [配置概览](reference/config/index.md)
- [模型配置](reference/config/model-configs.md) - 所有模型配置参数
- [训练配置](reference/config/training-configs.md) - 训练超参数
- [实验配置](reference/config/experiment-configs.md) - 实验框架配置

#### CLI 参考
- [CLI 概览](reference/cli/index.md)
- [adn-train](reference/cli/adn-train.md) - 训练命令
- [adn-eval](reference/cli/adn-eval.md) - 评估命令
- [adn-benchmark](reference/cli/adn-benchmark.md) - 基准测试

#### 其他参考
- [术语表](reference/glossary.md) - 专业术语解释
- [技术文档](TECHNICAL_DOCUMENTATION.md) - 综合技术文档
- [架构文档](ARCHITECTURE.md) - 系统架构说明

### 💡 解释文档 (Explanation)

理解概念和背景：

| 文档 | 内容 | 深度 |
|------|------|------|
| [设计决策记录](explanation/design-decisions.md) | 关键设计选择及原因 | 中等 |
| [架构演进](explanation/architecture-evolution.md) | 项目发展历程 | 浅层 |
| [论文对齐说明](explanation/paper-alignment.md) | 代码与论文对应关系 | 详细 |
| [性能优化原理](explanation/performance-optimization.md) | 优化策略详解 | 深入 |
| [研究背景](explanation/research-background.md) | 相关研究工作 | 中等 |

---

## 🗂️ 按角色导航

### 👨‍💻 研究人员

关注论文复现和实验：
1. [论文对齐说明](explanation/paper-alignment.md) - 理解代码与论文的对应
2. [复现论文结果](how-to/reproduce-paper.md) - 复现步骤
3. [运行实验](how-to/run-experiments.md) - 实验框架使用
4. [Reports 目录](reports/) - 实验报告

### 👩‍💻 开发者

关注代码开发和扩展：
1. [架构文档](ARCHITECTURE.md) - 理解系统架构
2. [API 参考](reference/api/) - 查阅接口文档
3. [添加新模块](how-to/add-new-module.md) - 扩展开发指南
4. [AGENTS.md](../AGENTS.md) - AI Agent 开发指南

### 🧪 实验工程师

关注模型训练和评估：
1. [训练第一个模型](getting-started/first-model.md) - 快速上手
2. [模型配置参考](reference/config/model-configs.md) - 配置参数
3. [调试训练](how-to/debug-training.md) - 问题排查
4. [评估模型](how-to/evaluate-model.md) - 评估方法

### 🔧 系统工程师

关注部署和性能：
1. [A100 完整指南](guides/A100_80G_COMPLETE_GUIDE.md) - 环境配置
2. [内存分析](how-to/profile-memory.md) - 性能优化
3. [RaBitQ 指南](guides/RABITQ_GUIDE.md) - 压缩部署
4. [性能优化原理](explanation/performance-optimization.md) - 优化策略

---

## 📊 项目文档

### 论文相关
- [论文草稿](../ADN_paper.md) - 主论文 (根目录)
- [MATDO-E 论文](../matdo-e_paper.md) - MATDO-E 论文 (根目录)
- [Papers 目录](papers/) - 更多论文变体

### 项目报告
- [Reports 目录](reports/) - 实验报告和验证结果
- [Project 目录](project/) - 项目管理和进度跟踪

### 审计文档
- [Audits 目录](audits/) - 代码审计和引用检查

---

## 🔗 快速链接

### 常用命令
```bash
# 安装
pip install -e ".[dev]"

# 测试
pytest tests/ -v --tb=short --ignore=tests/legacy

# 训练
python3 scripts/training/train_model.py --model-size small --output-dir results/small

# 实验
make quick     # 快速实验
make validate  # 验证实验
make full      # 完整实验
```

### 关键文件
- [项目 README](../README.md) - 项目概览
- [AGENTS.md](../AGENTS.md) - Agent 开发指南
- [PROJECT_ORGANIZATION.md](../PROJECT_ORGANIZATION.md) - 项目结构
- [Makefile](../Makefile) - 构建命令
- [pyproject.toml](../pyproject.toml) - 项目配置

### 外部资源
- [GitHub Issues](https://github.com/your-org/Adaptive-Deep-Networks/issues) - 问题反馈
- [Discussion](https://github.com/your-org/Adaptive-Deep-Networks/discussions) - 讨论区

---

## 🆘 获取帮助

### 遇到问题？

1. **查看常见问题**: [troubleshooting.md](getting-started/troubleshooting.md)
2. **搜索文档**: 使用页面顶部的搜索功能
3. **查看 Issues**: 在 GitHub Issues 中搜索类似问题
4. **发起讨论**: 在 GitHub Discussions 中提问

### 文档问题反馈

如果发现文档有误或缺失：
1. 在 GitHub Issues 中创建 `documentation` 标签的 issue
2. 或直接提交 PR 修复

---

## 📝 文档体系说明

本文档体系基于 [Divio Documentation System](https://documentation.divio.com/) 建立，将文档分为四类：

| 类型 | 目的 | 示例 |
|------|------|------|
| **教程** | 学习导向，手把手教学 | [训练第一个模型](getting-started/first-model.md) |
| **操作指南** | 任务导向，解决特定问题 | [调试训练](how-to/debug-training.md) |
| **参考** | 信息导向，精确技术细节 | [API 参考](reference/api/) |
| **解释** | 理解导向，阐述概念背景 | [设计决策](explanation/design-decisions.md) |

详细说明见 [DOCUMENTATION_SYSTEM.md](DOCUMENTATION_SYSTEM.md)。

---

*最后更新: 2026-04-24 | 文档版本: 1.0*
