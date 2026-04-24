# ADN 技术文档体系

> **文档版本**: 1.0
> **建立日期**: 2026-04-24
> **维护者**: 技术文档工程师

---

## 📚 文档体系架构

本文档体系采用 **Divio文档四象限模型**，将文档分为四类，每类有明确的目标读者和使用场景：

```
┌─────────────────────┬─────────────────────┐
│    📖 教程 (Tutorials)   │  📋 操作指南 (How-To Guides) │
│    (学习导向)            │  (任务导向)                  │
├─────────────────────┼─────────────────────┤
│    📖 参考 (Reference)   │  💡 解释 (Explanation)       │
│    (信息导向)            │  (理解导向)                  │
└─────────────────────┴─────────────────────┘
```

---

## 📁 文档目录结构

```
docs/
├── README.md                    # 文档首页导航
├── DOCUMENTATION_SYSTEM.md      # 本文档 - 体系说明
├── ARCHITECTURE.md              # 系统架构文档
├── TECHNICAL_DOCUMENTATION.md   # 综合技术文档
│
├── getting-started/             # 🚀 入门系列
│   ├── index.md                 # 入门导航
│   ├── installation.md          # 安装指南
│   ├── quickstart.md            # 5分钟快速开始
│   ├── first-model.md           # 训练你的第一个模型
│   └── troubleshooting.md       # 常见问题
│
├── tutorials/                   # 📖 教程系列 (学习导向)
│   ├── index.md                 # 教程导航
│   ├── tutorial-01-attnres.md   # 教程1: AttnRes入门
│   ├── tutorial-02-qttt.md      # 教程2: qTTT使用
│   ├── tutorial-03-rabitq.md    # 教程3: RaBitQ压缩
│   ├── tutorial-04-engram.md    # 教程4: Engram记忆
│   ├── tutorial-05-gating.md    # 教程5: 动态门控
│   └── tutorial-06-end-to-end.md # 教程6: 端到端训练
│
├── how-to/                      # 📋 操作指南 (任务导向)
│   ├── index.md                 # 指南导航
│   ├── train-small-model.md     # 如何训练小模型
│   ├── run-experiments.md       # 如何运行实验
│   ├── evaluate-model.md        # 如何评估模型
│   ├── debug-training.md        # 如何调试训练
│   ├── profile-memory.md        # 如何分析内存
│   ├── add-new-module.md        # 如何添加新模块
│   └── reproduce-paper.md       # 如何复现论文结果
│
├── reference/                   # 📖 参考文档 (信息导向)
│   ├── index.md                 # 参考导航
│   ├── api/                     # API文档
│   │   ├── index.md             # API概览
│   │   ├── attnres.md           # AttnRes API
│   │   ├── qttt.md              # qTTT API
│   │   ├── rabitq.md            # RaBitQ API
│   │   ├── engram.md            # Engram API
│   │   ├── gating.md            # Gating API
│   │   └── models.md            # Models API
│   ├── config/                  # 配置参考
│   │   ├── index.md             # 配置概览
│   │   ├── model-configs.md     # 模型配置
│   │   ├── training-configs.md  # 训练配置
│   │   └── experiment-configs.md # 实验配置
│   ├── cli/                     # CLI参考
│   │   ├── index.md             # CLI概览
│   │   ├── adn-train.md         # 训练命令
│   │   ├── adn-eval.md          # 评估命令
│   │   └── adn-benchmark.md     # 基准命令
│   └── glossary.md              # 术语表
│
├── explanation/                 # 💡 解释文档 (理解导向)
│   ├── index.md                 # 解释导航
│   ├── design-decisions.md      # 设计决策记录
│   ├── architecture-evolution.md # 架构演进
│   ├── paper-alignment.md       # 论文对齐说明
│   ├── performance-optimization.md # 性能优化原理
│   └── research-background.md   # 研究背景
│
├── guides/                      # 📚 环境特定指南 (保留)
│   ├── README_FOR_BEGINNERS.md  # 新手指南
│   ├── MATDO_E_A100_BEGINNER_GUIDE.md # A100指南
│   ├── A100_80G_COMPLETE_GUIDE.md # A100完整指南
│   ├── RABITQ_GUIDE.md          # RaBitQ指南
│   └── TURBOQUANT_TO_RABITQ_MIGRATION_GUIDE.md # 迁移指南
│
├── papers/                      # 📝 论文相关 (保留)
│   └── ...
│
├── reports/                     # 📊 报告 (保留)
│   └── ...
│
└── project/                     # 📋 项目管理 (保留)
    └── ...
```

---

## 🎯 文档类型说明

### 1. 教程 (Tutorials) - `tutorials/`

**目标**: 帮助初学者从零开始学习
**特点**: 手把手的步骤教学，强调学习体验
**示例**: "训练你的第一个AttnRes模型"

| 文档 | 目标读者 | 预计时间 | 前置知识 |
|------|----------|----------|----------|
| tutorial-01-attnres.md | 初学者 | 30分钟 | Python, PyTorch基础 |
| tutorial-02-qttt.md | 初学者 | 30分钟 | 完成教程1 |
| tutorial-03-rabitq.md | 初学者 | 30分钟 | 完成教程1 |
| tutorial-04-engram.md | 中级 | 45分钟 | 完成教程1-3 |
| tutorial-05-gating.md | 中级 | 45分钟 | 完成教程1-3 |
| tutorial-06-end-to-end.md | 中级 | 60分钟 | 完成教程1-5 |

### 2. 操作指南 (How-To Guides) - `how-to/`

**目标**: 帮助用户完成特定任务
**特点**: 问题导向，假设读者已具备基础知识
**示例**: "如何在小内存GPU上训练模型"

| 文档 | 适用场景 | 难度 |
|------|----------|------|
| train-small-model.md | 资源受限环境 | ⭐⭐ |
| run-experiments.md | 复现实验 | ⭐⭐ |
| evaluate-model.md | 模型评估 | ⭐⭐ |
| debug-training.md | 训练出错 | ⭐⭐⭐ |
| profile-memory.md | 性能优化 | ⭐⭐⭐ |
| add-new-module.md | 扩展开发 | ⭐⭐⭐⭐ |
| reproduce-paper.md | 论文复现 | ⭐⭐⭐ |

### 3. 参考文档 (Reference) - `reference/`

**目标**: 提供精确的技术信息
**特点**: 信息密集，结构清晰，便于查找
**示例**: API文档、配置参数表

| 文档类型 | 内容 | 更新频率 |
|----------|------|----------|
| API文档 | 类、函数、参数说明 | 每次代码变更 |
| 配置参考 | 所有配置选项 | 每次配置变更 |
| CLI参考 | 命令行参数 | 每次CLI变更 |
| 术语表 | 专业术语解释 | 按需更新 |

### 4. 解释文档 (Explanation) - `explanation/`

**目标**: 帮助读者理解概念和背景
**特点**: 理论导向，解释"为什么"
**示例**: "为什么AttnRes使用块结构"

| 文档 | 主题 | 深度 |
|------|------|------|
| design-decisions.md | 关键设计决策 | 中等 |
| architecture-evolution.md | 架构演进历史 | 浅层 |
| paper-alignment.md | 代码与论文对齐 | 详细 |
| performance-optimization.md | 优化原理 | 深入 |
| research-background.md | 研究背景 | 中等 |

---

## 🔄 文档维护流程

### 更新触发条件

```
代码变更 ──┬──> API变更 ──> 更新 reference/api/
           ├──> 配置变更 ──> 更新 reference/config/
           ├──> CLI变更 ──> 更新 reference/cli/
           ├──> 新功能 ──> 更新 tutorials/ + how-to/
           └──> Bug修复 ──> 更新 getting-started/troubleshooting.md
```

### 文档审查清单

- [ ] 代码示例经过测试，可运行
- [ ] 所有链接有效
- [ ] 版本号正确
- [ ] 术语使用一致
- [ ] 中英文术语对照完整

---

## 📝 文档写作规范

### 文件头模板

```markdown
---
title: "文档标题"
description: "简短描述"
category: "tutorials|how-to|reference|explanation"
difficulty: "beginner|intermediate|advanced"
time: "预计阅读时间"
prerequisites: ["前置知识1", "前置知识2"]
last_updated: "2026-04-24"
version: "0.2.0"
---

# 文档标题

> 一句话总结本文档的目的

## 概述
...
```

### 代码示例规范

- 所有代码示例必须可运行
- 包含必要的导入语句
- 提供预期输出示例
- 标注关键参数的含义

### 中英文对照表

| 英文术语 | 中文翻译 | 备注 |
|----------|----------|------|
| Attention Residuals | 注意力残差 | AttnRes |
| Query-Only Test-Time Training | 仅查询测试时训练 | qTTT |
| Block | 块 | AttnRes中的块结构 |
| Pseudo-Query | 伪查询 | 可学习的查询向量 |
| KV Cache | KV缓存 | Key-Value缓存 |
| Gating | 门控 | 动态计算控制 |
| Engram | 记忆印迹 | n-gram记忆机制 |

---

## 🔗 相关资源

- [项目README](../README.md)
- [AGENTS.md](../AGENTS.md) - AI Agent开发指南
- [PROJECT_ORGANIZATION.md](../PROJECT_ORGANIZATION.md) - 项目结构
- [GitHub Issues](https://github.com/your-org/Adaptive-Deep-Networks/issues)

---

*本文档体系遵循 Divio Documentation System 原则建立。*
