---
title: "参考文档"
description: "ADN 技术参考，API、配置、CLI 完整文档"
category: "reference"
difficulty: "intermediate"
last_updated: "2026-04-24"
---

# 参考文档

欢迎来到 ADN 参考文档！这里提供精确的技术信息，便于快速查阅。

---

## 📚 参考类型

### [API 参考](api/)

详细的 API 文档，包含类、函数、参数说明。

| 模块 | 描述 |
|------|------|
| [AttnRes](api/attnres.md) | 块注意力残差 |
| [qTTT](api/qttt.md) | 查询时训练 |
| [RaBitQ](api/rabitq.md) | KV 缓存压缩 |
| [Engram](api/engram.md) | n-gram 记忆 |
| [Gating](api/gating.md) | 动态门控 |
| [Models](api/models.md) | 模型定义 |

### [配置参考](config/)

所有配置选项的完整说明。

| 配置类型 | 描述 |
|----------|------|
| [模型配置](config/model-configs.md) | 模型架构参数 |
| [训练配置](config/training-configs.md) | 训练超参数 |
| [实验配置](config/experiment-configs.md) | 实验框架配置 |

### [CLI 参考](cli/)

命令行工具的完整文档。

| 命令 | 描述 |
|------|------|
| [adn-train](cli/adn-train.md) | 训练命令 |
| [adn-eval](cli/adn-eval.md) | 评估命令 |
| [adn-benchmark](cli/adn-benchmark.md) | 基准测试 |

### [术语表](glossary.md)

专业术语解释。

---

## 🔍 快速查找

### 按任务查找

**...查找类或函数定义**
→ [API 参考](api/)

**...了解配置参数**
→ [配置参考](config/)

**...查看命令行用法**
→ [CLI 参考](cli/)

**...理解专业术语**
→ [术语表](glossary.md)

---

## 📖 使用建议

### 开发者

- 编码时查阅 [API 参考](api/)
- 配置模型时查阅 [配置参考](config/)
- 写脚本时查阅 [CLI 参考](cli/)

### 研究人员

- 理解参数含义 → [配置参考](config/)
- 复现实验 → [CLI 参考](cli/)
- 理解术语 → [术语表](glossary.md)

### 系统工程师

- 部署配置 → [配置参考](config/)
- 命令行操作 → [CLI 参考](cli/)

---

## 📝 文档规范

### API 文档格式

```python
class ExampleClass:
    """简短描述。

    详细描述，包含使用场景和注意事项。

    Args:
        param1: 参数1说明
        param2: 参数2说明

    Returns:
        返回值说明

    Example:
        >>> example = ExampleClass(param1=1, param2=2)
        >>> result = example.method()
    """
```

### 配置文档格式

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `param1` | `int` | `32` | 参数说明 |

---

## 🔄 更新频率

| 文档类型 | 更新触发条件 |
|----------|--------------|
| API 参考 | 代码变更时 |
| 配置参考 | 配置变更时 |
| CLI 参考 | CLI 变更时 |
| 术语表 | 按需更新 |

---

## 📚 相关资源

- [入门系列](../getting-started/) - 基础设置
- [教程系列](../tutorials/) - 系统学习
- [操作指南](../how-to/) - 问题解决
- [解释文档](../explanation/) - 原理理解

---

*需要查找特定信息？使用页面搜索或查看左侧导航。*
