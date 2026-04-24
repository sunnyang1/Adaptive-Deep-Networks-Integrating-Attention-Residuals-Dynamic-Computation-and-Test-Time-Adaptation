---
title: "贡献指南"
description: "如何为 ADN 项目做出贡献"
category: "contributing"
difficulty: "intermediate"
last_updated: "2026-04-24"
---

# 贡献指南

感谢你对 ADN 项目的关注！本文档指导你如何为项目做出贡献。

---

## 🤝 贡献方式

### 1. 报告问题 (Issues)

发现 bug 或有功能建议？欢迎提交 Issue：

- **Bug 报告**: 描述问题、复现步骤、环境信息
- **功能请求**: 描述需求、使用场景、预期行为
- **文档问题**: 指出错误、缺失或不清晰的内容

### 2. 提交代码 (Pull Requests)

修复 bug 或实现新功能？欢迎提交 PR：

- 确保代码符合项目规范
- 添加必要的测试
- 更新相关文档

### 3. 改进文档

发现文档有误或可以改进？直接提交文档修改：

- 修正错误
- 补充缺失内容
- 改进表达清晰度

### 4. 分享经验

使用 ADN 的经验和技巧？欢迎分享：

- 撰写博客文章
- 制作教程视频
- 在 Discussion 中分享

---

## 📝 提交 Issue

### Bug 报告模板

```markdown
## 问题描述
清晰描述遇到的问题

## 复现步骤
1. 执行 '...'
2. 点击 '...'
3. 出现错误

## 预期行为
描述预期应该发生什么

## 实际行为
描述实际发生了什么

## 环境信息
- OS: [e.g. Ubuntu 22.04]
- Python: [e.g. 3.12.0]
- PyTorch: [e.g. 2.1.0]
- CUDA: [e.g. 12.1]
- ADN 版本: [e.g. 0.2.0]

## 错误日志
```
粘贴完整的错误日志
```

## 已尝试的解决方案
描述你尝试过的解决方法
```

### 功能请求模板

```markdown
## 功能描述
清晰描述你希望添加的功能

## 使用场景
描述这个功能的使用场景

## 预期行为
描述功能应该如何工作

## 可能的实现方案
如果你有实现思路，请描述

## 替代方案
描述你考虑过的替代方案
```

---

## 🔧 提交 Pull Request

### 工作流程

1. **Fork 仓库**
   ```bash
   # 点击 GitHub 上的 Fork 按钮
   ```

2. **克隆你的 Fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Adaptive-Deep-Networks.git
   cd Adaptive-Deep-Networks
   ```

3. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/bug-description
   ```

4. **进行修改**
   - 编写代码
   - 添加测试
   - 更新文档

5. **提交更改**
   ```bash
   git add .
   git commit -m "描述你的更改"
   ```

6. **推送到 Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **创建 Pull Request**
   - 在 GitHub 上点击 "New Pull Request"
   - 填写 PR 描述
   - 等待审查

### PR 描述模板

```markdown
## 描述
简要描述这个 PR 的目的

## 更改类型
- [ ] Bug 修复
- [ ] 新功能
- [ ] 性能优化
- [ ] 文档更新
- [ ] 代码重构
- [ ] 其他

## 测试
- [ ] 添加了单元测试
- [ ] 添加了集成测试
- [ ] 手动测试通过
- [ ] 所有现有测试通过

## 文档
- [ ] 更新了 API 文档
- [ ] 更新了用户文档
- [ ] 添加了代码注释

## 检查清单
- [ ] 代码符合项目规范
- [ ] 通过了 lint 检查
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
```

---

## 🎨 代码规范

### Python 代码风格

项目使用以下工具保证代码质量：

```bash
# 格式化
black src/ tests/ experiments/ scripts/

# 代码检查
ruff check src/ tests/ experiments/ scripts/

# 类型检查
mypy src/
```

### 代码规范要点

1. **格式化**: 使用 black，行长度 100
2. **导入**: 使用绝对导入
3. **类型提示**: 为公共 API 添加类型提示
4. **文档字符串**: 使用 Google 风格
5. **命名**: 遵循 PEP 8

### 示例代码

```python
"""模块描述。

详细描述模块的功能和使用方法。
"""

from typing import Optional, List
import torch
import torch.nn as nn


class ExampleClass(nn.Module):
    """简短描述。

    详细描述类的功能和使用场景。

    Args:
        param1: 参数1的说明
        param2: 参数2的说明

    Attributes:
        attr1: 属性1的说明

    Example:
        >>> example = ExampleClass(param1=1, param2=2)
        >>> result = example.forward(x)
    """

    def __init__(self, param1: int, param2: float = 0.5) -> None:
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状为 (batch, seq_len, hidden_dim)

        Returns:
            输出张量，形状为 (batch, seq_len, hidden_dim)

        Raises:
            ValueError: 当输入维度不匹配时
        """
        # 实现代码
        return x
```

---

## 🧪 测试要求

### 测试类型

1. **单元测试**: 测试单个函数或类
2. **集成测试**: 测试模块间的交互
3. **端到端测试**: 测试完整流程

### 运行测试

```bash
# 所有测试
pytest tests/ -v --tb=short --ignore=tests/legacy

# 单元测试
pytest tests/unit/ -v --tb=short

# 集成测试
pytest tests/integration/ -v --tb=short

# 带覆盖率
pytest tests/ -v --tb=short --ignore=tests/legacy --cov=src
```

### 编写测试

```python
# tests/unit/test_example.py
import pytest
import torch

from src.example import ExampleClass


class TestExampleClass:
    """Test ExampleClass."""

    def test_init(self):
        """Test initialization."""
        example = ExampleClass(param1=10)
        assert example.param1 == 10

    def test_forward(self):
        """Test forward pass."""
        example = ExampleClass(param1=10)
        x = torch.randn(2, 10, 512)
        output = example(x)
        assert output.shape == x.shape

    def test_forward_invalid_input(self):
        """Test forward with invalid input."""
        example = ExampleClass(param1=10)
        with pytest.raises(ValueError):
            example(torch.randn(10))  # Wrong shape
```

---

## 📝 文档贡献

### 文档类型

1. **API 文档**: 代码中的 docstrings
2. **用户文档**: `docs/` 目录下的 Markdown
3. **教程**: `docs/tutorials/` 中的手把手教程
4. **指南**: `docs/how-to/` 中的操作指南

### 文档规范

- 使用清晰的标题结构
- 提供代码示例
- 保持中英文术语一致
- 更新相关链接

### 文档审查清单

- [ ] 文档结构清晰
- [ ] 代码示例可运行
- [ ] 术语使用一致
- [ ] 链接有效
- [ ] 无拼写错误

---

## 🏷️ 标签说明

Issue 和 PR 使用以下标签：

| 标签 | 含义 | 使用场景 |
|------|------|----------|
| `bug` | Bug | 报告或修复问题 |
| `feature` | 新功能 | 请求或实现功能 |
| `documentation` | 文档 | 文档相关 |
| `enhancement` | 改进 | 代码改进 |
| `performance` | 性能 | 性能优化 |
| `refactor` | 重构 | 代码重构 |
| `good first issue` | 新手友好 | 适合新贡献者 |
| `help wanted` | 需要帮助 | 需要社区帮助 |

---

## 💬 沟通渠道

- **GitHub Issues**: Bug 报告和功能请求
- **GitHub Discussions**: 一般讨论和问题
- **Pull Requests**: 代码审查和讨论

---

## 🙏 感谢

感谢所有为 ADN 做出贡献的人！

---

*贡献指南持续更新中。如有疑问，请在 Discussions 中提问。*
