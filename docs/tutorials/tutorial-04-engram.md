---
title: "教程 4: Engram 记忆"
description: "学习 Engram n-gram 记忆机制的原理和使用方法"
category: "tutorials"
difficulty: "intermediate"
duration: "45分钟"
prerequisites: ["完成教程 1-3", "哈希表基础"]
last_updated: "2026-04-24"
---

# 教程 4: Engram 记忆

本教程将带你了解 ADN 的 **Engram n-gram 记忆机制**。你将学习它的设计原理、多粒度哈希映射，以及如何与 Transformer 集成来增强长上下文能力。

---

## 你将学习

- [ ] n-gram 记忆机制原理
- [ ] Engram 模块的配置和使用
- [ ] 多粒度哈希映射
- [ ] 与 Transformer 集成

---

## 1. 什么是 Engram？

### 1.1 背景

标准 Transformer 在处理长上下文时面临挑战：
- 注意力稀释：远距离 token 的注意力权重变得极小
- 计算成本：自注意力复杂度为 O(L²)
- 位置编码限制：难以处理超长序列

**Engram 的解决方案**: 使用 **n-gram 记忆机制** 来补充注意力，显式存储和检索重复模式。

### 1.2 核心思想

Engram 受到人类记忆的启发：
- **短期记忆**: 标准注意力机制处理最近的信息
- **长期记忆**: Engram 存储和检索历史中的重复模式

```
输入序列: [The, cat, sat, on, the, mat, and, the, cat, ...]
              ↓
┌─────────────────────────────────────────┐
│  n=2: (The, cat)→sat, (cat, sat)→on, ...│
│  n=3: (The, cat, sat)→on, ...           │
│  n=4: (The, cat, sat, on)→the, ...      │
└─────────────────────────────────────────┘
              ↓
当再次看到 "The cat sat" 时，快速检索后续 token
```

### 1.3 工作流程

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   输入序列    │────>│  n-gram 提取  │────>│  哈希编码    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  ↓
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   增强表示    │<────│  记忆检索    │<────│  记忆存储    │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## 2. 多粒度哈希映射

### 2.1 为什么需要多粒度？

不同长度的 n-gram 捕获不同级别的模式：

| n-gram 长度 | 捕获的模式 | 示例 |
|-------------|-----------|------|
| n=2 | 局部搭配 | "New York", "machine learning" |
| n=3 | 短语结构 | "in order to", "due to the" |
| n=4 | 句法模式 | "The quick brown fox" |

### 2.2 哈希策略

Engram 使用**多哈希函数**来减少碰撞：

```python
# 对于每个 n-gram，计算多个哈希值
hash_values = [
    hash1(ngram),
    hash2(ngram),
    hash3(ngram)
]

# 存储到对应的桶中
for h in hash_values:
    memory_table[h % table_size] = embedding
```

---

## 3. 代码实践

### 3.1 基础配置

```python
from src.engram import EngramConfig, Engram

# 创建配置
config = EngramConfig(
    ngram_sizes=[2, 3, 4],      # 多粒度 n-gram
    embedding_dim=512,           # 嵌入维度
    memory_size=100000,          # 记忆表大小
    num_hashes=3,                # 哈希函数数量
    use_compression=True         # 启用压缩
)

# 创建 Engram 模块
engram = Engram(config)

print(f"Engram Config:")
print(f"  N-gram sizes: {config.ngram_sizes}")
print(f"  Embedding dim: {config.embedding_dim}")
print(f"  Memory size: {config.memory_size:,}")
```

### 3.2 存储和检索

```python
import torch

# 模拟 token 序列
token_ids = torch.randint(0, 50000, (1, 100))  # [batch, seq_len]

# 存储到记忆
engram.store(token_ids)

# 检索记忆
query_tokens = torch.randint(0, 50000, (1, 10))
retrieved = engram.retrieve(query_tokens)

print(f"Query shape: {query_tokens.shape}")
print(f"Retrieved shape: {retrieved.shape}")
```

### 3.3 与 Transformer 集成

```python
from src.models import AdaptiveTransformer, AttnResSmallConfig
from src.engram import EngramConfig

# 创建带 Engram 的配置
config = AttnResSmallConfig(
    use_engram=True,
    engram_config=EngramConfig(
        ngram_sizes=[2, 3, 4],
        embedding_dim=1408,  # 与 hidden_dim 匹配
        memory_size=50000
    )
)

# 创建模型
model = AdaptiveTransformer(config)

# 前向传播自动使用 Engram
output = model(input_ids)
```

---

## 4. 完整示例

```python
#!/usr/bin/env python3
"""
Engram 记忆示例
运行: python tutorial_04_engram_demo.py
"""

import torch
from src.engram import Engram, EngramConfig

def demo_basic_engram():
    """演示基本的 Engram 使用"""
    print("=" * 60)
    print("演示 1: 基本 Engram 配置")
    print("=" * 60)

    # 不同场景的配置
    configs = {
        '轻量级': EngramConfig(
            ngram_sizes=[2, 3],
            embedding_dim=256,
            memory_size=10000
        ),
        '标准': EngramConfig(
            ngram_sizes=[2, 3, 4],
            embedding_dim=512,
            memory_size=50000
        ),
        '高性能': EngramConfig(
            ngram_sizes=[2, 3, 4, 5],
            embedding_dim=1024,
            memory_size=100000
        ),
    }

    for name, config in configs.items():
        engram = Engram(config)
        memory_mb = config.memory_size * config.embedding_dim * 4 / 1024 / 1024

        print(f"\n{name}配置:")
        print(f"  N-gram: {config.ngram_sizes}")
        print(f"  内存: {memory_mb:.1f} MB")
        print(f"  参数量: {sum(p.numel() for p in engram.parameters()):,}")

def demo_ngram_extraction():
    """演示 n-gram 提取"""
    print("\n" + "=" * 60)
    print("演示 2: N-gram 提取")
    print("=" * 60)

    # 模拟文本序列
    tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'looked', 'at', 'the', 'bird']
    token_ids = list(range(len(tokens)))

    print(f"\n输入序列: {' '.join(tokens)}")
    print(f"Token IDs: {token_ids}")

    # 提取 n-grams
    for n in [2, 3, 4]:
        ngrams = []
        for i in range(len(token_ids) - n + 1):
            ngram = tuple(token_ids[i:i+n])
            ngrams.append((ngram, tokens[i:i+n]))

        print(f"\n{n}-grams ({len(ngrams)} 个):")
        for i, (ids, words) in enumerate(ngrams[:5]):  # 只显示前5个
            print(f"  {ids} -> '{' '.join(words)}'")

def demo_memory_retrieval():
    """演示记忆检索"""
    print("\n" + "=" * 60)
    print("演示 3: 记忆检索")
    print("=" * 60)

    config = EngramConfig(
        ngram_sizes=[2, 3],
        embedding_dim=128,
        memory_size=1000
    )
    engram = Engram(config)

    # 存储一些模式
    patterns = [
        torch.tensor([[1, 2, 3, 4, 5]]),
        torch.tensor([[2, 3, 4, 5, 6]]),
        torch.tensor([[1, 2, 3, 7, 8]]),
    ]

    print("\n存储模式:")
    for i, pattern in enumerate(patterns):
        engram.store(pattern)
        print(f"  模式 {i+1}: {pattern[0].tolist()}")

    # 检索相似模式
    query = torch.tensor([[1, 2, 3]])
    retrieved = engram.retrieve(query)

    print(f"\n查询: {query[0].tolist()}")
    print(f"检索结果形状: {retrieved.shape}")
    print(f"检索结果范数: {torch.norm(retrieved).item():.4f}")

def demo_hit_rate_analysis():
    """分析记忆命中率"""
    print("\n" + "=" * 60)
    print("演示 4: 记忆命中率分析")
    print("=" * 60)

    config = EngramConfig(
        ngram_sizes=[2, 3, 4],
        embedding_dim=256,
        memory_size=5000
    )
    engram = Engram(config)

    # 生成重复模式的数据
    torch.manual_seed(42)

    # 存储阶段
    stored_patterns = set()
    for _ in range(100):
        pattern = torch.randint(0, 1000, (1, 20))
        engram.store(pattern)
        # 记录 3-grams
        for i in range(18):
            stored_patterns.add(tuple(pattern[0, i:i+3].tolist()))

    # 测试阶段
    hits = 0
    total = 0
    for _ in range(50):
        query = torch.randint(0, 1000, (1, 10))
        for i in range(8):
            ngram = tuple(query[0, i:i+3].tolist())
            total += 1
            if ngram in stored_patterns:
                hits += 1

    hit_rate = hits / total * 100
    print(f"\n存储模式数: {len(stored_patterns)}")
    print(f"查询次数: {total}")
    print(f"命中次数: {hits}")
    print(f"命中率: {hit_rate:.1f}%")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Engram 记忆教程")
    print("=" * 60 + "\n")

    demo_basic_engram()
    demo_ngram_extraction()
    demo_memory_retrieval()
    demo_hit_rate_analysis()

    print("\n" + "=" * 60)
    print("教程完成！")
    print("=" * 60)
    print("\n下一步: 学习教程 5 - 动态门控")
```

---

## 5. 配置参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ngram_sizes` | List[int] | [2,3,4] | n-gram 长度列表 |
| `embedding_dim` | int | 512 | 嵌入向量维度 |
| `memory_size` | int | 50000 | 记忆表大小 |
| `num_hashes` | int | 3 | 哈希函数数量 |
| `use_compression` | bool | True | 是否压缩存储 |

### 最佳实践

1. **n-gram 选择**: [2,3,4] 是通用配置，捕获从局部搭配到短语结构
2. **记忆表大小**: 根据序列长度和词汇量调整，通常 50K-100K
3. **哈希数量**: 3-5 个哈希函数可平衡碰撞率和计算成本

---

## 6. 练习

### 练习 1: 对比不同 n-gram 配置

```python
for sizes in [[2], [2,3], [2,3,4], [2,3,4,5]]:
    config = EngramConfig(ngram_sizes=sizes)
    # 测试检索质量和内存使用
```

### 练习 2: 实现自定义哈希函数

```python
def custom_hash(ngram, seed):
    """自定义哈希函数"""
    result = seed
    for token in ngram:
        result = (result * 31 + token) % (2**32)
    return result
```

---

## 7. 常见问题

**Q: Engram 与标准注意力如何配合？**
A: Engram 作为补充机制，其输出会与注意力输出融合（通常是相加或拼接）。

**Q: 记忆表满了怎么办？**
A: 使用 LRU 淘汰策略或频率加权替换。

**Q: Engram 适合什么场景？**
A: 特别适合有重复模式的长文本（代码、法律文档、技术手册）。

---

## 8. 下一步

完成本教程后，你应该能够：
- [x] 理解 n-gram 记忆机制
- [x] 配置和使用 Engram 模块
- [x] 与 Transformer 集成

**下一步**: [教程 5: 动态门控](tutorial-05-gating.md)

---

## 参考

- [Engram API 文档](../reference/api/engram.md)
- [论文 §5.5](../papers/adn-paper.md#55-engram-memory)
- [源码](../../src/engram/)
