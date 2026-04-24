---
title: "如何评估模型"
description: "ADN 模型评估方法和指标"
category: "how-to"
difficulty: "intermediate"
duration: "25分钟"
last_updated: "2026-04-24"
---

# 如何评估模型

本指南介绍 ADN 模型的评估方法和常用指标。

---

## 评估指标

### 1. 困惑度 (Perplexity)

困惑度是语言模型的标准评估指标：

```python
import torch
import math

def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch.get('labels', input_ids)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity

# 使用
ppl = compute_perplexity(model, eval_dataloader, device)
print(f"困惑度: {ppl:.2f}")
```

### 2. Needle-in-Haystack 测试

测试长上下文检索能力：

```bash
# 运行 needle-in-haystack 评估
python scripts/evaluation/run_benchmarks.py \
    --benchmark needle \
    --model-path ./outputs/my_model \
    --context-lengths 1024 4096 16384 32768
```

```python
from src.benchmarks import NeedleInHaystackBenchmark

# 创建评估器
benchmark = NeedleInHaystackBenchmark(
    model=model,
    tokenizer=tokenizer,
    context_lengths=[1024, 4096, 16384],
    depths_per_length=10
)

# 运行评估
results = benchmark.run()
benchmark.plot_results(results, output_path='needle_results.png')
```

### 3. MATH 数据集评估

```bash
# 运行 MATH 评估
python scripts/evaluation/run_benchmarks.py \
    --benchmark math \
    --model-path ./outputs/my_model \
    --difficulty-levels 1 2 3 4 5
```

### 4. FLOP 效率分析

```python
from src.benchmarks import FLOPAnalyzer

analyzer = FLOPAnalyzer(model)

# 分析不同序列长度
for seq_len in [512, 1024, 2048, 4096]:
    stats = analyzer.analyze(seq_len=seq_len)
    print(f"序列长度 {seq_len}:")
    print(f"  FLOPs: {stats['total_flops'] / 1e9:.2f} GFLOPs")
    print(f"  内存: {stats['memory_mb']:.2f} MB")
    print(f"  参数利用率: {stats['param_utilization']:.2%}")
```

---

## 完整评估流程

### 步骤 1: 准备评估脚本

```python
#!/usr/bin/env python3
"""
模型评估脚本
"""

import argparse
import torch
from pathlib import Path

from src.models import AdaptiveTransformer
from src.benchmarks import (
    NeedleInHaystackBenchmark,
    MATHBenchmark,
    FLOPAnalyzer
)

def evaluate_perplexity(model, dataloader, device):
    """评估困惑度"""
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item()
            count += 1

    avg_loss = total_loss / count
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def evaluate_needle(model, tokenizer, device):
    """评估长上下文能力"""
    benchmark = NeedleInHaystackBenchmark(
        model=model,
        tokenizer=tokenizer,
        context_lengths=[1024, 4096, 16384, 32768],
        depths_per_length=10,
        num_trials=5
    )

    results = benchmark.run()

    # 计算平均准确率
    avg_accuracy = sum(r['accuracy'] for r in results) / len(results)

    return {
        'accuracy': avg_accuracy,
        'detailed_results': results
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--eval-data', required=True)
    parser.add_argument('--output-dir', default='./eval_results')
    args = parser.parse_args()

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    model = AdaptiveTransformer(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("开始评估...")

    # 1. 困惑度
    # ppl = evaluate_perplexity(model, dataloader, device)
    # print(f"困惑度: {ppl:.2f}")

    # 2. Needle-in-Haystack
    # needle_results = evaluate_needle(model, tokenizer, device)
    # print(f"Needle 准确率: {needle_results['accuracy']:.2%}")

    # 3. FLOP 分析
    analyzer = FLOPAnalyzer(model)
    for seq_len in [1024, 2048, 4096]:
        stats = analyzer.analyze(seq_len=seq_len)
        print(f"序列长度 {seq_len}: {stats['total_flops']/1e9:.2f} GFLOPs")

    print("评估完成!")

if __name__ == '__main__':
    main()
```

### 步骤 2: 运行评估

```bash
# 基础评估
python evaluate_model.py \
    --model-path ./outputs/my_model/final_model.pt \
    --eval-data ./data/validation \
    --output-dir ./eval_results

# 使用 Makefile
make evaluate MODEL_PATH=./outputs/my_model/final_model.pt
```

---

## 对比评估

### 与基线模型对比

```python
import pandas as pd

# 收集结果
results = {
    'Model': ['Baseline', 'ADN-Small', 'ADN-Medium'],
    'Perplexity': [12.5, 11.8, 10.2],
    'Needle@32K': [0.45, 0.72, 0.85],
    'FLOPs (G)': [150, 120, 200]
}

df = pd.DataFrame(results)
print(df.to_string(index=False))
```

### 生成对比报告

```bash
# 生成详细评估报告
python scripts/evaluation/generate_report.py \
    --results-dir ./eval_results \
    --output ./eval_report.md
```

---

## 持续评估

### 集成到训练流程

```python
# 在训练循环中定期评估
if step % eval_every == 0:
    eval_results = evaluate(model, eval_dataloader)

    # 记录到 tensorboard
    writer.add_scalar('Eval/Perplexity', eval_results['perplexity'], step)
    writer.add_scalar('Eval/Accuracy', eval_results['accuracy'], step)

    # 保存最佳模型
    if eval_results['perplexity'] < best_perplexity:
        best_perplexity = eval_results['perplexity']
        torch.save(model.state_dict(), 'best_model.pt')
```

---

## 参考

- [评估脚本](../../scripts/evaluation/)
- [基准测试](../../src/benchmarks/)
- [实验框架](../../experiments/)
