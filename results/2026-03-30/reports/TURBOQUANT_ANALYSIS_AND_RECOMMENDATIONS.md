# TurboQuant Small Model Test Analysis

## 执行摘要

本报告基于 Small Model (2.2B 参数) 的 TurboQuant 组件测试结果，分析当前实现与论文目标的差距，并提供改进建议。

**测试时间**: 2026-03-30  
**硬件**: Apple Silicon (MPS)  
**模型配置**: 32 layers, 2048 hidden dim, 32 heads, head_dim=64

---

## 1. 测试结果汇总

### 1.1 压缩率测试结果

| 组件 | 当前实现 | 论文目标 | 状态 |
|------|----------|----------|------|
| PolarQuant (3-bit) | 5.0× | - | ✅ 良好 |
| Full TurboQuant | 3.81× | 6×+ | ⚠️ 低于目标 |
| KV Cache 减少 | 3.81× | 5.7× | ⚠️ 低于目标 |

### 1.2 准确性测试结果

| 指标 | 当前值 | 目标 | 状态 |
|------|--------|------|------|
| 注意力相对误差 | 89.16% | < 5% | ❌ 需改进 |
| 余弦相似度 | 0.66 | > 0.95 | ❌ 需改进 |

### 1.3 PolarQuant 按位宽分析

| 角度位宽 | 压缩率 | 相对误差 | SNR |
|----------|--------|----------|-----|
| 2-bit | 7.21× | 88.20% | 1.1 dB |
| 3-bit | 5.00× | 52.18% | 5.6 dB |
| 4-bit | 3.82× | 34.09% | 9.3 dB |
| 5-bit | 3.82× | 23.84% | 12.5 dB |

**观察**: 即使 5-bit 配置，相对误差仍有 23.84%，远高于论文声称的 "zero accuracy loss"。

### 1.4 QJL 残差校正分析

| 投影维度 | 压缩率 | 相对误差 | 偏差 |
|----------|--------|----------|------|
| 32 | 32× | 97.54% | 0.15 |
| 64 | 16× | 98.23% | 0.12 |
| 128 | 8× | 98.47% | 0.11 |
| 256 | 4× | 99.03% | 0.11 |

**观察**: QJL 实现存在显著误差，且偏差不为零，表明实现可能有误。

---

## 2. 差距分析

### 2.1 压缩率差距原因

**论文配置推测**:
```
论文 TurboQuant (6×+ 压缩):
- PolarQuant: 4-bit 角度量化 (非 3-bit)
- QJL: 较小的投影维度 (~64)
- 总计: ~5-bit/元素

当前实现:
- PolarQuant: 3-bit 角度量化
- QJL: 64-bit 投影 (对于 head_dim=64)
- 总计: ~4-bit + 1-bit = 5-bit/元素，但实现效率较低
```

**head_dim 影响**:
- 论文可能基于更大的 head_dim (如 128 或 256)
- head_dim=64 时，固定开销 (radius + QJL) 占比更大

### 2.2 准确性差距原因

**PolarQuant 问题**:
1. Lloyd-Max 量化器可能未正确收敛
2. 角度分布假设 (Beta(2,2)) 可能与实际不符
3. Hadamard Transform 的实现可能有精度损失

**QJL 问题**:
1. 偏差不为零，表明无偏估计实现有误
2. 相对误差极高，投影矩阵可能未正确归一化

**关键公式检查**:
```python
# QJL 无偏估计器 (论文)
Prod_JL(q, k) = (π/2m) * ||k||_2 * <Sq, sign(Se)>

# 当前实现
estimated_dot = (math.pi / (2 * self.proj_dim)) * inner_product
# 缺少 ||k||_2 缩放!
```

---

## 3. 改进建议

### 3.1 立即修复

#### 修复 1: QJL 偏差校正
```python
def decompress_for_dot_product(self, signs, query, key_norm):
    """
    正确的无偏估计器实现
    """
    # Project query
    Sq = query @ self.S.T
    
    # Inner product with signs
    inner_product = (Sq * signs).sum(dim=-1)
    
    # Unbiased scaling WITH key norm
    estimated_dot = (math.pi / (2 * self.proj_dim)) * inner_product * key_norm
    
    return estimated_dot
```

#### 修复 2: 增加 PolarQuant 位宽
```python
# 当前
angle_bits = 3  # 3-bit 角度

# 建议测试
angle_bits = 4  # 4-bit 角度
# 或
angle_bits = 5  # 5-bit 角度
```

#### 修复 3: 优化 QJL 投影维度
```python
# 当前
qjl_proj_dim = 64  # 对于 head_dim=64

# 建议
qjl_proj_dim = 32  # 减少开销
```

### 3.2 算法改进

#### 改进 1: 自适应 Lloyd-Max
- 使用实际数据分布而非预设的 Beta(2,2)
- 在线更新量化中心点

#### 改进 2: 分块量化
- 对 head_dim=64，可考虑分块处理
- 每 32 维度一组，减少量化误差

#### 改进 3: 混合精度
- Key 使用更高精度 (5-bit)
- Value 使用较低精度 (3-bit)
- 论文表明 Value 对量化更鲁棒

### 3.3 配置优化

#### 配置 A: 平衡模式 (推荐)
```python
TurboQuantConfig(
    angle_bits=4,        # 4-bit PolarQuant
    qjl_proj_dim=32,     # 较小 QJL
)
# 预期压缩率: ~5×
# 预期误差: ~10-15%
```

#### 配置 B: 高精度模式
```python
TurboQuantConfig(
    angle_bits=5,        # 5-bit PolarQuant
    qjl_proj_dim=64,     # 标准 QJL
)
# 预期压缩率: ~4×
# 预期误差: ~5-10%
```

#### 配置 C: 高压缩模式
```python
TurboQuantConfig(
    angle_bits=3,        # 3-bit PolarQuant
    qjl_proj_dim=16,     # 最小 QJL
)
# 预期压缩率: ~6-7×
# 预期误差: ~20-30%
```

---

## 4. 论文 vs 实现对比

### 4.1 论文声明回顾

| 声明 | 论文值 | 当前实现 | 差距 |
|------|--------|----------|------|
| 内存减少 | 6×+ | 3.81× | -2.2× |
| KV Cache 减少 | 5.7× (16GB→2.8GB) | 3.81× | -1.9× |
| 零精度损失 | 是 | 否 | 需修复 |
| 8× 吞吐量提升 | 是 | 0.13× (更慢) | 无 Tensor Cores |

### 4.2 可能差异来源

1. **硬件差异**: 论文使用 NVIDIA H100 Tensor Cores，当前为 MPS
2. **模型尺寸**: 论文主要报告 Medium (8.7B) 和 Large (27B) 结果
3. **head_dim**: 更大模型的 head_dim 可能更大 (128 vs 64)
4. **校准数据**: 论文可能使用少量校准数据优化量化参数
5. **实现细节**: 某些算法细节可能未在论文中完全披露

---

## 5. 后续行动计划

### Phase 1: 修复关键 Bug (1-2 天)
- [ ] 修复 QJL 无偏估计器实现
- [ ] 验证 PolarQuant 角度量化分布
- [ ] 添加单元测试验证各组件

### Phase 2: 参数调优 (2-3 天)
- [ ] 扫描 angle_bits (2-6) 和 qjl_proj_dim (16-128)
- [ ] 建立压缩率-准确率 Pareto 前沿
- [ ] 确定最优配置

### Phase 3: 集成测试 (1-2 天)
- [ ] 端到端注意力测试
- [ ] 与 Small Model 集成
- [ ] 长序列 KV Cache 测试

### Phase 4: 文档与报告 (1 天)
- [ ] 更新实现文档
- [ ] 生成对比报告
- [ ] 提交改进 PR

---

## 6. 实验数据文件

所有原始数据已保存至:
- `results/turboquant_small_model_tests.json`
- `results/turboquant_small_model_report.txt`

---

## 7. 结论

当前 TurboQuant 实现展示了基本的量化功能，但距离论文声称的性能还有差距。主要问题集中在:

1. **QJL 实现 Bug**: 无偏估计器缺少关键缩放因子
2. **压缩率配置**: 需要调整 angle_bits 和 proj_dim 平衡
3. **硬件限制**: CPU/MPS 无法展示 Tensor Core 加速效果

建议优先修复 QJL Bug，然后进行参数调优以达到接近论文目标的性能。

---

*Report generated on 2026-03-30*
