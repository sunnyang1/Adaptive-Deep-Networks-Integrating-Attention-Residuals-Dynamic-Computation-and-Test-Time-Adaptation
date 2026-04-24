# TurboQuant → RaBitQ 迁移指南

> **项目**：Adaptive Deep Networks
> **日期**：2026-04-23
> **状态**：部分完成（`src/rabitq/` 已创建，引用未全面迁移）

---

## 一、迁移现状

| 维度 | 状态 |
|------|------|
| `src/rabitq/` 核心模块 | ✅ 已创建 |
| `src/turboquant/` 旧代码 | ⚠️ 仍存在（需保留或删除） |
| `src/gating/depth_priority.py` | ❌ 注释和参数仍引用 TurboQuant |
| `src/qttt/polar_adaptation.py` | ❌ 注释和参数仍引用 TurboQuant |
| 测试文件 | ❌ legacy 测试仍存在，新测试未创建 |
| 脚本文件 | ❌ 大量 legacy 脚本未清理 |
| 文档 | ❌ ~15 个 md 文件仍有 TurboQuant 引用 |
| 配置文件 | ❌ pyproject.toml 等未更新 |

**残留引用总数**：约 107 处（分布在 60+ 文件中）

---

## 二、迁移分步执行计划

### 第 1 步：清理 `src/turboquant/` 旧模块

**操作**：删除 `src/turboquant/` 整个目录（`src/rabitq/` 已包含完整实现）

```bash
# 方式1：直接删除（旧代码已在 archive/ 中备份）
rm -rf src/turboquant/

# 方式2：如果想保留 git 历史
git rm -r src/turboquant/
```

**⚠️ 注意**：先确认 `archive/src_legacy/turboquant/` 中有完整备份。

**验证**：
```bash
ls src/turboquant/  # 应报错：No such file or directory
```

---

### 第 2 步：更新 `src/gating/depth_priority.py`

**目标**：将所有 TurboQuant 引用替换为 RaBitQ

**具体修改**：

1. 文件头注释：
   - `"Section 4.2 of Adaptive Deep Networks RaBitQ version"` → 保持
   - `"TurboQuant acceleration"` → `"RaBitQ compression"`

2. 类文档字符串（第 24-27 行）：
   ```python
   # 修改前
   Under TurboQuant acceleration:
   - C_qTTT^Turbo ≈ (1/8) * C_qTTT^Standard
   - T_think ≈ 16 * N_qTTT * k (vs 2 * N_qTTT * k without TurboQuant)

   # 修改后
   Under RaBitQ compression:
   - C_qTTT^RaBitQ ≈ (1/8) * C_qTTT^Standard
   - T_think ≈ 16 * N_qTTT * k (vs 2 * N_qTTT * k without RaBitQ)
   ```

3. 搜索所有 `turboquant`/`TurboQuant` 出现处，逐一替换：
   ```bash
   grep -n -i "turboquant" src/gating/depth_priority.py
   ```

4. 参数名替换（如有）：
   - `turboquant_enabled` → `rabitq_enabled`
   - `turboquant_bits` → `rabitq_bits`

---

### 第 3 步：更新 `src/qttt/polar_adaptation.py`

**目标**：将所有 TurboQuant 引用替换为 RaBitQ

**具体修改**：

1. 文件头注释（第 4 行）：
   ```python
   # 修改前
   Based on: Section 3.3 of Adaptive Deep Networks TurboQuant version

   # 修改后
   Based on: Section 3.3 of Adaptive Deep Networks
   ```

2. 第 10 行：
   ```python
   # 修改前
   4. Integration with TurboQuant for 8× cost reduction

   # 修改后
   4. Integration with RaBitQ for 4-bit KV-cache compression
   ```

3. 搜索全部替换：
   ```bash
   grep -n -i "turboquant" src/qttt/polar_adaptation.py
   ```

4. 参数名替换（在 `PolarQTTTConfig` 和相关方法中）：
   - `use_turboquant` → `use_rabitq`
   - `turboquant_bits` → `rabitq_bits`

---

### 第 4 步：更新其他源码引用

**文件清单**（按优先级排序）：

```bash
# 1. 测试配置
# tests/conftest.py — 1处引用

# 2. 实验模块
# experiments/real_model/model_loader.py — 4处引用
# experiments/real_model/validator.py — 1处引用

# 3. 脚本
# scripts/validate_rabitq.py — 7处引用（已部分更新，检查遗漏）
# scripts/experiments/paper_metrics_summary.py — 16处引用
# scripts/experiments/test_small_model_datasets.py — 8处引用
```

**批量搜索命令**（排除已归档和 legacy 目录）：
```bash
grep -r -l -i "turboquant" --include="*.py" \
  src/ tests/ scripts/experiments/ experiments/ \
  | grep -v legacy | grep -v __pycache__
```

对每个文件执行：
```bash
# 预览替换
sed -i '' 's/TurboQuant/RaBitQ/g; s/turboquant/rabitq/g; s/TURBOQUANT/RABITQ/g' <文件路径>
```

---

### 第 5 步：清理 legacy 测试

**操作**：确认 legacy 目录中的 TurboQuant 测试已被忽略

**需确认的文件**：
- `tests/legacy/test_mnn_turboquant.py`
- `tests/legacy/test_polar_components.py`
- `tests/legacy/test_turboquant.py`
- `tests/legacy/test_turboquant_core.py`
- `scripts/legacy/test_turboquant_small.py`
- `scripts/legacy/mnn_turboquant_demo.py`
- `scripts/legacy/turboquant_refactored_demo.py`
- `scripts/experiments/legacy/validate_turboquant_setup.py`
- `scripts/experiments/legacy/test_turboquant_v3_improved.py`
- `scripts/experiments/legacy/test_turboquant_on_small_model.py`
- `scripts/experiments/legacy/turboquant_v3_demo.py`
- `experiments/validation/legacy/turboquant_compression.py`

**验证**：这些文件已在 `tests/conftest.py` 的 `collect_ignore` 中排除。

```bash
# 运行测试确认 legacy 不被收集
pytest tests/ --collect-only 2>&1 | grep -i "turboquant"
# 预期：无结果
```

---

### 第 6 步：创建 RaBitQ 测试

**操作**：基于现有 `src/rabitq/` 创建对应测试文件

**需创建的测试**：
```
tests/unit/test_rabitq.py          # RaBitQ 核心功能测试
tests/unit/test_rabitq_cache.py    # RaBitQCache 测试
tests/unit/test_rabitq_estimator.py # MSECompressor + estimator 测试
```

**测试要点**：
```python
# test_rabitq.py 必须覆盖
1. RaBitQ.compress() / decompress() 往返一致性
2. RaBitQConfig 参数验证（bits=2/4/8）
3. RaBitQCache 正确的 KV cache 压缩/解压
4. MSECompressor.fit() + compress() 数值精度（< 1e-3 误差）
5. bit-packing / unpacking 位操作正确性
```

**运行测试**：
```bash
pytest tests/unit/test_rabitq*.py -v
```

---

### 第 7 步：更新文档（Markdown）

**按优先级分批处理**：

#### 7a. 高优先级（项目文档）

| 文件 | 操作 |
|------|------|
| `AGENTS.md` | 替换所有 TurboQuant → RaBitQ 引用 |
| `PROJECT_ORGANIZATION.md` | 约 30 处引用，批量替换 |
| `docs/ARCHITECTURE.md` | 9 处引用，更新架构描述 |
| `docs/README.md` | 2 处引用 |

#### 7b. 中优先级（技术文档）

| 文件 | 操作 |
|------|------|
| `docs/guides/TURBOQUANT_REFACTORED.md` | 重命名为 `RABITQ_GUIDE.md`，全面改写 |
| `docs/guides/TURBOQUANT_V3.md` | 重命名为 `RABITQ_ALGORITHM.md`，改写 |
| `docs/guides/MNN_TURBOQUANT_IMPROVEMENTS.md` | 重命名为 `RABITQ_IMPROVEMENTS.md` |
| `docs/reports/implementation/TURBOQUANT_IMPLEMENTATION_SUMMARY.md` | 重命名为 `RABITQ_IMPLEMENTATION_SUMMARY.md` |
| `docs/project/reports/TURBOQUANT_V3_SUMMARY.md` | **删除**（已过时） |
| `docs/project/IMPLEMENTATION_AUDIT.md` | 5 处引用更新 |
| `experiments/docs/TURBOQUANT_EXPERIMENTS.md` | 重命名为 `RABITQ_EXPERIMENTS.md` |
| `experiments/README.md` | 6 处引用更新 |

#### 7c. 低优先级（归档和报告）

| 文件 | 操作 |
|------|------|
| `archive/` 下的所有文件 | **不修改**（归档保留历史原貌） |
| `docs/reports/research/*.md` | **不修改**（研究报告保留原始引用） |
| `docs/papers/Adaptive_Deep_Networks_TurboQuant.md` | 重命名为 `Adaptive_Deep_Networks_RaBitQ.md` |

**批量搜索命令**：
```bash
# 搜索非 archive 目录下的残留引用
grep -r -l -i "turboquant" --include="*.md" \
  | grep -v archive | grep -v node_modules
```

---

### 第 8 步：更新配置文件

**8a. `pyproject.toml`**
```bash
grep -n -i "turboquant" pyproject.toml
# 替换所有引用
```

**8b. `configs/experiments/validation_targets.yaml`**
```bash
grep -n -i "turboquant" configs/experiments/
# 更新实验配置中的模块名
```

**8c. `experiments/Makefile`**
```bash
grep -n -i "turboquant" experiments/Makefile
# 更新 make target 名称
```

**8d. `experiments/turboquant/` 目录重命名**
```bash
git mv experiments/turboquant/ experiments/rabitq/
# 更新目录内的 README.md
```

---

### 第 9 步：更新论文相关文件

| 文件 | 操作 |
|------|------|
| `QASP_paper_cn.md` | 3 处 TurboQuant → RaBitQ |
| `QASP_paper.aux` | LaTeX 辅助文件，重新编译即可 |

**不修改**：
- `ADN_paper.md`（plan 已确认无需修改）
- `matdo-e_paper.md`（plan 已确认无需修改）

---

### 第 10 步：全局验证

**10a. 残留引用扫描**

```bash
# 搜索所有非 archive 目录的残留
grep -r -i "turboquant" --include="*.py" --include="*.md" --include="*.yaml" --include="*.toml" \
  src/ tests/ scripts/ experiments/ docs/ configs/ \
  | grep -v archive | grep -v __pycache__ | grep -v legacy
```

**预期结果**：无输出（0 匹配）

**10b. 导入测试**

```python
# 验证 RaBitQ 可正常导入
python3 -c "from src.rabitq import RaBitQ, RaBitQConfig, RaBitQCache; print('OK')"
```

**10c. 全量测试**

```bash
pytest tests/ -v --ignore=tests/legacy --tb=short
```

**10d. Lint 检查**

```bash
make lint
# 或手动
ruff check src/rabitq/ src/gating/ src/qttt/
mypy src/rabitq/
```

---

### 第 11 步：提交变更

```bash
# 查看所有变更
git status

# 分阶段提交
git add src/rabitq/ src/gating/ src/qttt/
git commit -m "refactor: migrate TurboQuant references to RaBitQ in core modules"

git add tests/unit/test_rabitq*.py
git commit -m "test: add RaBitQ unit tests"

git add AGENTS.md PROJECT_ORGANIZATION.md docs/
git commit -m "docs: update all documentation from TurboQuant to RaBitQ"

git add pyproject.toml configs/ experiments/
git commit -m "chore: update config files for RaBitQ migration"
```

---

## 三、风险与注意事项

### ⚠️ 高风险

1. **`polar_adaptation.py` 参数名变更**可能影响已有实验配置和 checkpoint 兼容性
   - **缓解**：在 `PolarQTTTConfig` 中保留旧参数名的 `@deprecated` 别名

2. **`depth_priority.py` 参数变更**影响 gating 策略
   - **缓解**：先在测试中验证行为一致性

### ⚡ 中风险

3. **文档重命名**可能导致外部链接失效
   - **缓解**：在 README.md 中添加重定向说明

4. **legacy 测试删除**可能丢失有价值的测试用例
   - **缓解**：迁移有价值的测试逻辑到新测试文件

### 🟢 低风险

5. **archive/ 目录**保持不变，无需处理
6. **注释中的引用**不影响功能，可延后处理

---

## 四、文件影响统计

| 类别 | 文件数 | 操作类型 |
|------|--------|----------|
| 源码 `src/` | ~10 | 修改 |
| 测试 `tests/` | ~6 | 新建 + 删除 legacy |
| 脚本 `scripts/` | ~10 | 修改 + 清理 legacy |
| 文档 `docs/` | ~15 | 重命名 + 修改 |
| 配置 `configs/` | ~3 | 修改 |
| 实验配置 `experiments/` | ~5 | 重命名 + 修改 |
| 论文 | ~2 | 修改 |
| **总计** | **~51** | — |

---

## 五、执行时间估算

| 步骤 | 预估时间 | 并行可能性 |
|------|----------|------------|
| 第 1 步：删除旧模块 | 2 分钟 | — |
| 第 2-3 步：更新核心源码 | 15 分钟 | ✅ 可并行 |
| 第 4 步：更新其他源码 | 20 分钟 | ✅ 可并行 |
| 第 5 步：清理 legacy | 5 分钟 | — |
| 第 6 步：创建测试 | 30 分钟 | 与文档并行 |
| 第 7 步：更新文档 | 25 分钟 | 与测试并行 |
| 第 8 步：更新配置 | 10 分钟 | — |
| 第 9 步：更新论文 | 5 分钟 | — |
| 第 10 步：全局验证 | 15 分钟 | — |
| 第 11 步：提交 | 5 分钟 | — |
| **总计** | **~2 小时** | **实际 ~1.5 小时** |

---

## 六、回滚方案

如果迁移过程中出现严重问题：

```bash
# 回滚所有未提交的变更
git checkout -- .

# 如果已提交，回退到迁移前的 commit
git log --oneline  # 找到迁移前的 commit hash
git revert <commit-hash>

# 或硬回退（仅在确认无误时使用）
git reset --hard <commit-hash>
```

`archive/` 目录保留了完整的 TurboQuant 历史代码，可作为参考。

---

## 七、完成后检查清单

- [ ] `grep -r "turboquant" src/ tests/ scripts/ experiments/ docs/ configs/ | grep -v archive | grep -v legacy` 无输出
- [ ] `python3 -c "from src.rabitq import RaBitQ, RaBitQConfig"` 成功
- [ ] `pytest tests/unit/test_rabitq*.py -v` 全部通过
- [ ] `pytest tests/ --ignore=tests/legacy -v` 全部通过
- [ ] `make lint` 无新增错误
- [ ] `AGENTS.md` 中无 TurboQuant 残留
- [ ] `PROJECT_ORGANIZATION.md` 中无 TurboQuant 残留
- [ ] 所有文档重命名完成，旧文件已删除
- [ ] Git 提交历史清晰，按模块分批提交
