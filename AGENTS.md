# Adaptive Deep Networks (ADN) — Agent Guide

## Project Overview

This is the research codebase for the paper *"Adaptive Deep Networks: Integrating Attention Residuals, Dynamic Computation, and Test-Time Adaptation"*. It trains, validates, and benchmarks a modular transformer architecture designed for efficient long-context inference.

The repository contains three paper-aligned modules:

1. **ADN Core** — `AttnRes` (Block Attention Residuals), `qTTT` (query-only test-time training), `RaBitQ` (KV-cache compression), `Engram` (n-gram memory), and dynamic gating.
2. **QASP** — Quality-Aware Stiefel Projection (`QASP/` and `adn/qasp/`).
3. **MATDO-E** — Unified Resource Model (`MATDO-new/` and `adn/matdo_e/`).

The codebase is pure Python/PyTorch. No Docker, databases, or external services are required.

## Technology Stack

- **Python**: >=3.12 (system default on the primary development environment; CI also tests 3.9–3.11).
- **PyTorch**: >=2.0.0 (CPU wheels used in CI; CUDA/MPS used locally depending on hardware).
- **Hugging Face**: `transformers>=4.35.0`, `datasets>=2.14.0`, `accelerate>=0.24.0`.
- **Scientific**: `numpy>=1.24.0`, `scipy>=1.11.0`, `pandas>=2.0.0`.
- **Visualization**: `matplotlib>=3.8.0`, `seaborn>=0.13.0`.
- **Logging/Tracking**: `tqdm>=4.66.0`, `wandb>=0.15.0`.
- **Optional GPU kernels**: `flash-attn>=2.3.0` and `triton>=2.1.0` (Linux only).

## Key Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Root package config (`adn` v0.2.0). Declares dependencies, optional dev extras, and tool configs for **black**, **ruff**, **mypy**, **pytest**, and **coverage**. |
| `MATDO-new/pyproject.toml` | Separate installable package (`matdo_new` v0.1.0) for the MATDO-E runtime. |
| `requirements.txt` | Runtime and development dependencies (duplicates much of `pyproject.toml` for convenience). |
| `Makefile` | High-level targets: `install`, `test`, `test-cov`, `lint`, `format`, `clean`, `quick`, `full`, `core`, `validate`, `paper-metrics`, and paper-aligned training wrappers. |
| `.github/workflows/pr.yml` | PR quick checks (format + fast unit tests on Python 3.10). |
| `.github/workflows/validate.yml` | Full validation: code quality (black, flake8, mypy) + unit tests across Python 3.9/3.10/3.11. |
| `.github/workflows/benchmark.yml` | Benchmark workflow. |
| `configs/experiments/*.yaml` | Experiment-specific configurations. |
| `configs/ds_config_h20.json` | DeepSpeed config for H20 GPUs. |

## Repository Layout

For an exhaustive directory map (in Chinese) see [`PROJECT_ORGANIZATION.md`](PROJECT_ORGANIZATION.md). The summary below focuses on what an agent needs to know.

### Source Code — Dual Tree Reality

**`src/` — Legacy but actively working source of truth**

This is the import path used by training scripts, most tests, and the existing CI. Key subpackages:

- `src/attnres/` — Block Attention Residuals (`block_attnres.py`, `pseudo_query.py`, `polar_pseudo_query.py`).
- `src/gating/` — Reconstruction loss, threshold calibration (`threshold.py`), ponder gate (`ponder_gate.py`), depth priority.
- `src/qttt/` — Query-only test-time training (`adaptation.py`, `margin_loss.py`, `batch_adaptation.py`, `polar_adaptation.py`, `adaptive_config.py`).
- `src/models/` — `AdaptiveTransformer`, `AdaptiveLayer`, `AdaptiveAttention`, `AdaptiveMLP`, `IncrementalState`, `IncrementalGenerator`, `IncrementalKVCache`, and `configs.py` (`ModelConfig`, `AttnResSmallConfig`, `AttnResMediumConfig`, `AttnResLargeConfig`, `AttnResT4Config`).
- `src/rabitq/` — RaBitQ KV-cache compression (`api.py`, `quantizer.py`, `rotation.py`, `packing.py`, `cache.py`, `estimator.py`) plus a `legacy/` subdirectory.
- `src/rabitq/` — RaBitQ compression (`api.py`, `quantizer.py`, `rotation.py`, `compressor.py`, `cache.py`) plus a `legacy/` subdirectory.
- `src/engram/` — Engram n-gram memory (`engram_module.py`, `embeddings.py`, `ngram_hash.py`, `compressed_tokenizer.py`, `integration.py`, `config.py`).
- `src/benchmarks/` — Needle-in-Haystack, MATH eval, FLOP analysis.

**`adn/` — New unified package**

This package is intended to become the canonical surface and is now importable thanks to the `adn/core/` modules (`config.py`, `base.py`, `types.py`).

- `adn/core/` — Base configurations (`ModelConfig`, `ADNConfig`) and components (`RMSNorm`, `SwiGLU`, `BaseModule`).
- `adn/attention/` — Mirrors `src/attnres/`.
- `adn/models/` — Mirrors `src/models/`.
- `adn/qttt/`, `adn/gating/`, `adn/memory/`, `adn/quantization/`, `adn/qasp/`, `adn/matdo_e/`, `adn/experiments/`, `adn/utils/`.

### Standalone Paper Packages

- **`QASP/`** — Self-contained QASP package with its own `models/`, `adaptation/`, `inference/`, `configs/`, `scripts/`, and `experiments/`. It is importable as `QASP` when the repo root is on `PYTHONPATH`. See `QASP/__init__.py` for public exports and the semantic distinction between `forward`/`prefill` (canonical full-sequence) and `step` (prefix-growing incremental).
- **`MATDO-new/`** — Installable via `pip install -e MATDO-new`. Provides `matdo_new.apps.generate` and `matdo_new.apps.run_experiments` CLI entry points. It is intentionally separate from the legacy `experiments/matdo/` orchestration.

### Scripts, Experiments, and Tests

- **`scripts/training/`** — Canonical training entrypoint is `scripts/training/train_model.py`, which dispatches through `BaseTrainer`. Size-specific wrappers exist (`train_small.py`, `train_medium.py`, `train_large.py`, `train_t4.py`, `train_h20.py`, `train_streaming.py`, `train_unified.py`).
- **`scripts/evaluation/`** — `run_benchmarks.py`, `validate_models.py`, etc.
- **`scripts/setup/`** — Environment setup scripts (`a100_setup.sh`, `check_env.py`, `lambda_setup.sh`, etc.).
- **`experiments/`** — Experiment frameworks and runners.
  - `experiments/run_experiments_unified.py` — Unified runner for core, validation, and paper-metric experiments.
  - `experiments/core/` — Representation burial, margin analysis, gradient flow, FLOP equivalence, synergy, auxiliary head, Table3/Table5/Table8 experiments.
  - `experiments/validation/` — Table1–Table9 validation scripts.
  - `experiments/matdo/` — Legacy MATDO experiment orchestration (still used for real-model US1–US6 workflows).
  - `experiments/real_model/` — Real-model needle-in-haystack and validation.
- **`tests/`** — Test suite.
  - `tests/unit/` — 25+ unit test files (~4,500 lines).
  - `tests/integration/` — 15+ integration test files (~1,700 lines).
  - `tests/e2e/` — End-to-end smoke tests.
  - `tests/benchmark/` — Performance benchmarks.
  - `tests/legacy/` — Deprecated tests that fail on import; **always ignored** by default (`collect_ignore = ["legacy"]` in `conftest.py`).

### Submission Manuscripts (keep at root)

Do **not** move these into `docs/` or elsewhere:

- `ADN_paper.md`
- `matdo-e_paper.md`

## Build, Install, and Common Commands

Install the root package and dev tools:

```bash
pip install -e ".[dev]"
# Or additionally:
pip install tqdm matplotlib seaborn pandas scipy
```

Install MATDO-new separately when needed:

```bash
pip install -e MATDO-new
```

Run tests:

```bash
# Fast unit tests only
pytest tests/unit/ -v --tb=short

# All tests (skip legacy)
pytest tests/ -v --tb=short --ignore=tests/legacy

# With coverage
pytest tests/ -v --tb=short --ignore=tests/legacy --cov=src --cov-report=html --cov-report=term
```

Lint and format:

```bash
make lint      # black --check + ruff check + mypy src/
make format    # black src/ experiments/ scripts/ tests/
```

Experiments:

```bash
make quick              # quick mode across all categories
make full               # full experiment suite
make core               # core experiments only
make validate           # validation experiments
make paper-metrics      # paper-facing metrics
make list               # list available experiments
```

Training:

```bash
python3 scripts/training/train_model.py --model-size small --output-dir results/small
```

Use **`python3`** (not `python`) on this repo.

## Code Style and Conventions

- **Formatter**: `black` with `line-length = 100`, `target-version = ['py312']`.
- **Linter**: `ruff` (pycodestyle, Pyflakes, isort, pep8-naming, pyupgrade, bugbear, comprehensions, simplify). `E501` is ignored (handled by black). `N806`/`N803` are ignored for scientific/ML naming conventions.
- **Type checker**: `mypy` with `python_version = "3.12"`, `ignore_missing_imports = true`, gradual adoption (`disallow_untyped_defs = false`).
- **Test naming**: `test_*.py` files, `Test*` classes, `test_*` functions.
- **Docstrings**: Modules and public classes/functions carry extensive Google-style docstrings with `>>>` examples. Chinese comments appear in some `adn/` modules and in `PROJECT_ORGANIZATION.md`.
- **Import style**: Mixed. Legacy code uses `from src.attnres...` and `from src.models...` after inserting the repo root into `sys.path`. Newer `adn/` code uses top-level `adn.*` imports and is now functional with the addition of `adn/core/`.

## Architecture and Key Design Decisions

### AttnRes (`src/attnres/`)

- **Zero Initialization**: Pseudo-queries initialize to zero for training stability (§5.3).
- **Block Structure**: Reduces memory from O(Ld) to O(Nd).
- **Two-Phase**: Phase 1 (parallel inter-block) + Phase 2 (sequential intra-block).
- **Optimal Block Count**: N≈8 recovers most FullAttnRes benefit (§5.3 Fig 6).
- **RMSNorm on Keys**: Critical for performance (without: +0.006/+0.004 loss).
- **Single-Head Depth Attention**: Multi-head hurts performance (1.752 vs 1.746).
- **Softmax over Sigmoid**: Competitive normalization forces sharper selection.

### Gating (`src/gating/`)

- **Signal**: Reconstruction loss as a difficulty proxy.
- **Calibration**: EMA or target-rate threshold.
- **Ponder Gate**: Conditional qTTT triggering based on uncertainty (entropy + max-probability heuristics). Modes: `strict`, `balanced`, `lenient`. Integrated into `generate(use_qttt='adaptive')`.
- **Adaptive qTTT Config**: Dynamic adjustment of steps and LR. Modes: `fast`, `balanced`, `quality`.

### qTTT (`src/qttt/`)

- **Frozen KV**: Keys and values from prefill never change.
- **Query-only**: Only query parameters are updated.
- **Margin Loss**: Explicit logit margin maximization.

### Incremental KV Cache (`src/models/incremental_*.py`)

- **IncrementalState**: Explicit state management for generation (KV cache per layer, block representation management, memory stats).
- **IncrementalGenerator**: High-level API with `prefill()` for O(T×L) initialization and `step()` for O(L) per-token generation.
- **Target**: Maintain ~30% adaptation rate.

### QASP (`QASP/`)

- **Canonical semantics**: Value-weighted AttnRes and block statistics are defined on a **single full-sequence forward** over a fixed context. Use `QASPTransformer.forward` or `prefill` for that definition.
- **Incremental `step`**: With `use_attnres=True`, block summaries use a **growing prefix**; logits need not match `forward` on the extended sequence. Documented in `QASP/__init__.py`, `QASP/models/qasp_transformer.py`, and `tests/integration/test_qasp_prefill_step_numeric_parity.py`.

### Model Configurations (from `src/models/configs.py`)

| Size   | Params | Layers | Hidden | Heads | Blocks | d_model/L_b | H/L_b |
| ------ | ------ | ------ | ------ | ----- | ------ | ----------- | ----- |
| Small  | 1.1B   | 32     | 1408   | 8     | 8      | 44.0        | 0.25  |
| Medium | 5.7B   | 56     | 2496   | 16    | 8      | 44.6        | 0.29  |
| Large  | 23.0B  | 88     | 4032   | 18    | 11     | 45.8        | 0.21  |

> **Architecture Optimized for AttnRes (§5.4.1)**: Paper shows AttnRes shifts optimal `d_model/L_b` from ~60 (baseline) to ~45 (AttnRes), favoring deeper, narrower networks. Head ratio `H/L_b ≈ 0.3` is optimal for both.

## Testing Strategy

- **Framework**: `pytest` with markers: `slow`, `gpu`, `unit`, `integration`.
- **Fixtures**: `conftest.py` provides `device`, `rng`, `torch_rng`, `sample_tensor`, `model_config_small`, `model_config_medium`, and an autouse `reset_torch_seed` fixture.
- **PYTHONPATH**: `conftest.py` injects the repo root so `import src.*` works reliably.
- **Legacy exclusion**: `tests/legacy/` is ignored by default via `collect_ignore = ["legacy"]` because those tests target removed/renamed APIs (old RaBitQ V1/V2, MNNRaBitQ).
- **CI**: PR workflow runs fast unit tests on Python 3.10. Validation workflow runs code-quality checks plus unit tests on Python 3.9, 3.10, and 3.11.

## Deployment and Environment Notes

- **No virtual environment required** on the primary macOS dev machine; packages install to `~/.local` and `~/.local/bin` is on `PATH`.
- **Hugging Face mirrors**: If `huggingface.co` fails, set `export HF_ENDPOINT=https://hf-mirror.com`.
- **Offline machines**: Must rely on a populated Hugging Face Hub cache or provide local dataset paths.
- **Lambda AI / Cloud**: Use `scripts/setup/lambda_setup.sh` for cloud instance setup.

## Known Issues (Pre-existing)

1. **`tests/legacy/` deprecated** — Always pass `--ignore=tests/legacy` to pytest; those tests fail on import against removed APIs.
2. **Mypy pre-existing errors** — `mypy src/` may report `Source file found twice under different module names` and other issues. These are not environment-specific.
3. **Unified experiment runner CLI mismatch** — `run_experiments_unified.py` passes `--output-dir` (hyphenated) to sub-scripts, but some individual experiment scripts expect `--output_dir` (underscored), causing exit code 2. Running experiments individually with `--output_dir` works.
4. **Use `python3` not `python`** — The latter is not linked by default.
5. **Unit test failures** — A subset of unit tests fail due to pre-existing code/test mismatches (e.g., gating `DynamicThreshold` API changes, `TwoPhaseBlockAttnRes.norm` missing, `compute_reconstruction_loss` shape bugs). These are documented in the codebase and are not caused by the environment.

## Security Considerations

- No secrets, API keys, or credentials are stored in the repository.
- `.gitignore` excludes model checkpoints (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`), datasets (`data/`, `datasets/`), and cloud credentials (`*.pem`, `*.key`, `.aws/`, `.gcp/`, `.azure/`).
- This is a research codebase; do not run untrusted model checkpoints in production environments.

## References

- Paper drafts: `ADN_paper.md` (root), `matdo-e_paper.md` (root).
- Detailed layout (Chinese): `PROJECT_ORGANIZATION.md`.
- Architecture diagrams: `docs/ARCHITECTURE.md`.
- QASP manuscript: `QASP_paper.tex` / `QASP_paper_cn.md`.
