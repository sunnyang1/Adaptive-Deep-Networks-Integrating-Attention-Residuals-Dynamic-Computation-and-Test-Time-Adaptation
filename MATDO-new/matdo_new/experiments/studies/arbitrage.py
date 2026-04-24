"""Heterogeneous Resource Arbitrage Study — MATDO-E §4 Validation.

Validates:
  * Theorem 4.1 (Heterogeneous Arbitrage Inequality): Engram postpones the
    context wall when  ζ > η / (E_max · ε_target).
  * Theorem 4.2 (Optimality under convex relaxation): with E > 0 the joint
    (R, M, T, E) solution Pareto-dominates the E = 0 baseline.
  * Cross-architecture robustness: the inequality is architecture-agnostic
    and depends only on (ζ, η, E_max, ε_target).

The study simulates three architecture profiles matching the paper's
Table 1 (LLaMA-2-7B MHA, Mistral-7B sliding-window GQA, Qwen-2-7B GQA)
by varying the model-level error-model coefficients while keeping the
hardware budget fixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from matdo_new.core.config import MATDOConfig
from matdo_new.core.resource_theory import m_min_closed_form, rho_context_wall
from matdo_new.experiments.baselines import ExperimentResult

# ---------------------------------------------------------------------------
# Architecture profiles (Table 1 style, §5.1)
# ---------------------------------------------------------------------------

#: Pre-calibrated coefficient sets that reproduce the paper's wall positions.
#: Differences reflect MHA vs. sliding-window GQA vs. GQA attention patterns.
ARCHITECTURE_PROFILES: dict[str, dict[str, float]] = {
    "llama2-7b-mha": {
        # Standard MHA: moderate β, low δ
        "alpha": 0.015,
        "beta": 2.0,
        "gamma": 0.10,
        "delta": 0.005,
        "epsilon": 0.002,
        "zeta": 0.35,
        "eta": 0.5,
    },
    "mistral-7b-swgqa": {
        # Sliding-window GQA: higher scope sensitivity (β), lower γ
        "alpha": 0.012,
        "beta": 2.4,
        "gamma": 0.09,
        "delta": 0.006,
        "epsilon": 0.0025,
        "zeta": 0.32,
        "eta": 0.48,
    },
    "qwen2-7b-gqa": {
        # GQA: better quantisation robustness (lower α), slightly higher γ
        "alpha": 0.010,
        "beta": 1.8,
        "gamma": 0.11,
        "delta": 0.004,
        "epsilon": 0.0018,
        "zeta": 0.38,
        "eta": 0.52,
    },
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArbitragePoint:
    """Arbitrage analysis at a single (architecture, E_max) configuration."""

    arch_name: str
    e_max: int
    target_error: float

    # Arithmetic checks
    inequality_lhs: float  # ζ
    inequality_rhs: float  # η / (E_max · ε_target)
    inequality_holds: bool

    # Wall positions
    rho_ctx_baseline: float  # without Engram (E=0)
    rho_ctx_engram: float  # with Engram (E=E_max)
    wall_postponed: bool  # rho_ctx_engram > rho_ctx_baseline

    # Quantified improvement
    wall_shift: float  # rho_ctx_engram - rho_ctx_baseline (should be >0)
    m_min_baseline: float
    m_min_engram: float


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_arbitrage(
    arch_name: str,
    config: MATDOConfig,
    *,
    r_bits: int = 2,
) -> ArbitragePoint:
    """Compute the arbitrage analysis for one architecture config."""
    target = config.target_error

    # Arithmetic inequality check (Eq. 5)
    lhs = config.zeta
    rhs = config.eta / (max(config.e_max, 1) * target) if target > 0 else math.inf
    ineq_holds = lhs > rhs

    # Baseline wall (E=0)
    m_min_base = m_min_closed_form(
        r_bits=r_bits,
        target_error=target,
        config=config,
        engram_entries=0,
    )
    rho_ctx_base = rho_context_wall(r_bits=r_bits, m_min=m_min_base, config=config)

    # Engram-assisted wall (E=E_max)
    m_min_eng = m_min_closed_form(
        r_bits=r_bits,
        target_error=target,
        config=config,
        engram_entries=config.e_max,
    )
    rho_ctx_eng = rho_context_wall(r_bits=r_bits, m_min=m_min_eng, config=config)

    wall_postponed = rho_ctx_eng > rho_ctx_base

    return ArbitragePoint(
        arch_name=arch_name,
        e_max=config.e_max,
        target_error=target,
        inequality_lhs=lhs,
        inequality_rhs=rhs,
        inequality_holds=ineq_holds,
        rho_ctx_baseline=rho_ctx_base,
        rho_ctx_engram=rho_ctx_eng,
        wall_postponed=wall_postponed,
        wall_shift=rho_ctx_eng - rho_ctx_base,
        m_min_baseline=m_min_base,
        m_min_engram=m_min_eng,
    )


# ---------------------------------------------------------------------------
# Engram E_max sensitivity sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArbitrageSweepResult:
    """Wall-position vs. E_max sweep for one architecture."""

    arch_name: str
    points: tuple[ArbitragePoint, ...]


def sweep_engram_capacity(
    arch_name: str,
    config: MATDOConfig,
    *,
    e_max_values: list[int] | None = None,
    r_bits: int = 2,
) -> ArbitrageSweepResult:
    """Sweep over different Engram table sizes and record wall shift."""
    if e_max_values is None:
        e_max_values = [0, 1_000, 8_000, 32_000, 64_000, 128_000, 256_000]

    points: list[ArbitragePoint] = []
    for e_max in e_max_values:
        cfg_e = replace(config, e_max=e_max)
        # When e_max==0 use E=0 directly
        pt = evaluate_arbitrage(arch_name, cfg_e, r_bits=r_bits)
        points.append(pt)
    return ArbitrageSweepResult(arch_name=arch_name, points=tuple(points))


# ---------------------------------------------------------------------------
# Pareto dominance check (Theorem 4.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParetoDominanceResult:
    """Whether the E>0 policy Pareto-dominates E=0."""

    arch_name: str
    baseline_error: float
    engram_error: float
    baseline_t_steps: int
    engram_t_steps: int
    pareto_dominates: bool  # lower error AND lower or equal T


def check_pareto_dominance(
    arch_name: str,
    config: MATDOConfig,
    *,
    rho_hbm: float = 0.95,
    rho_dram: float = 0.20,
    r_bits: int = 2,
) -> ParetoDominanceResult:
    """Compare E=0 vs E=E_max policies at fixed (rho_hbm, rho_dram)."""
    from matdo_new.core.error_model import estimate_error, required_adaptation_steps
    from matdo_new.core.resource_theory import dram_max_engram_entries, hbm_max_m_blocks

    m_cap = hbm_max_m_blocks(rho_hbm, r_bits, config)

    # Baseline (no Engram)
    t_base = required_adaptation_steps(
        r_bits=r_bits,
        m_blocks=m_cap,
        engram_entries=0,
        target_error=config.target_error,
        config=config,
    )
    err_base = estimate_error(
        r_bits=r_bits,
        m_blocks=m_cap,
        t_steps=max(1, t_base),
        engram_entries=0,
        config=config,
    ).total

    # Engram-assisted
    e_effective = min(config.e_max, dram_max_engram_entries(rho_dram, config))
    t_eng = required_adaptation_steps(
        r_bits=r_bits,
        m_blocks=m_cap,
        engram_entries=e_effective,
        target_error=config.target_error,
        config=config,
    )
    err_eng = estimate_error(
        r_bits=r_bits,
        m_blocks=m_cap,
        t_steps=max(1, t_eng),
        engram_entries=e_effective,
        config=config,
    ).total

    pareto = (err_eng <= err_base) and (t_eng <= t_base)

    return ParetoDominanceResult(
        arch_name=arch_name,
        baseline_error=err_base,
        engram_error=err_eng,
        baseline_t_steps=t_base,
        engram_t_steps=t_eng,
        pareto_dominates=pareto,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_arbitrage_study(
    *,
    architecture_profiles: dict[str, dict[str, float]] | None = None,
    base_config: MATDOConfig | None = None,
    r_bits: int = 2,
    pareto_rho_hbm: float = 0.95,
    pareto_rho_dram: float = 0.20,
) -> tuple[ExperimentResult, ...]:
    """Run the full arbitrage validation study.

    Parameters
    ----------
    architecture_profiles:
        Dict of ``{arch_name: coefficient_overrides}``.  Defaults to the
        three profiles from :data:`ARCHITECTURE_PROFILES`.
    base_config:
        Base hardware configuration to augment with per-architecture coefficients.
    r_bits:
        Quantisation bits (fixed across the study).
    pareto_rho_hbm / pareto_rho_dram:
        HBM / DRAM utilisation used for the Pareto-dominance check.

    Returns
    -------
    Tuple of :class:`ExperimentResult` records (one per architecture).
    """
    profiles = architecture_profiles or ARCHITECTURE_PROFILES
    cfg_base = base_config or MATDOConfig()

    results: list[ExperimentResult] = []
    for arch_name, overrides in profiles.items():
        cfg = replace(cfg_base, **overrides)  # type: ignore[arg-type]

        # Core arbitrage check
        arb = evaluate_arbitrage(arch_name, cfg, r_bits=r_bits)

        # Pareto dominance
        pareto = check_pareto_dominance(
            arch_name,
            cfg,
            rho_hbm=pareto_rho_hbm,
            rho_dram=pareto_rho_dram,
            r_bits=r_bits,
        )

        metrics: dict[str, bool | int | float | str] = {
            "inequality_lhs_zeta": arb.inequality_lhs,
            "inequality_rhs": arb.inequality_rhs,
            "arbitrage_inequality_holds": arb.inequality_holds,
            "rho_ctx_baseline": arb.rho_ctx_baseline,
            "rho_ctx_engram": arb.rho_ctx_engram,
            "wall_postponed": arb.wall_postponed,
            "wall_shift": arb.wall_shift,
            "m_min_baseline": arb.m_min_baseline,
            "m_min_engram": arb.m_min_engram,
            # Pareto
            "pareto_dominates": pareto.pareto_dominates,
            "baseline_error": pareto.baseline_error,
            "engram_error": pareto.engram_error,
            "baseline_t_steps": pareto.baseline_t_steps,
            "engram_t_steps": pareto.engram_t_steps,
            # Combined pass
            "theorem_41_passes": arb.inequality_holds and arb.wall_postponed,
            "theorem_42_passes": pareto.pareto_dominates,
        }

        results.append(
            ExperimentResult(
                name=f"arbitrage:{arch_name}",
                kind="arbitrage",
                metrics=metrics,
                metadata={
                    "arch_name": arch_name,
                    "r_bits": r_bits,
                    "e_max": cfg.e_max,
                    "target_error": cfg.target_error,
                },
            )
        )
    return tuple(results)
