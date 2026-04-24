"""MATDO-E Paper Proposition Validator.

Aggregates results from all MATDO-E validation studies and produces a
structured report that maps each theorem/proposition to a PASS/FAIL
verdict with evidence.

Paper propositions checked:
  P1. Theorem 3.4 — Wall ordering: ρ_comp < ρ_ctx (requires compute_budget_flops)
  P2. §3.3 — Quadratic blow-up: T* ∝ (ρ_ctx−ρ)^{−2} (exponent ≈ −2.0)
  P3. Theorem 4.1 — Arbitrage inequality holds ∀ architectures in the paper
  P4. Theorem 4.1 — Context wall is postponed when inequality holds
  P5. Theorem 4.2 — Engram policy Pareto-dominates E=0 baseline
  P6. §5.2 (Table 1) — Analytic wall positions match paper within 2% tolerance
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from matdo_new.core.config import MATDOConfig
from matdo_new.experiments.baselines import ExperimentResult
from matdo_new.experiments.studies.arbitrage import run_arbitrage_study
from matdo_new.experiments.studies.architecture_sweep import run_architecture_sweep
from matdo_new.experiments.studies.wall_dynamics import run_wall_dynamics_study

# ---------------------------------------------------------------------------
# Verdict model
# ---------------------------------------------------------------------------

Verdict = Literal["PASS", "FAIL", "SKIP"]


@dataclass(frozen=True)
class PropositionVerdict:
    """Verdict for a single paper proposition."""

    proposition_id: str  # e.g. "P1", "P2"
    title: str
    verdict: Verdict
    evidence: str  # human-readable explanation
    supporting_results: tuple[str, ...]  # ExperimentResult names used


@dataclass(frozen=True)
class ValidationReport:
    """Full validation report for the MATDO-E paper."""

    verdicts: tuple[PropositionVerdict, ...]
    all_passed: bool
    pass_count: int
    fail_count: int
    skip_count: int
    summary: str


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get(results: list[ExperimentResult], name_prefix: str) -> list[ExperimentResult]:
    return [r for r in results if r.name.startswith(name_prefix)]


def _metric(result: ExperimentResult, key: str, default: object = None) -> object:
    return result.metrics.get(key, default)


# ---------------------------------------------------------------------------
# Per-proposition evaluation functions
# ---------------------------------------------------------------------------


def _check_p1_wall_ordering(
    wall_results: list[ExperimentResult],
) -> PropositionVerdict:
    """Theorem 3.4: ρ_comp < ρ_ctx."""
    relevant = [r for r in wall_results if "rho_comp" in r.metrics]
    if not relevant:
        return PropositionVerdict(
            proposition_id="P1",
            title="Theorem 3.4: Wall ordering ρ_comp < ρ_ctx",
            verdict="SKIP",
            evidence="No compute_budget_flops set — ρ_comp not computed. "
            "Set MATDOConfig(compute_budget_flops=...) to test P1.",
            supporting_results=(),
        )
    passed = [r for r in relevant if _metric(r, "wall_ordering_holds") is True]
    verdict: Verdict = "PASS" if len(passed) == len(relevant) else "FAIL"
    evidence = f"{len(passed)}/{len(relevant)} configs satisfy ρ_comp < ρ_ctx. " + (
        f"Gaps: {[round(float(_metric(r,'wall_gap',0)),4) for r in passed]}"  # type: ignore[arg-type]
        if passed
        else "No passing configs."
    )
    return PropositionVerdict(
        proposition_id="P1",
        title="Theorem 3.4: Wall ordering ρ_comp < ρ_ctx",
        verdict=verdict,
        evidence=evidence,
        supporting_results=tuple(r.name for r in relevant),
    )


def _check_p2_quadratic_blowup(
    wall_results: list[ExperimentResult],
) -> PropositionVerdict:
    """§3.3: T* ∝ (ρ_ctx−ρ)^{−2}."""
    relevant = [
        r for r in wall_results if not math.isnan(float(_metric(r, "quadratic_exponent", math.nan)))  # type: ignore[arg-type]
    ]
    if not relevant:
        return PropositionVerdict(
            proposition_id="P2",
            title="§3.3: Quadratic blow-up T* ∝ (ρ_ctx−ρ)^{−2}",
            verdict="SKIP",
            evidence="No valid quadratic fit found (insufficient near-wall points).",
            supporting_results=(),
        )
    close = [r for r in relevant if _metric(r, "exponent_close_to_minus2") is True]
    verdict = "PASS" if close else "FAIL"
    exponents = [round(float(_metric(r, "quadratic_exponent", math.nan)), 3) for r in relevant]  # type: ignore[arg-type]
    r2_vals = [round(float(_metric(r, "quadratic_r2", 0.0)), 3) for r in relevant]  # type: ignore[arg-type]
    evidence = (
        f"Fitted exponents: {exponents}  R²: {r2_vals}. "
        f"{len(close)}/{len(relevant)} are within ±0.5 of −2.0."
    )
    return PropositionVerdict(
        proposition_id="P2",
        title="§3.3: Quadratic blow-up T* ∝ (ρ_ctx−ρ)^{−2}",
        verdict=verdict,
        evidence=evidence,
        supporting_results=tuple(r.name for r in relevant),
    )


def _check_p3_arbitrage_inequality(
    arb_results: list[ExperimentResult],
) -> PropositionVerdict:
    """Theorem 4.1 (LHS): Arbitrage inequality holds for all three architectures."""
    if not arb_results:
        return PropositionVerdict(
            proposition_id="P3",
            title="Theorem 4.1: Arbitrage inequality ζ > η/(E_max·ε_target)",
            verdict="SKIP",
            evidence="No arbitrage results available.",
            supporting_results=(),
        )
    passed = [r for r in arb_results if _metric(r, "arbitrage_inequality_holds") is True]
    verdict = "PASS" if len(passed) == len(arb_results) else "FAIL"
    lhs_vals = [round(float(_metric(r, "inequality_lhs_zeta", 0.0)), 4) for r in arb_results]  # type: ignore[arg-type]
    rhs_vals = [round(float(_metric(r, "inequality_rhs", 0.0)), 6) for r in arb_results]  # type: ignore[arg-type]
    evidence = (
        f"ζ values: {lhs_vals}, thresholds: {rhs_vals}. "
        f"{len(passed)}/{len(arb_results)} architectures pass."
    )
    return PropositionVerdict(
        proposition_id="P3",
        title="Theorem 4.1: Arbitrage inequality ζ > η/(E_max·ε_target)",
        verdict=verdict,
        evidence=evidence,
        supporting_results=tuple(r.name for r in arb_results),
    )


def _check_p4_wall_postponement(
    arb_results: list[ExperimentResult],
) -> PropositionVerdict:
    """Theorem 4.1 (consequence): Context wall is postponed by Engram."""
    if not arb_results:
        return PropositionVerdict(
            proposition_id="P4",
            title="Theorem 4.1: Context wall postponed by Engram",
            verdict="SKIP",
            evidence="No arbitrage results available.",
            supporting_results=(),
        )
    passed = [r for r in arb_results if _metric(r, "wall_postponed") is True]
    verdict = "PASS" if len(passed) == len(arb_results) else "FAIL"
    shifts = [round(float(_metric(r, "wall_shift", 0.0)), 4) for r in arb_results]  # type: ignore[arg-type]
    evidence = (
        f"Wall shifts: {shifts}. "
        f"{len(passed)}/{len(arb_results)} architectures show wall postponement."
    )
    return PropositionVerdict(
        proposition_id="P4",
        title="Theorem 4.1: Context wall postponed by Engram",
        verdict=verdict,
        evidence=evidence,
        supporting_results=tuple(r.name for r in arb_results),
    )


def _check_p5_pareto_dominance(
    arb_results: list[ExperimentResult],
) -> PropositionVerdict:
    """Theorem 4.2: Engram policy Pareto-dominates E=0 baseline."""
    if not arb_results:
        return PropositionVerdict(
            proposition_id="P5",
            title="Theorem 4.2: Pareto dominance of Engram policy",
            verdict="SKIP",
            evidence="No arbitrage results available.",
            supporting_results=(),
        )
    passed = [r for r in arb_results if _metric(r, "pareto_dominates") is True]
    verdict = "PASS" if len(passed) == len(arb_results) else "FAIL"
    err_base = [round(float(_metric(r, "baseline_error", 0.0)), 5) for r in arb_results]  # type: ignore[arg-type]
    err_eng = [round(float(_metric(r, "engram_error", 0.0)), 5) for r in arb_results]  # type: ignore[arg-type]
    evidence = (
        f"Baseline errors: {err_base}, Engram errors: {err_eng}. "
        f"{len(passed)}/{len(arb_results)} architectures: Engram Pareto-dominates."
    )
    return PropositionVerdict(
        proposition_id="P5",
        title="Theorem 4.2: Pareto dominance of Engram policy",
        verdict=verdict,
        evidence=evidence,
        supporting_results=tuple(r.name for r in arb_results),
    )


def _check_p6_table1(
    arch_results: list[ExperimentResult],
) -> PropositionVerdict:
    """§5.2 Table 1: MATDO-E wall positions match paper; wall shift is positive."""
    if not arch_results:
        return PropositionVerdict(
            proposition_id="P6",
            title="§5.2 Table 1: Analytic wall positions vs. paper",
            verdict="SKIP",
            evidence="No architecture sweep results available.",
            supporting_results=(),
        )
    passed = [r for r in arch_results if _metric(r, "table1_validates") is True]
    verdict = "PASS" if len(passed) == len(arch_results) else "FAIL"

    row_details = []
    for r in arch_results:
        arch = str(_metric(r, "arch_name", r.name))
        matdo_ok = bool(_metric(r, "rho_matdo_within_tolerance", False))
        shift = round(float(_metric(r, "wall_shift", 0.0)), 4)  # type: ignore[arg-type]
        paper_shift = round(float(_metric(r, "paper_wall_shift", 0.0)), 4)  # type: ignore[arg-type]
        row_details.append(
            f"{arch}: matdo_ok={matdo_ok}, " f"shift={shift:.4f} (paper: {paper_shift:.4f})"
        )

    evidence = (
        f"{len(passed)}/{len(arch_results)} architectures pass Table 1 checks. "
        + " | ".join(row_details)
    )
    return PropositionVerdict(
        proposition_id="P6",
        title="§5.2 Table 1: MATDO-E wall positions vs. paper (positive wall shift)",
        verdict=verdict,
        evidence=evidence,
        supporting_results=tuple(r.name for r in arch_results),
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _summarize(verdicts: list[PropositionVerdict]) -> str:
    pass_c = sum(1 for v in verdicts if v.verdict == "PASS")
    fail_c = sum(1 for v in verdicts if v.verdict == "FAIL")
    skip_c = sum(1 for v in verdicts if v.verdict == "SKIP")
    lines = [
        "=" * 60,
        "MATDO-E Paper Validation Report",
        "=" * 60,
    ]
    for v in verdicts:
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "—"}[v.verdict]
        lines.append(f"  [{icon}] {v.proposition_id}: {v.title}")
        lines.append(f"       {v.evidence}")
    lines += [
        "-" * 60,
        f"  PASS: {pass_c}  FAIL: {fail_c}  SKIP: {skip_c}",
        "=" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_validation(
    *,
    config: MATDOConfig | None = None,
    wall_dynamics_n_grid: int = 300,
) -> ValidationReport:
    """Run all MATDO-E validation studies and produce a :class:`ValidationReport`.

    Parameters
    ----------
    config:
        Base MATDO configuration.  If None, uses defaults from :class:`MATDOConfig`.
        To test Theorem 3.4, pass a config with ``compute_budget_flops`` set.
    wall_dynamics_n_grid:
        Resolution of the HBM utilisation sweep.

    Returns
    -------
    :class:`ValidationReport` with per-proposition verdicts and summary.
    """
    cfg = config or MATDOConfig()

    # --- Wall dynamics study (P1, P2) ---
    wall_configs = {
        "default": cfg,
    }
    # Add a compute-budget variant so Theorem 3.4 can be checked
    cfg_with_budget = MATDOConfig(compute_budget_flops=1e16)
    wall_configs["with-compute-budget"] = cfg_with_budget

    wall_results = list(run_wall_dynamics_study(wall_configs, n_grid=wall_dynamics_n_grid))

    # --- Arbitrage study (P3, P4, P5) ---
    arb_results = list(run_arbitrage_study())

    # --- Architecture sweep (P6) ---
    arch_results = list(run_architecture_sweep())

    # --- Evaluate propositions ---
    verdicts = [
        _check_p1_wall_ordering(wall_results),
        _check_p2_quadratic_blowup(wall_results),
        _check_p3_arbitrage_inequality(arb_results),
        _check_p4_wall_postponement(arb_results),
        _check_p5_pareto_dominance(arb_results),
        _check_p6_table1(arch_results),
    ]

    pass_count = sum(1 for v in verdicts if v.verdict == "PASS")
    fail_count = sum(1 for v in verdicts if v.verdict == "FAIL")
    skip_count = sum(1 for v in verdicts if v.verdict == "SKIP")
    all_passed = fail_count == 0

    return ValidationReport(
        verdicts=tuple(verdicts),
        all_passed=all_passed,
        pass_count=pass_count,
        fail_count=fail_count,
        skip_count=skip_count,
        summary=_summarize(verdicts),
    )


def print_validation_report(report: ValidationReport) -> None:
    """Print the validation report to stdout."""
    print(report.summary)
    if report.all_passed:
        print("\n🎉  All non-skipped propositions PASSED.")
    else:
        print(f"\n⚠️  {report.fail_count} proposition(s) FAILED.")
