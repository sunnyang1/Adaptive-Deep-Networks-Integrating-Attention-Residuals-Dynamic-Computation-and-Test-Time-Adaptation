"""Tests for arbitrage study (MATDO-E §4 validation)."""

from __future__ import annotations

from matdo_new.core.config import MATDOConfig
from matdo_new.experiments.studies.arbitrage import (
    ARCHITECTURE_PROFILES,
    check_pareto_dominance,
    evaluate_arbitrage,
    run_arbitrage_study,
    sweep_engram_capacity,
)


class TestEvaluateArbitrage:
    def test_inequality_holds_default_config(self) -> None:
        cfg = MATDOConfig()
        pt = evaluate_arbitrage("test", cfg, r_bits=2)
        assert pt.inequality_lhs == cfg.zeta
        assert pt.inequality_rhs == cfg.eta / (cfg.e_max * cfg.target_error)
        assert pt.inequality_holds == (cfg.zeta > cfg.eta / (cfg.e_max * cfg.target_error))

    def test_wall_postponed_when_inequality_holds(self) -> None:
        cfg = MATDOConfig()  # Default: inequality holds
        pt = evaluate_arbitrage("test", cfg, r_bits=2)
        # When the inequality holds, wall should be postponed
        assert pt.wall_postponed == pt.inequality_holds

    def test_wall_not_postponed_when_inequality_fails(self) -> None:
        # Inequality fails when zeta <= eta / (E_max * target_error)
        # Use zeta=0.01 so that 0.01 <= 0.5/(1000*0.05)=0.01 is False
        cfg = MATDOConfig(zeta=0.01, eta=0.5, e_max=1000, target_error=0.05)
        pt = evaluate_arbitrage("test", cfg, r_bits=2)
        assert pt.inequality_holds is False
        assert pt.wall_postponed is False

    def test_wall_shift_is_positive_when_postponed(self) -> None:
        cfg = MATDOConfig()
        pt = evaluate_arbitrage("test", cfg, r_bits=2)
        if pt.wall_postponed:
            assert pt.wall_shift > 0.0
            assert pt.m_min_engram < pt.m_min_baseline


class TestSweepEngramCapacity:
    def test_sweep_returns_all_e_max_values(self) -> None:
        cfg = MATDOConfig()
        e_values = [0, 64_000, 128_000]
        sweep = sweep_engram_capacity("test", cfg, e_max_values=e_values, r_bits=2)
        assert len(sweep.points) == len(e_values)
        assert sweep.arch_name == "test"

    def test_wall_shift_increases_with_e_max(self) -> None:
        cfg = MATDOConfig()
        sweep = sweep_engram_capacity("test", cfg, r_bits=2)
        shifts = [p.wall_shift for p in sweep.points]
        # Wall shift should be monotonic non-decreasing
        for i in range(1, len(shifts)):
            assert shifts[i] >= shifts[i - 1]


class TestParetoDominance:
    def test_engram_reduces_error(self) -> None:
        cfg = MATDOConfig()
        result = check_pareto_dominance("test", cfg, rho_hbm=0.90, rho_dram=0.20, r_bits=2)
        assert result.baseline_t_steps >= 0
        assert result.engram_t_steps >= 0
        # With Engram, error should be lower or equal
        assert result.engram_error <= result.baseline_error
        # T steps should not increase
        assert result.engram_t_steps <= result.baseline_t_steps
        assert result.pareto_dominates is True


class TestRunArbitrageStudy:
    def test_returns_one_result_per_profile(self) -> None:
        results = run_arbitrage_study()
        assert len(results) == len(ARCHITECTURE_PROFILES)

    def test_all_architectures_pass_inequality(self) -> None:
        results = run_arbitrage_study()
        for r in results:
            assert r.metrics["arbitrage_inequality_holds"] is True
            assert r.metrics["theorem_41_passes"] is True

    def test_metrics_structure(self) -> None:
        results = run_arbitrage_study()
        for r in results:
            assert "inequality_lhs_zeta" in r.metrics
            assert "wall_shift" in r.metrics
            assert "pareto_dominates" in r.metrics
            assert "theorem_41_passes" in r.metrics
            assert "theorem_42_passes" in r.metrics
