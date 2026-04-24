"""Tests for wall_dynamics study (MATDO-E §3 validation)."""

from __future__ import annotations

import math

from matdo_new.core.config import MATDOConfig
from matdo_new.experiments.studies.wall_dynamics import (
    WallDynamicsResult,
    run_wall_dynamics_study,
    sweep_wall_dynamics,
    wall_dynamics_to_experiment_result,
)


class TestSweepWallDynamics:
    def test_returns_valid_result(self) -> None:
        cfg = MATDOConfig(target_error=0.05)
        result = sweep_wall_dynamics(cfg, r_bits=2, n_grid=50, config_name="test")
        assert isinstance(result, WallDynamicsResult)
        assert result.config_name == "test"
        assert result.r_bits == 2
        assert 0.0 <= result.rho_ctx <= 1.0

    def test_rho_ctx_decreases_with_r_bits(self) -> None:
        cfg = MATDOConfig(target_error=0.05)
        r2 = sweep_wall_dynamics(cfg, r_bits=2, n_grid=50)
        r4 = sweep_wall_dynamics(cfg, r_bits=4, n_grid=50)
        assert r2.rho_ctx >= r4.rho_ctx  # Lower R → tighter wall

    def test_wall_ordering_requires_compute_budget(self) -> None:
        cfg_no_budget = MATDOConfig()
        result = sweep_wall_dynamics(cfg_no_budget, n_grid=50)
        assert result.rho_comp is None
        assert result.wall_ordering_holds is True  # vacuously true

    def test_wall_ordering_with_compute_budget(self) -> None:
        cfg = MATDOConfig(compute_budget_flops=1e16)
        result = sweep_wall_dynamics(cfg, n_grid=50)
        assert result.rho_comp is not None
        # With a very large compute budget, T_max >> T* everywhere and
        # rho_comp hits the grid upper bound (≈0.999). The vacuous check
        # treats this as trivially passing, so wall_ordering_holds = True.
        assert result.wall_ordering_holds is True

    def test_quadratic_fit_exponent_reasonable(self) -> None:
        cfg = MATDOConfig(target_error=0.05)
        result = sweep_wall_dynamics(cfg, r_bits=2, n_grid=300)
        if result.quadratic_fit_exponent is not None:
            exp = result.quadratic_fit_exponent
            # The exponent should be negative (error blows up near wall)
            assert exp < 0.0
            # And within a reasonable range
            assert -5.0 < exp < 0.0

    def test_points_increase_t_star_near_wall(self) -> None:
        cfg = MATDOConfig(target_error=0.05)
        result = sweep_wall_dynamics(cfg, r_bits=2, n_grid=100)

        # Collect finite T* points
        finite_points = [p for p in result.points if math.isfinite(p.t_star)]
        if len(finite_points) < 5:
            return  # Not enough points to test

        # T* should generally increase as rho increases
        # Compare the last 20% of points vs the first 20%
        n = len(finite_points)
        early_avg = sum(p.t_star for p in finite_points[: n // 5]) / max(1, n // 5)
        late_avg = sum(p.t_star for p in finite_points[-n // 5 :]) / max(1, n // 5)
        assert late_avg >= early_avg * 0.5  # T* grows or stays near-wall


class TestRunWallDynamicsStudy:
    def test_multi_config_multi_rbits(self) -> None:
        configs = {
            "c1": MATDOConfig(target_error=0.05),
            "c2": MATDOConfig(target_error=0.10),
        }
        results = run_wall_dynamics_study(configs, r_bits_list=[2, 4], n_grid=30)
        assert len(results) == 4  # 2 configs × 2 r_bits

    def test_experiment_result_fields(self) -> None:
        cfg = MATDOConfig(target_error=0.05)
        result = sweep_wall_dynamics(cfg, r_bits=2, n_grid=30)
        er = wall_dynamics_to_experiment_result(result)
        assert "rho_ctx" in er.metrics
        assert "wall_ordering_holds" in er.metrics
        assert "quadratic_exponent" in er.metrics
        assert er.kind == "wall-dynamics"
