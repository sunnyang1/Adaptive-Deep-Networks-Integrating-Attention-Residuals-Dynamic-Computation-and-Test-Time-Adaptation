"""Tests for architecture sweep study (MATDO-E §5 Table 1 validation)."""

from __future__ import annotations

from matdo_new.experiments.studies.architecture_sweep import (
    ARCH_SIMULATION_PROFILES,
    run_architecture_sweep,
    simulate_architecture,
)


class TestSimulateArchitecture:
    def test_rho_matdo_greater_than_baseline(self) -> None:
        for arch_name, profile in ARCH_SIMULATION_PROFILES.items():
            sim = simulate_architecture(arch_name, profile, r_bits=2)
            # MATDO-E (R=2 + Engram) should give a higher wall than baseline (R=8, no Engram)
            assert (
                sim.rho_ctx_matdo > sim.rho_ctx_baseline
            ), f"{arch_name}: MATDO-E wall should exceed baseline wall"

    def test_wall_shift_positive(self) -> None:
        for arch_name, profile in ARCH_SIMULATION_PROFILES.items():
            sim = simulate_architecture(arch_name, profile, r_bits=2)
            assert sim.wall_shift > 0.0, f"{arch_name}: wall shift should be positive"

    def test_table1_validation_tolerance(self) -> None:
        for arch_name, profile in ARCH_SIMULATION_PROFILES.items():
            sim = simulate_architecture(arch_name, profile, r_bits=2)
            # Baseline model (R=8, analytic) ≠ vLLM empirical → always False
            assert (
                sim.rho_baseline_within_tolerance is False
            ), f"{arch_name}: baseline model ≠ vLLM empirical baseline"
            # MATDO-E column should be within tolerance
            assert (
                sim.rho_matdo_within_tolerance
            ), f"{arch_name}: MATDO rho should be within tolerance of paper value"

    def test_t_reduction_positive(self) -> None:
        for arch_name, profile in ARCH_SIMULATION_PROFILES.items():
            sim = simulate_architecture(arch_name, profile, r_bits=2)
            assert sim.t_reduction_pct >= 0.0, f"{arch_name}: T* reduction should be non-negative"

    def test_table1_validates_all_architectures(self) -> None:
        all_pass = True
        for arch_name, profile in ARCH_SIMULATION_PROFILES.items():
            sim = simulate_architecture(arch_name, profile, r_bits=2)
            if not sim.table1_validates:
                all_pass = False
        assert all_pass, "All architectures should validate Table 1"


class TestRunArchitectureSweep:
    def test_returns_one_result_per_architecture(self) -> None:
        results = run_architecture_sweep()
        assert len(results) == len(ARCH_SIMULATION_PROFILES)

    def test_result_names(self) -> None:
        results = run_architecture_sweep()
        result_names = {r.name for r in results}
        expected = {f"arch-sweep:{name}" for name in ARCH_SIMULATION_PROFILES}
        assert result_names == expected

    def test_metrics_include_table1_columns(self) -> None:
        results = run_architecture_sweep()
        for r in results:
            assert "rho_ctx_baseline" in r.metrics
            assert "rho_ctx_matdo" in r.metrics
            assert "wall_shift" in r.metrics
            assert "wall_shift_pct" in r.metrics
            assert "table1_validates" in r.metrics
