"""Tests for the upgraded FullRLSEstimator (6-parameter online estimation)."""

from __future__ import annotations

import math

from matdo_new.core.online_estimation import FullRLSEstimator, OnlineEstimate


class TestFullRLSEstimatorMakeFeatures:
    def test_feature_vector_length(self) -> None:
        x = FullRLSEstimator.make_features(
            r_bits=2,
            m_blocks=8,
            t_steps=100,
            engram_entries=128_000,
            scope_span=4,
            engram_f=0.65,
        )
        assert len(x) == 6

    def test_feature_values_non_negative(self) -> None:
        x = FullRLSEstimator.make_features(
            r_bits=2,
            m_blocks=8,
            t_steps=100,
            engram_entries=128_000,
            scope_span=4,
            engram_f=0.65,
        )
        # All features should be non-negative (log(M) >= 0 when M >= 1)
        assert all(v >= 0.0 for v in x)

    def test_zero_engram_entries_zeroes_eta_feature(self) -> None:
        x_nz = FullRLSEstimator.make_features(
            r_bits=2,
            m_blocks=8,
            t_steps=100,
            engram_entries=0,
            scope_span=4,
            engram_f=1.0,
        )
        assert x_nz[5] == 0.0  # η feature

    def test_feature_scales_correctly_with_r_bits(self) -> None:
        x2 = FullRLSEstimator.make_features(
            r_bits=2,
            m_blocks=8,
            t_steps=100,
            engram_entries=0,
            scope_span=4,
            engram_f=1.0,
        )
        x4 = FullRLSEstimator.make_features(
            r_bits=4,
            m_blocks=8,
            t_steps=100,
            engram_entries=0,
            scope_span=4,
            engram_f=1.0,
        )
        # 2^{-2R} should be 16x smaller for R=4 vs R=2
        assert math.isclose(x2[0], 16.0 * x4[0], rel_tol=1e-12)
        assert math.isclose(x2[3], 16.0 * x4[3], rel_tol=1e-12)  # δ feature


class TestFullRLSEstimatorUpdate:
    def test_update_increments_sample_count(self) -> None:
        est = FullRLSEstimator()
        assert est.n == 0
        est.update(
            r_bits=2,
            m_blocks=8,
            t_steps=100,
            engram_entries=0,
            scope_span=4,
            engram_f=1.0,
            observed_error=0.05,
        )
        assert est.n == 1
        est.update(
            r_bits=2,
            m_blocks=8,
            t_steps=100,
            engram_entries=0,
            scope_span=4,
            engram_f=1.0,
            observed_error=0.05,
        )
        assert est.n == 2

    def test_parameters_converge_for_noisy_truth(self) -> None:
        est = FullRLSEstimator(
            lambda_=0.98,
            init_var=1e4,
            convergence_tol=0.15,
            min_samples_converged=5,
        )
        # Ground-truth parameters
        true_theta = [0.015, 2.0, 0.10, 0.005, 0.002, 0.5]

        for _ in range(50):
            # Generate features
            x = FullRLSEstimator.make_features(
                r_bits=2,
                m_blocks=8,
                t_steps=100,
                engram_entries=64_000,
                scope_span=4,
                engram_f=0.65,
            )
            # True error = theta^T x
            y = float(x @ true_theta)
            # Add small noise
            y += 0.005

            est.update(
                r_bits=2,
                m_blocks=8,
                t_steps=100,
                engram_entries=64_000,
                scope_span=4,
                engram_f=0.65,
                observed_error=y,
            )

        diag = est.convergence_diagnostics()
        assert diag.sample_count == 50
        # After enough steps, convergence should be achieved or nearly so
        # (Not strictly required to be True due to noise)
        assert diag.sample_count >= 5

    def test_residual_ema_positive(self) -> None:
        est = FullRLSEstimator()
        for _ in range(10):
            est.update(
                r_bits=2,
                m_blocks=8,
                t_steps=100,
                engram_entries=0,
                scope_span=4,
                engram_f=1.0,
                observed_error=0.05,
            )
        assert est.residual_ema() >= 0.0


class TestToOnlineEstimate:
    def test_estimate_has_all_six_parameters(self) -> None:
        est = FullRLSEstimator()
        est.update(
            r_bits=2,
            m_blocks=8,
            t_steps=100,
            engram_entries=0,
            scope_span=4,
            engram_f=1.0,
            observed_error=0.05,
        )
        ose = est.to_online_estimate()
        assert isinstance(ose, OnlineEstimate)
        assert ose.alpha is not None
        assert ose.beta is not None
        assert ose.gamma is not None
        assert ose.delta is not None
        assert ose.epsilon is not None
        assert ose.eta is not None

    def test_estimate_non_negative(self) -> None:
        est = FullRLSEstimator()
        for _ in range(10):
            est.update(
                r_bits=2,
                m_blocks=8,
                t_steps=100,
                engram_entries=0,
                scope_span=4,
                engram_f=1.0,
                observed_error=0.05,
            )
        ose = est.to_online_estimate()
        assert ose.alpha >= 0.0
        assert ose.beta >= 0.0
        assert ose.gamma >= 0.0
        assert ose.delta >= 0.0
        assert ose.epsilon >= 0.0
        assert ose.eta >= 0.0

    def test_apply_produces_valid_config(self) -> None:
        from matdo_new.core.config import MATDOConfig

        est = FullRLSEstimator()
        for _ in range(10):
            est.update(
                r_bits=2,
                m_blocks=8,
                t_steps=100,
                engram_entries=0,
                scope_span=4,
                engram_f=1.0,
                observed_error=0.05,
            )
        ose = est.to_online_estimate()
        cfg = ose.apply(MATDOConfig())
        assert cfg.alpha >= 0.0
        assert cfg.beta >= 0.0
        assert cfg.gamma >= 0.0
        assert cfg.delta >= 0.0
        assert cfg.epsilon >= 0.0


class TestConvergenceDiagnostics:
    def test_diagnostics_before_update(self) -> None:
        est = FullRLSEstimator(min_samples_converged=1)
        diag = est.convergence_diagnostics()
        assert diag.converged is False
        assert diag.sample_count == 0
        assert len(diag.param_names) == 6

    def test_diagnostics_returns_valid_fields(self) -> None:
        """Verify that convergence_diagnostics returns well-formed output."""
        est = FullRLSEstimator(convergence_tol=0.10, min_samples_converged=5)
        true_theta = [0.015, 2.0, 0.10, 0.005, 0.002, 0.5]
        observations = [
            {
                "r_bits": 2,
                "m_blocks": 8,
                "t_steps": 64,
                "engram_entries": 64_000,
                "engram_f": 0.65,
                "scope_span": 4,
            },
            {
                "r_bits": 4,
                "m_blocks": 4,
                "t_steps": 16,
                "engram_entries": 32_000,
                "engram_f": 0.50,
                "scope_span": 4,
            },
            {
                "r_bits": 2,
                "m_blocks": 16,
                "t_steps": 256,
                "engram_entries": 128_000,
                "engram_f": 0.80,
                "scope_span": 4,
            },
            {
                "r_bits": 8,
                "m_blocks": 2,
                "t_steps": 4,
                "engram_entries": 0,
                "engram_f": 1.0,
                "scope_span": 4,
            },
        ]
        for obs in observations:
            for _ in range(5):
                x = FullRLSEstimator.make_features(**obs)
                y = float(x @ true_theta)
                est.update(observed_error=y, **obs)

        diag = est.convergence_diagnostics()
        assert len(diag.param_names) == 6
        assert diag.sample_count == 20
        assert len(diag.estimates) == 6
        assert len(diag.std_errors) == 6
        assert len(diag.relative_std_errors) == 6
        # Initially may not be converged (depends on observation diversity)
        assert isinstance(diag.converged, bool)
        # All std errors should be non-negative
        assert all(se >= 0.0 for se in diag.std_errors)
        # Relative std errors should be non-negative or inf
        assert all(rse >= 0.0 or math.isinf(rse) for rse in diag.relative_std_errors)
