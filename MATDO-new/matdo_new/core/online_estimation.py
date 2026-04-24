"""Online parameter estimation for the MATDO-E error model (Appendix C).

Two estimators are provided:

* :class:`OnlineRLSEstimator` — the original 2-parameter (δ, ε) RLS from the
  paper, kept for backward compatibility.
* :class:`FullRLSEstimator` — upgraded 6-parameter estimator that jointly
  tracks (α, β, γ, δ, ε, η) via recursive-least-squares with an optional
  forgetting factor.  It also exposes a :meth:`convergence_diagnostics`
  method that reports per-parameter running standard errors and an overall
  convergence flag.

Both estimators produce an :class:`OnlineEstimate` that can be injected into
a :class:`MATDOConfig` via :meth:`OnlineEstimate.apply`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np

from matdo_new.core.config import MATDOConfig

# ---------------------------------------------------------------------------
# Shared estimate snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnlineEstimate:
    """Online parameter estimates for the MATDO-E error model.

    The 2-parameter variant (δ, ε) is the original Appendix-C estimate.
    The 6-parameter variant additionally carries (α, β, γ, η) so that the
    full error model can be recalibrated at runtime.
    """

    delta: float
    epsilon: float
    forgetting_factor: float = 0.95
    sample_count: int = 0

    # Extended 6-parameter fields (None ⇒ not estimated yet)
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None
    eta: float | None = None

    def apply(self, config: MATDOConfig) -> MATDOConfig:
        """Inject the latest estimates into a config snapshot.

        Only replaces a parameter if the estimate is available and positive.
        """
        kwargs: dict[str, float] = {
            "delta": max(0.0, self.delta),
            "epsilon": max(0.0, self.epsilon),
        }
        if self.alpha is not None and self.alpha > 0:
            kwargs["alpha"] = self.alpha
        if self.beta is not None and self.beta > 0:
            kwargs["beta"] = self.beta
        if self.gamma is not None and self.gamma > 0:
            kwargs["gamma"] = self.gamma
        if self.eta is not None and self.eta > 0:
            kwargs["eta"] = self.eta
        return replace(config, **kwargs)


# ---------------------------------------------------------------------------
# Original 2-parameter RLS (backward-compatible)
# ---------------------------------------------------------------------------


@dataclass
class OnlineRLSEstimator:
    """Appendix C: RLS for (delta, epsilon) with forgetting factor lambda.

    Regress y ~ delta * x0 + epsilon * x1 with
    ``x0 = 2^(-2R) / M``, ``x1 = ln(M) / T`` (paper feature map).
    """

    lambda_: float = 0.98
    theta: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    P: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=float) * 100.0)
    n: int = 0

    def update(self, x: np.ndarray, y: float) -> OnlineRLSEstimator:
        x = np.asarray(x, dtype=float).reshape(2)
        denom = self.lambda_ + float(x @ self.P @ x)
        k = self.P @ x / denom
        err = y - float(x @ self.theta)
        self.theta = self.theta + k * err
        self.P = (self.P - np.outer(k, x @ self.P)) / self.lambda_
        self.n += 1
        return self

    def to_online_estimate(self) -> OnlineEstimate:
        d = float(self.theta[0])
        e = float(self.theta[1])
        return OnlineEstimate(
            delta=max(0.0, d),
            epsilon=max(0.0, e),
            forgetting_factor=self.lambda_,
            sample_count=int(self.n),
        )


# ---------------------------------------------------------------------------
# Convergence diagnostic
# ---------------------------------------------------------------------------


class ConvergenceDiagnostics(NamedTuple):
    """Per-parameter standard errors and overall convergence flag.

    Attributes
    ----------
    param_names:
        Names of the tracked parameters in estimation order.
    estimates:
        Current point estimates.
    std_errors:
        Approximate standard errors derived from the RLS covariance matrix
        (square root of diagonal of P * noise_variance_estimate).
    relative_std_errors:
        ``std_error / |estimate|`` for each parameter (inf when estimate ≈ 0).
    converged:
        True when all relative standard errors are below ``tol`` and the
        estimator has seen at least ``min_samples`` observations.
    sample_count:
        Total number of observations processed.
    """

    param_names: tuple[str, ...]
    estimates: tuple[float, ...]
    std_errors: tuple[float, ...]
    relative_std_errors: tuple[float, ...]
    converged: bool
    sample_count: int


# ---------------------------------------------------------------------------
# 6-parameter full RLS estimator
# ---------------------------------------------------------------------------

#: Feature-map documentation for the 6-parameter RLS
#:
#: The error model is:
#:   ε(R,M,T,E) = α·2^{-2R}  +  β·f(E)/(M·S)  +  γ/√T
#:              + δ·2^{-2R}/M  +  ε·ln(M)/T  +  η/E
#:
#: Rewritten for linear regression:
#:   y = θ^T x   where
#:   x[0] = 2^{-2R}                    → coefficient α
#:   x[1] = f(E) / (M · S)             → coefficient β
#:   x[2] = 1 / sqrt(T)                → coefficient γ
#:   x[3] = 2^{-2R} / M                → coefficient δ
#:   x[4] = ln(M) / T                  → coefficient ε
#:   x[5] = 1/E  (0 when E=0)          → coefficient η
_FULL_PARAM_NAMES: tuple[str, ...] = ("alpha", "beta", "gamma", "delta", "epsilon", "eta")
_N_PARAMS = len(_FULL_PARAM_NAMES)


@dataclass
class FullRLSEstimator:
    """Full 6-parameter RLS estimator for the MATDO-E error model.

    Jointly tracks (α, β, γ, δ, ε, η) from online observations of the
    form ``(R, M, T, E, scope_span, engram_f, observed_error)``.

    Parameters
    ----------
    lambda_:
        Forgetting factor in (0, 1].  0.95–0.99 is recommended.
    init_var:
        Initial diagonal variance of the covariance matrix P.
    noise_var_ema:
        Exponential-moving-average coefficient for the noise variance
        estimate used in convergence diagnostics.
    convergence_tol:
        Relative standard-error threshold below which a parameter is
        considered converged.
    min_samples_converged:
        Minimum number of observations before convergence can be declared.
    """

    lambda_: float = 0.97
    init_var: float = 1_000.0
    noise_var_ema: float = 0.05
    convergence_tol: float = 0.10
    min_samples_converged: int = 20

    # Internal state
    theta: np.ndarray = field(default_factory=lambda: np.zeros(_N_PARAMS, dtype=float))
    P: np.ndarray = field(default_factory=lambda: np.eye(_N_PARAMS, dtype=float) * 1_000.0)
    n: int = 0
    _noise_var: float = field(default=1.0, init=False)
    _residual_ema: float = field(default=0.0, init=False)

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    @staticmethod
    def make_features(
        *,
        r_bits: int,
        m_blocks: int,
        t_steps: int,
        engram_entries: int,
        scope_span: int,
        engram_f: float,
    ) -> np.ndarray:
        """Build the 6-element feature vector for one observation.

        Parameters
        ----------
        r_bits:       Quantisation bits R.
        m_blocks:     HBM blocks M (≥1).
        t_steps:      Adaptation steps T (≥1).
        engram_entries: External memory entries E.
        scope_span:   Scope span S (from MATDOConfig.scope_span).
        engram_f:     f(E) = 1 − ζ(1 − e^{−E/E₀}) (from config.engram_compensation).
        """
        r = float(r_bits)
        m = max(1.0, float(m_blocks))
        t = max(1.0, float(t_steps))
        s = max(1.0, float(scope_span))
        e = max(0.0, float(engram_entries))

        x = np.zeros(_N_PARAMS, dtype=float)
        x[0] = 2.0 ** (-2.0 * r)  # α feature
        x[1] = engram_f / (m * s)  # β feature
        x[2] = 1.0 / math.sqrt(t)  # γ feature
        x[3] = (2.0 ** (-2.0 * r)) / m  # δ feature
        x[4] = math.log(m) / t  # ε feature
        x[5] = 1.0 / e if e > 0 else 0.0  # η feature
        return x

    # ------------------------------------------------------------------
    # Online update (RLS with forgetting factor)
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        r_bits: int,
        m_blocks: int,
        t_steps: int,
        engram_entries: int,
        scope_span: int,
        engram_f: float,
        observed_error: float,
    ) -> FullRLSEstimator:
        """Ingest one observation and update the parameter estimates in place.

        Parameters
        ----------
        observed_error:
            Ground-truth end-to-end error (e.g. perplexity residual or
            task-accuracy gap) measured at the given (R, M, T, E) point.

        Returns
        -------
        self (for chaining).
        """
        x = self.make_features(
            r_bits=r_bits,
            m_blocks=m_blocks,
            t_steps=t_steps,
            engram_entries=engram_entries,
            scope_span=scope_span,
            engram_f=engram_f,
        )
        y = float(observed_error)

        # Kalman gain
        denom = self.lambda_ + float(x @ self.P @ x)
        k = (self.P @ x) / denom
        residual = y - float(x @ self.theta)

        # Parameter update
        self.theta = self.theta + k * residual

        # Covariance update (Joseph form for numerical stability)
        I_kx = np.eye(_N_PARAMS) - np.outer(k, x)
        self.P = (I_kx @ self.P @ I_kx.T + np.outer(k, k) * (1.0 - self.lambda_)) / self.lambda_

        # Noise variance EMA (used in convergence diagnostics)
        self._noise_var = (
            1.0 - self.noise_var_ema
        ) * self._noise_var + self.noise_var_ema * residual**2
        self._residual_ema = (
            1.0 - self.noise_var_ema
        ) * self._residual_ema + self.noise_var_ema * abs(residual)

        self.n += 1
        return self

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_online_estimate(self) -> OnlineEstimate:
        """Return current estimates as an :class:`OnlineEstimate`."""
        th = self.theta
        return OnlineEstimate(
            alpha=float(max(0.0, th[0])),
            beta=float(max(0.0, th[1])),
            gamma=float(max(0.0, th[2])),
            delta=float(max(0.0, th[3])),
            epsilon=float(max(0.0, th[4])),
            eta=float(max(0.0, th[5])),
            forgetting_factor=self.lambda_,
            sample_count=int(self.n),
        )

    def convergence_diagnostics(self) -> ConvergenceDiagnostics:
        """Return per-parameter standard errors and a convergence flag.

        Standard errors are computed as:
            se_i = sqrt( P[i,i] * noise_var )

        Relative standard error:
            rse_i = se_i / |theta_i|  (inf when theta_i ≈ 0)

        Convergence is declared when all rse_i < tol and n >= min_samples.
        """
        noise_var = max(1e-12, self._noise_var)
        std_errors: list[float] = []
        rel_std_errors: list[float] = []
        for i in range(_N_PARAMS):
            se = math.sqrt(max(0.0, float(self.P[i, i])) * noise_var)
            std_errors.append(se)
            est_abs = abs(float(self.theta[i]))
            rse = se / est_abs if est_abs > 1e-12 else math.inf
            rel_std_errors.append(rse)

        all_converged = self.n >= self.min_samples_converged and all(
            rse < self.convergence_tol for rse in rel_std_errors
        )

        return ConvergenceDiagnostics(
            param_names=_FULL_PARAM_NAMES,
            estimates=tuple(float(th) for th in self.theta),
            std_errors=tuple(std_errors),
            relative_std_errors=tuple(rel_std_errors),
            converged=all_converged,
            sample_count=int(self.n),
        )

    def residual_ema(self) -> float:
        """Exponentially weighted mean absolute residual (training signal quality)."""
        return float(self._residual_ema)
