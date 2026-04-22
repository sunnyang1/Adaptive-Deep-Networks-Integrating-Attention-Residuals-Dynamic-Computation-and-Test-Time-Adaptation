from __future__ import annotations

from dataclasses import dataclass

from matdo_new.core.config import MATDOConfig
from matdo_new.core.online_estimation import OnlineEstimate
from matdo_new.core.policy import PolicyDecision, RuntimeObservation, solve_policy


@dataclass
class MATDOScheduler:
    """Thin wrapper that applies the MATDO policy to runtime observations."""

    config: MATDOConfig = MATDOConfig()
    online_estimate: OnlineEstimate | None = None

    def update_online_estimate(self, estimate: OnlineEstimate) -> None:
        self.online_estimate = estimate

    def decide(self, observation: RuntimeObservation) -> PolicyDecision:
        return solve_policy(
            observation=observation,
            config=self.config,
            online_estimate=self.online_estimate,
        )
