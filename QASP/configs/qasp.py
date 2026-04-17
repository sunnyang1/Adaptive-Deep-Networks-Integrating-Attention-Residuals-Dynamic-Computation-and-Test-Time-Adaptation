"""Configuration objects for QASP adaptation behavior."""

from dataclasses import dataclass


@dataclass
class QASPConfig:
    """Hyperparameters controlling matrix-space adaptation."""

    step_size: float = 1e-2
    num_adapt_steps: int = 1
    ns_iters: int = 8
    epsilon: float = 1e-6
    low_pass_ratio: float = 0.25
    entropy_threshold: float = 0.7
    confidence_threshold: float = 0.4

