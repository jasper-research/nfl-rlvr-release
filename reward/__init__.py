"""Calibration RLVR reward for NFL win-probability prediction.

- grpo_reward: the per-rollout Brier reward (the paper's core) + TRL wrapper + dyn-sampling.
- extract:     thinking-aware probability parsing.
- metrics:     eval-only batch metrics (Brier, ECE, reliability, Murphy decomposition).
"""

from .extract import answer_region, extract_probability
from .grpo_reward import compute_reward, has_sufficient_variance, make_reward_fn
from .metrics import (
    binary_accuracy,
    brier_score,
    expected_calibration_error,
    log_loss,
    maximum_calibration_error,
    murphy_decomposition,
    reliability_curve,
)

__all__ = [
    "answer_region",
    "extract_probability",
    "compute_reward",
    "has_sufficient_variance",
    "make_reward_fn",
    "binary_accuracy",
    "brier_score",
    "expected_calibration_error",
    "log_loss",
    "maximum_calibration_error",
    "murphy_decomposition",
    "reliability_curve",
]
