"""Evaluation helpers for measuring and tuning PhotoMatch."""

from .threshold_tuning import (
    ThresholdMetrics,
    evaluate_threshold,
    tune_confidence_tiers,
    tune_threshold,
)

__all__ = [
    "ThresholdMetrics",
    "evaluate_threshold",
    "tune_threshold",
    "tune_confidence_tiers",
]
