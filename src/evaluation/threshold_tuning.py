"""FAR/FRR measurement and threshold selection for face-match scores."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdMetrics:
    """Error rates produced by accepting scores at or above a threshold."""

    threshold: float
    far: float
    frr: float
    false_accepts: int
    false_rejects: int
    genuine_count: int
    impostor_count: int


def _validated_inputs(
    scores: np.ndarray, genuine_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(genuine_labels)
    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("scores and genuine_labels must be one-dimensional")
    if scores.size != labels.size or scores.size == 0:
        raise ValueError("scores and genuine_labels must have the same non-zero length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")

    labels = labels.astype(bool)
    if not labels.any() or labels.all():
        raise ValueError("both genuine and impostor examples are required")
    return scores, labels


def evaluate_threshold(
    scores: np.ndarray, genuine_labels: np.ndarray, threshold: float
) -> ThresholdMetrics:
    """Calculate false acceptance and false rejection rates at ``threshold``."""
    scores, labels = _validated_inputs(scores, genuine_labels)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    accepted = scores >= threshold
    false_accepts = int(np.count_nonzero(accepted & ~labels))
    false_rejects = int(np.count_nonzero(~accepted & labels))
    genuine_count = int(np.count_nonzero(labels))
    impostor_count = int(np.count_nonzero(~labels))
    return ThresholdMetrics(
        threshold=float(threshold),
        far=false_accepts / impostor_count,
        frr=false_rejects / genuine_count,
        false_accepts=false_accepts,
        false_rejects=false_rejects,
        genuine_count=genuine_count,
        impostor_count=impostor_count,
    )


def tune_threshold(
    scores: np.ndarray, genuine_labels: np.ndarray, max_far: float
) -> ThresholdMetrics:
    """Choose the lowest-FRR threshold whose FAR does not exceed ``max_far``.

    Ties prefer the lower FAR, then the higher threshold. This makes the
    selected operating point conservative without rejecting extra genuine
    comparisons.
    """
    scores, labels = _validated_inputs(scores, genuine_labels)
    if not 0.0 <= max_far <= 1.0:
        raise ValueError("max_far must be between 0 and 1")

    candidates = np.append(
        np.unique(scores), np.nextafter(float(np.max(scores)), np.inf)
    )
    metrics = [evaluate_threshold(scores, labels, value) for value in candidates]
    feasible = [metric for metric in metrics if metric.far <= max_far]
    return min(feasible, key=lambda metric: (metric.frr, metric.far, -metric.threshold))


def tune_confidence_tiers(
    scores: np.ndarray,
    genuine_labels: np.ndarray,
    confident_max_far: float = 0.01,
    possible_max_far: float = 0.10,
) -> tuple[ThresholdMetrics, ThresholdMetrics]:
    """Tune strict confident and permissive possible-match thresholds."""
    if confident_max_far > possible_max_far:
        raise ValueError("confident_max_far cannot exceed possible_max_far")

    confident = tune_threshold(scores, genuine_labels, confident_max_far)
    possible = tune_threshold(scores, genuine_labels, possible_max_far)
    if confident.threshold < possible.threshold:
        confident = evaluate_threshold(scores, genuine_labels, possible.threshold)
    return confident, possible
