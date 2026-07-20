import numpy as np
import pytest

from src.evaluation import evaluate_threshold, tune_confidence_tiers, tune_threshold


@pytest.fixture
def labeled_scores():
    return (
        np.array([0.90, 0.80, 0.60, 0.55, 0.45, 0.30]),
        np.array([True, True, True, False, False, False]),
    )


def test_evaluate_threshold_calculates_far_and_frr(labeled_scores):
    scores, labels = labeled_scores
    result = evaluate_threshold(scores, labels, threshold=0.70)
    assert result.far == 0.0
    assert result.frr == pytest.approx(1 / 3)
    assert result.false_accepts == 0
    assert result.false_rejects == 1


def test_threshold_boundary_is_accepted():
    result = evaluate_threshold(
        np.array([0.70, 0.20]), np.array([True, False]), threshold=0.70
    )
    assert result.frr == 0.0


def test_tune_threshold_respects_far_limit(labeled_scores):
    scores, labels = labeled_scores
    result = tune_threshold(scores, labels, max_far=0.0)
    assert result.threshold == pytest.approx(0.60)
    assert result.far == 0.0
    assert result.frr == 0.0


def test_tune_threshold_trades_far_for_lower_frr():
    scores = np.array([0.90, 0.50, 0.60, 0.40])
    labels = np.array([True, True, False, False])
    strict = tune_threshold(scores, labels, max_far=0.0)
    permissive = tune_threshold(scores, labels, max_far=0.5)
    assert strict.frr == 0.5
    assert permissive.frr == 0.0
    assert permissive.far == 0.5


def test_tiers_keep_confident_threshold_at_least_as_high(labeled_scores):
    scores, labels = labeled_scores
    confident, possible = tune_confidence_tiers(
        scores, labels, confident_max_far=0.0, possible_max_far=0.34
    )
    assert confident.threshold >= possible.threshold
    assert confident.far <= 0.0
    assert possible.far <= 0.34


@pytest.mark.parametrize(
    "scores,labels",
    [
        ([0.8], [True]),
        ([0.2], [False]),
        ([], []),
    ],
)
def test_requires_genuine_and_impostor_examples(scores, labels):
    with pytest.raises(ValueError):
        tune_threshold(np.asarray(scores), np.asarray(labels), max_far=0.1)


def test_rejects_invalid_far_limit(labeled_scores):
    scores, labels = labeled_scores
    with pytest.raises(ValueError, match="max_far"):
        tune_threshold(scores, labels, max_far=1.1)
