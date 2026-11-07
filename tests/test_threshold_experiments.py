import numpy as np
import pytest

from src.evaluation.threshold_experiments import (
    classification_metrics,
    evaluate_clustering_grid,
    evaluate_matching_grid,
    inclusive_grid,
    save_threshold_artifacts,
)


def test_classification_metrics_include_confusion_counts_and_f1():
    result = classification_metrics(
        [0.90, 0.75, 0.60, 0.20], [True, False, True, False], 0.70
    )
    assert result.true_positives == 1
    assert result.true_negatives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)


def test_matching_grid_only_keeps_valid_pairs_and_ranks_false_accepts_first():
    results = evaluate_matching_grid(
        [0.90, 0.70, 0.65, 0.20],
        [True, True, False, False],
        [0.60, 0.70],
        [0.65, 0.80],
    )
    assert all(item.possible_threshold < item.confident_threshold for item in results)
    assert results[0].confident.false_positives == 0


def test_clustering_grid_reports_false_merges_and_splits_deterministically():
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.05, 0.99, 0.0],
        ],
        dtype=np.float32,
    )
    results = evaluate_clustering_grid(
        embeddings,
        ["person-a", "person-a", "person-b", "person-b"],
        [0.80],
        [0.80],
        neighbors=3,
    )
    assert results[0].false_merges == 0
    assert results[0].false_splits == 0
    assert results[0].cluster_purity == 1.0
    assert results[0].unassigned_faces == 0


def test_inclusive_grid_validates_and_includes_end():
    assert inclusive_grid(0.5, 0.6, 0.05) == [0.5, 0.55, 0.6]
    with pytest.raises(ValueError):
        inclusive_grid(0.8, 0.5, 0.1)


def test_threshold_artifacts_warn_when_false_merges_remain(tmp_path):
    matching = evaluate_matching_grid(
        [0.9, 0.65, 0.2],
        [True, False, False],
        [0.6],
        [0.8],
    )
    clustering = evaluate_clustering_grid(
        np.asarray(
            [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03]],
            dtype=np.float32,
        ),
        ["alice", "alice", "bob", "bob"],
        [0.8],
        [0.8],
        neighbors=3,
    )

    csv_path, report_path = save_threshold_artifacts(tmp_path, matching, clustering)

    assert csv_path.is_file()
    report = report_path.read_text(encoding="utf-8")
    assert "DO NOT DEPLOY THIS CLUSTER SETTING" in report
    assert "cross-person false merges" in report
