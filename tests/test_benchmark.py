"""
Tests for src/evaluation/benchmark.py. Mocks out the detection/indexing
pipeline (same pattern as tests/test_matching.py) so these run fast and
need no real models or photos — only the orchestration logic (timing,
score labeling, dedup) is under test here.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.benchmark import run_benchmark
from src.evaluation.ground_truth import EvaluationSet, Identity
from src.indexing import IndexedFace


@dataclass
class FakeFace:
    bbox: tuple
    confidence: float
    embedding: np.ndarray


class FakeIndex:
    def __init__(self, scores, metadata):
        self._scores = scores
        self._metadata = metadata

    def search(self, query_embedding, k=50):
        return self._scores, self._metadata


@pytest.fixture
def one_identity_eval_set():
    return EvaluationSet(
        event_id="grad2026",
        raw_dir=Path("/fake/raw"),
        identities=[
            Identity(
                name="alice",
                selfie_path=Path("/fake/alice_selfie.jpg"),
                ground_truth_filenames={"photo1.jpg", "photo2.jpg"},
            )
        ],
    )


def test_run_benchmark_computes_metrics_and_labels_scores(monkeypatch, one_identity_eval_set):
    fake_face = FakeFace(bbox=(0, 0, 10, 10), confidence=0.99, embedding=np.ones(8))
    fake_index = FakeIndex(
        scores=[0.9, 0.8, 0.7],
        metadata=[
            IndexedFace(photo_path="/fake/raw/photo1.jpg", bbox=(0, 0, 5, 5), confidence=0.9),
            IndexedFace(photo_path="/fake/raw/other.jpg", bbox=(0, 0, 5, 5), confidence=0.9),
            IndexedFace(photo_path="/fake/raw/photo2.jpg", bbox=(0, 0, 5, 5), confidence=0.9),
        ],
    )

    monkeypatch.setattr(
        "src.evaluation.benchmark.load_evaluation_set", lambda event_id: one_identity_eval_set
    )
    monkeypatch.setattr(
        "src.evaluation.benchmark.load_event_index", lambda event_id: fake_index
    )
    monkeypatch.setattr(
        "src.evaluation.benchmark.detect_and_embed", lambda img: [fake_face]
    )
    monkeypatch.setattr(
        "src.evaluation.benchmark.preprocess_image", lambda img: img
    )
    monkeypatch.setattr(
        "src.evaluation.benchmark.cv2.imread", lambda path: np.zeros((10, 10, 3), dtype=np.uint8)
    )
    monkeypatch.setattr("src.evaluation.benchmark._warm_up_models", lambda: None)

    result = run_benchmark("grad2026", k=3)

    assert result.event_id == "grad2026"
    assert len(result.identity_results) == 1

    alice_result = result.identity_results[0]
    assert alice_result.name == "alice"
    # photo1.jpg and photo2.jpg are her ground truth; other.jpg is not
    assert alice_result.metrics.num_retrieved_relevant == 2
    assert alice_result.metrics.num_relevant == 2
    assert alice_result.query_time_seconds >= 0

    # scores for her ground-truth photos go to genuine, the rest to impostor
    assert sorted(result.genuine_scores) == [0.7, 0.9]
    assert result.impostor_scores == [0.8]


def test_run_benchmark_raises_when_selfie_has_no_face(monkeypatch, one_identity_eval_set):
    monkeypatch.setattr(
        "src.evaluation.benchmark.load_evaluation_set", lambda event_id: one_identity_eval_set
    )
    monkeypatch.setattr(
        "src.evaluation.benchmark.load_event_index", lambda event_id: FakeIndex([], [])
    )
    monkeypatch.setattr("src.evaluation.benchmark.detect_and_embed", lambda img: [])
    monkeypatch.setattr("src.evaluation.benchmark.preprocess_image", lambda img: img)
    monkeypatch.setattr(
        "src.evaluation.benchmark.cv2.imread", lambda path: np.zeros((10, 10, 3), dtype=np.uint8)
    )
    monkeypatch.setattr("src.evaluation.benchmark._warm_up_models", lambda: None)

    with pytest.raises(ValueError, match="No face detected"):
        run_benchmark("grad2026")


def test_run_benchmark_raises_when_event_not_indexed(monkeypatch, one_identity_eval_set):
    monkeypatch.setattr(
        "src.evaluation.benchmark.load_evaluation_set", lambda event_id: one_identity_eval_set
    )

    def raise_not_found(event_id):
        raise FileNotFoundError()

    monkeypatch.setattr("src.evaluation.benchmark.load_event_index", raise_not_found)
    monkeypatch.setattr("src.evaluation.benchmark._warm_up_models", lambda: None)

    with pytest.raises(FileNotFoundError, match="no built index"):
        run_benchmark("grad2026")
