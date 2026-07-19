"""
Tests for src/matching. Uses monkeypatching to isolate matcher.py from
the real detection/indexing pipelines (which need MTCNN/deepface/FAISS
data) so these tests run fast and need no models or sample photos.

Run with: pytest tests/test_matching.py -v
"""

from dataclasses import dataclass

import numpy as np
import pytest

from src.matching import cosine_similarity, classify_tier
from src.matching.matcher import match_selfie, NoFaceDetectedError, EventNotIndexedError
from src.indexing import IndexedFace
from config import CONFIDENT_MATCH_THRESHOLD, POSSIBLE_MATCH_THRESHOLD


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_does_not_crash(self):
        a = np.zeros(4)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        assert cosine_similarity(a, b) == 0.0


class TestClassifyTier:
    def test_above_confident_threshold(self):
        assert classify_tier(CONFIDENT_MATCH_THRESHOLD + 0.01) == "confident"

    def test_between_thresholds(self):
        midpoint = (CONFIDENT_MATCH_THRESHOLD + POSSIBLE_MATCH_THRESHOLD) / 2
        assert classify_tier(midpoint) == "possible"

    def test_below_possible_threshold(self):
        assert classify_tier(POSSIBLE_MATCH_THRESHOLD - 0.01) is None

    def test_exact_boundaries_are_inclusive(self):
        assert classify_tier(CONFIDENT_MATCH_THRESHOLD) == "confident"
        assert classify_tier(POSSIBLE_MATCH_THRESHOLD) == "possible"


@dataclass
class FakeFace:
    bbox: tuple
    confidence: float
    embedding: np.ndarray


class FakeIndex:
    """Stands in for a real EventIndex — returns pre-set search results."""

    def __init__(self, scores, metadata):
        self._scores = scores
        self._metadata = metadata

    def search(self, query_embedding, k=50):
        return self._scores, self._metadata


class TestMatchSelfie:
    def test_raises_when_no_face_in_selfie(self, monkeypatch):
        monkeypatch.setattr("src.matching.matcher.detect_and_embed", lambda img: [])
        monkeypatch.setattr("src.matching.matcher.preprocess_image", lambda img: img)

        with pytest.raises(NoFaceDetectedError):
            match_selfie(np.zeros((10, 10, 3), dtype=np.uint8), event_id="test_event")

    def test_raises_when_event_not_indexed(self, monkeypatch):
        fake_face = FakeFace(bbox=(0, 0, 10, 10), confidence=0.99, embedding=np.ones(8))
        monkeypatch.setattr("src.matching.matcher.detect_and_embed", lambda img: [fake_face])
        monkeypatch.setattr("src.matching.matcher.preprocess_image", lambda img: img)

        def raise_not_found(event_id):
            raise FileNotFoundError()

        monkeypatch.setattr("src.matching.matcher.load_event_index", raise_not_found)

        with pytest.raises(EventNotIndexedError):
            match_selfie(np.zeros((10, 10, 3), dtype=np.uint8), event_id="unindexed_event")

    def test_dedups_by_photo_keeping_best_score(self, monkeypatch):
        fake_face = FakeFace(bbox=(0, 0, 10, 10), confidence=0.99, embedding=np.ones(8))
        monkeypatch.setattr("src.matching.matcher.detect_and_embed", lambda img: [fake_face])
        monkeypatch.setattr("src.matching.matcher.preprocess_image", lambda img: img)

        # Same photo appears twice (two faces in one group photo) with
        # different scores — only the higher one should survive.
        scores = [0.9, 0.7, 0.6]
        metadata = [
            IndexedFace(photo_path="group.jpg", bbox=(0, 0, 5, 5), confidence=0.9),
            IndexedFace(photo_path="group.jpg", bbox=(5, 5, 5, 5), confidence=0.8),
            IndexedFace(photo_path="solo.jpg", bbox=(0, 0, 5, 5), confidence=0.95),
        ]
        fake_index = FakeIndex(scores, metadata)
        monkeypatch.setattr(
            "src.matching.matcher.load_event_index", lambda event_id: fake_index
        )

        results = match_selfie(np.zeros((10, 10, 3), dtype=np.uint8), event_id="test_event")

        all_paths = [m.photo_path for m in results["confident"] + results["possible"]]
        assert all_paths.count("group.jpg") == 1  # deduped, not two entries

        group_match = next(
            m for m in results["confident"] + results["possible"] if m.photo_path == "group.jpg"
        )
        assert group_match.score == 0.9  # kept the higher of the two scores

    def test_results_sorted_descending(self, monkeypatch):
        fake_face = FakeFace(bbox=(0, 0, 10, 10), confidence=0.99, embedding=np.ones(8))
        monkeypatch.setattr("src.matching.matcher.detect_and_embed", lambda img: [fake_face])
        monkeypatch.setattr("src.matching.matcher.preprocess_image", lambda img: img)

        scores = [0.70, 0.95, 0.80]
        metadata = [
            IndexedFace(photo_path="c.jpg", bbox=(0, 0, 5, 5), confidence=0.9),
            IndexedFace(photo_path="a.jpg", bbox=(0, 0, 5, 5), confidence=0.9),
            IndexedFace(photo_path="b.jpg", bbox=(0, 0, 5, 5), confidence=0.9),
        ]
        fake_index = FakeIndex(scores, metadata)
        monkeypatch.setattr(
            "src.matching.matcher.load_event_index", lambda event_id: fake_index
        )

        results = match_selfie(np.zeros((10, 10, 3), dtype=np.uint8), event_id="test_event")
        confident_paths = [m.photo_path for m in results["confident"]]
        assert confident_paths == ["a.jpg", "b.jpg", "c.jpg"]  # 0.95, 0.80, 0.70
