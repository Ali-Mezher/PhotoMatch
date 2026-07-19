"""
Tests for src/indexing. Runs against a real (small, in-memory) FAISS
index with synthetic embeddings — no photos or ML models needed, since
FAISS itself is lightweight and doesn't require GPU/heavy deps.

Run with: pytest tests/test_indexing.py -v
"""

import numpy as np
import pytest

from src.indexing import EventIndex, IndexedFace


def _normalize(v):
    return v / np.linalg.norm(v)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def populated_index(rng):
    index = EventIndex(dim=8)
    target = _normalize(rng.random(8).astype(np.float32))
    close = _normalize(target + 0.02 * rng.random(8).astype(np.float32))
    far = _normalize(rng.random(8).astype(np.float32))

    index.add(
        [target, close, far],
        [
            IndexedFace(photo_path="a.jpg", bbox=(0, 0, 10, 10), confidence=0.99),
            IndexedFace(photo_path="b.jpg", bbox=(0, 0, 10, 10), confidence=0.95),
            IndexedFace(photo_path="c.jpg", bbox=(0, 0, 10, 10), confidence=0.90),
        ],
    )
    return index, target


class TestEventIndex:
    def test_add_increases_length(self, populated_index):
        index, _ = populated_index
        assert len(index) == 3

    def test_add_rejects_mismatched_lengths(self):
        index = EventIndex(dim=8)
        with pytest.raises(ValueError):
            index.add(
                [np.zeros(8, dtype=np.float32)],
                [
                    IndexedFace(photo_path="a.jpg", bbox=(0, 0, 1, 1), confidence=1.0),
                    IndexedFace(photo_path="b.jpg", bbox=(0, 0, 1, 1), confidence=1.0),
                ],
            )

    def test_search_returns_closest_first(self, populated_index):
        index, target = populated_index
        scores, metadata = index.search(target, k=3)

        assert len(scores) == 3
        assert metadata[0].photo_path == "a.jpg"  # exact match should be first
        assert scores[0] >= scores[1] >= scores[2]  # descending order

    def test_search_on_empty_index_returns_empty(self):
        index = EventIndex(dim=8)
        scores, metadata = index.search(np.zeros(8, dtype=np.float32), k=5)
        assert scores == []
        assert metadata == []

    def test_search_k_larger_than_index_size(self, populated_index):
        index, target = populated_index
        scores, metadata = index.search(target, k=1000)
        assert len(scores) == 3  # capped at index size, no crash

    def test_save_and_load_roundtrip(self, populated_index, tmp_path):
        index, target = populated_index
        index.save(tmp_path)

        reloaded = EventIndex.load(tmp_path)
        assert len(reloaded) == len(index)

        scores, metadata = reloaded.search(target, k=1)
        assert metadata[0].photo_path == "a.jpg"

    def test_load_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EventIndex.load(tmp_path / "does_not_exist")
