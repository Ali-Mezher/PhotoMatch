"""
Tests for src/indexing. Runs against a real (small, in-memory) FAISS
index with synthetic embeddings — no photos or ML models needed, since
FAISS itself is lightweight and doesn't require GPU/heavy deps.

Run with: pytest tests/test_indexing.py -v
"""

import json
from dataclasses import dataclass

import numpy as np
import pytest

from src.indexing import EventIndex, IndexedFace, build_event_index, load_event_index


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

    def test_add_rejects_wrong_dimension(self):
        index = EventIndex(dim=8)
        with pytest.raises(ValueError, match="shape"):
            index.add(
                [np.zeros(7, dtype=np.float32)],
                [IndexedFace("a.jpg", (0, 0, 1, 1), 1.0)],
            )

    def test_add_normalizes_embeddings(self):
        index = EventIndex(dim=2)
        index.add(
            [np.array([10.0, 0.0], dtype=np.float32)],
            [IndexedFace("a.jpg", (0, 0, 1, 1), 1.0)],
        )
        scores, _ = index.search(np.array([2.0, 0.0], dtype=np.float32), k=1)
        assert scores[0] == pytest.approx(1.0)

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
        assert index.embeddings().shape == (0, 8)

    def test_embeddings_reconstruct_in_metadata_order(self, populated_index):
        index, target = populated_index

        reconstructed = index.embeddings()

        assert reconstructed.shape == (3, 8)
        assert np.dot(reconstructed[0], target) == pytest.approx(1.0)

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

    def test_load_rejects_mismatched_metadata(self, populated_index, tmp_path):
        index, _ = populated_index
        index.save(tmp_path)
        (tmp_path / "metadata.json").write_text(json.dumps([]))

        with pytest.raises(ValueError, match="inconsistent"):
            EventIndex.load(tmp_path)


@dataclass
class _FakeFace:
    bbox: tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray


def test_event_index_end_to_end(tmp_path, monkeypatch):
    """Build, save, load, and query one event through the public API."""
    event_root = tmp_path / "events" / "graduation"
    raw_dir = event_root / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "group.JPG").write_bytes(b"test image placeholder")

    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0

    monkeypatch.setattr(
        "src.indexing.build_index.event_dir", lambda event_id: tmp_path / "events" / event_id
    )
    monkeypatch.setattr(
        "src.indexing.build_index.cv2.imread",
        lambda path: np.zeros((20, 20, 3), dtype=np.uint8),
    )
    monkeypatch.setattr("src.indexing.build_index.preprocess_image", lambda image: image)
    monkeypatch.setattr(
        "src.indexing.build_index.detect_and_embed",
        lambda image: [_FakeFace((1, 2, 10, 12), 0.98, embedding)],
    )

    built = build_event_index("graduation", show_progress=False)
    loaded = load_event_index("graduation")
    scores, metadata = loaded.search(embedding, k=1)

    assert len(built) == len(loaded) == 1
    assert scores[0] == pytest.approx(1.0)
    assert metadata[0].photo_path.endswith("group.JPG")
    assert tuple(metadata[0].bbox) == (1, 2, 10, 12)
