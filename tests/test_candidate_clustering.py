"""Tests for the conservative desktop candidate-clustering artifact."""

import json

import numpy as np
import pytest

from src.clustering import (
    cluster_event,
    cluster_event_if_needed,
    cluster_index,
    load_cluster_artifact,
)
from src.indexing import EventIndex, IndexedFace


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _index(vectors):
    index = EventIndex(dim=len(vectors[0]))
    index.add(
        [_unit(vector) for vector in vectors],
        [
            IndexedFace(f"photo_{number}.jpg", (number, 0, 10, 10), 0.99)
            for number in range(len(vectors))
        ],
    )
    return index


def test_clusters_strong_faces_and_leaves_noise_unclustered():
    index = _index(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.10, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.99, 0.10],
            [0.0, 0.0, 1.0],
        ]
    )

    result = cluster_index(
        index, neighbors=4, edge_similarity=0.90, cohesion_similarity=0.90
    )

    assert [len(cluster.members) for cluster in result.clusters] == [2, 2]
    assert [member.face_index for member in result.unclustered] == [4]


def test_cohesion_guard_rejects_weakly_chained_component():
    index = _index([[1.0, 0.0], [0.87, 0.50], [0.50, 0.87]])

    result = cluster_index(
        index, neighbors=2, edge_similarity=0.75, cohesion_similarity=0.90
    )

    assert result.clusters == []
    assert [member.face_index for member in result.unclustered] == [0, 1, 2]


def test_cluster_event_writes_staff_review_artifact(tmp_path, monkeypatch):
    index = _index([[1.0, 0.0], [0.99, 0.10]])
    event_path = tmp_path / "test_event"
    monkeypatch.setattr(
        "src.clustering.clusterer.load_event_index", lambda event_id: index
    )
    monkeypatch.setattr(
        "src.clustering.clusterer.event_dir", lambda event_id: event_path
    )

    result, output_path = cluster_event(
        "test_event",
        neighbors=1,
        edge_similarity=0.90,
        cohesion_similarity=0.90,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["cluster_count"] == len(result.clusters) == 1
    assert payload["clusters"][0]["members"][0]["photo_path"] == "photo_0.jpg"


def test_cluster_event_if_needed_reuses_current_artifact(tmp_path, monkeypatch):
    event_path = tmp_path / "test_event"
    artifact_path = event_path / "indexed" / "clusters.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "event_id": "test_event",
                "configuration": {
                    "neighbors": 50,
                    "edge_similarity": 0.68,
                    "cohesion_similarity": 0.65,
                    "min_cluster_size": 2,
                },
                "clusters": [],
                "unclustered_count": 3,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.clustering.clusterer.event_dir", lambda event_id: event_path
    )
    monkeypatch.setattr(
        "src.clustering.clusterer.cluster_event",
        lambda event_id: pytest.fail("current artifacts must not be recalculated"),
    )

    payload, generated = cluster_event_if_needed("test_event")

    assert generated is False
    assert payload == load_cluster_artifact("test_event")
