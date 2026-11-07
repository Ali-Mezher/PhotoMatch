import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.evaluation.benchmark import BenchmarkResult, ScoreObservation
from src.evaluation.exports import (
    export_labeled_embeddings,
    prepare_face_label_workspace,
    save_matching_scores,
)
from src.evaluation.ground_truth import EvaluationSet, Identity
from src.indexing import EventIndex, IndexedFace


def _evaluation_fixture(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    photo = np.full((80, 100, 3), 180, dtype=np.uint8)
    assert cv2.imwrite(str(raw_dir / "group.jpg"), photo)
    identities = [
        Identity("alice", (tmp_path / "alice.jpg",), {"group.jpg"}),
        Identity("bob", (tmp_path / "bob.jpg",), {"group.jpg"}),
    ]
    evaluation_set = EvaluationSet("event", raw_dir, identities)
    index = EventIndex(dim=3)
    vectors = [
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.99, 0.1, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([0.1, 0.99, 0.0], dtype=np.float32),
    ]
    metadata = [
        IndexedFace(str(raw_dir / "group.jpg"), (5 + i * 20, 5, 15, 20), 0.99)
        for i in range(4)
    ]
    index.add(vectors, metadata)
    return evaluation_set, index


def _write_labels(path: Path, labels: list[str]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row, identity in zip(rows, labels):
        row["identity"] = identity
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_matching_score_export_is_ready_for_threshold_grid(tmp_path):
    benchmark = BenchmarkResult(
        "event",
        10,
        [],
        [0.9],
        [0.3],
        [
            ScoreObservation("alice", "front.jpg", "a.jpg", 0.9, True),
            ScoreObservation("alice", "front.jpg", "b.jpg", 0.3, False),
        ],
    )

    path = save_matching_scores(benchmark, tmp_path)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["label"] for row in rows] == ["genuine", "impostor"]
    assert rows[0]["selfie_filename"] == "front.jpg"


def test_face_workspace_creates_crops_without_guessing_identity(tmp_path):
    evaluation_set, index = _evaluation_fixture(tmp_path)

    workspace = prepare_face_label_workspace(evaluation_set, index, tmp_path / "eval")

    with workspace.worksheet_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert workspace.face_count == 4
    assert workspace.labeled_count == 0
    assert all(not row["identity"] for row in rows)
    assert all(row["possible_identities"] == "alice;bob" for row in rows)
    assert len(list(workspace.crops_dir.glob("*.jpg"))) == 4


def test_labeled_embedding_export_uses_confirmed_labels(tmp_path):
    evaluation_set, index = _evaluation_fixture(tmp_path)
    workspace = prepare_face_label_workspace(evaluation_set, index, tmp_path / "eval")
    _write_labels(workspace.worksheet_path, ["alice", "alice", "bob", "bob"])

    result = export_labeled_embeddings(
        index,
        workspace.worksheet_path,
        tmp_path / "eval",
        allowed_identities={"alice", "bob"},
    )

    assert result.embeddings.shape == (4, 3)
    assert result.identities == ("alice", "alice", "bob", "bob")
    assert result.unlabeled_count == 0
    assert result.csv_path.is_file()


def test_regenerating_face_workspace_preserves_confirmed_labels(tmp_path):
    evaluation_set, index = _evaluation_fixture(tmp_path)
    first = prepare_face_label_workspace(evaluation_set, index, tmp_path / "eval")
    _write_labels(first.worksheet_path, ["alice", "alice", "bob", "bob"])

    regenerated = prepare_face_label_workspace(
        evaluation_set, index, tmp_path / "eval"
    )

    assert regenerated.labeled_count == 4
    with regenerated.worksheet_path.open(newline="", encoding="utf-8") as handle:
        assert [row["identity"] for row in csv.DictReader(handle)] == [
            "alice",
            "alice",
            "bob",
            "bob",
        ]


def test_labeled_embedding_export_rejects_unknown_identity(tmp_path):
    evaluation_set, index = _evaluation_fixture(tmp_path)
    workspace = prepare_face_label_workspace(evaluation_set, index, tmp_path / "eval")
    _write_labels(workspace.worksheet_path, ["alice", "alice", "mallory", "bob"])

    with pytest.raises(ValueError, match="unknown identity"):
        export_labeled_embeddings(
            index,
            workspace.worksheet_path,
            tmp_path / "eval",
            allowed_identities={"alice", "bob"},
        )
