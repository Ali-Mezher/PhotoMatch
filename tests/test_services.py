"""Compatibility checks for the issue #23 application-service facade."""

from pathlib import Path

import numpy as np
import pytest

from src.indexing import IndexBuildOutcome
from src.services import IndexStatus, IndexingService, PhotoMatchService


class _Updater:
    def __init__(self):
        self.calls = []
        self.indexed = set()

    def __call__(self, event_id, paths, rebuild, show_progress):
        paths = [Path(path) for path in paths]
        self.calls.append((event_id, paths, rebuild))
        self.indexed.add(event_id)
        return object(), [
            IndexBuildOutcome(path, "indexed", face_count=1) for path in paths
        ]

    def exists(self, event_id):
        return event_id in self.indexed


def _indexing(tmp_path):
    events_dir = tmp_path / "events"
    updater = _Updater()
    indexing = IndexingService(
        events_dir=events_dir,
        database_path=tmp_path / "status.sqlite3",
        index_updater=updater,
        index_exists=updater.exists,
    )
    return indexing, updater, events_dir


def test_facade_adopts_legacy_folder_into_incremental_sql_state(tmp_path):
    indexing, _, events_dir = _indexing(tmp_path)
    raw = events_dir / "graduation" / "raw"
    raw.mkdir(parents=True)
    (raw / "photo.jpg").write_bytes(b"placeholder")
    service = PhotoMatchService(
        events_dir=events_dir,
        indexing_service=indexing,
    )

    assert service.discover_events() == ["graduation"]
    assert service.reconcile_event("graduation") is True
    event = service.get_event("graduation")
    assert event is not None
    assert event.status is IndexStatus.PENDING
    assert event.total_images == 1
    assert service.list_image_statuses("graduation")[0].status is IndexStatus.PENDING


def test_facade_returns_interface_neutral_search_result(tmp_path):
    indexing, _, events_dir = _indexing(tmp_path)
    expected = {"confident": ["top"], "possible": ["maybe"]}
    service = PhotoMatchService(
        events_dir=events_dir,
        indexing_service=indexing,
        matcher=lambda image, event_id, top_k: expected,
    )

    result = service.search_selfie(np.zeros((1, 1, 3)), "graduation")

    assert result.confident == ["top"]
    assert result.possible == ["maybe"]


def test_facade_preserves_single_worker_resource_guarantee(tmp_path):
    with pytest.raises(ValueError, match="must be 1"):
        PhotoMatchService(
            events_dir=tmp_path / "events",
            database_path=tmp_path / "status.sqlite3",
            max_index_workers=2,
        )
