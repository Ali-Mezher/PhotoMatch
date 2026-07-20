from pathlib import Path

import numpy as np

from src.services.indexing_manager import IndexingManager
from src.services import IndexStatus, PhotoMatchService


def _make_event(root: Path, event_id: str = "graduation") -> Path:
    raw = root / event_id / "raw"
    raw.mkdir(parents=True)
    photo = raw / "photo.jpg"
    photo.write_bytes(b"placeholder")
    return photo


def test_reconcile_tracks_new_event_and_unchanged_event(tmp_path):
    _make_event(tmp_path)
    service = PhotoMatchService(events_dir=tmp_path, database_path=tmp_path / "status.sqlite3")

    assert service.reconcile_event("graduation") is True
    event = service.get_event("graduation")
    assert event.status is IndexStatus.PENDING
    assert event.total_images == 1
    assert service.list_image_statuses("graduation")[0].status is IndexStatus.PENDING

    indexed = tmp_path / "graduation" / "indexed"
    indexed.mkdir()
    (indexed / "faces.faiss").write_bytes(b"index")
    (indexed / "metadata.json").write_text("[]")
    fingerprint = service.store.get_fingerprint("graduation")
    service.store.upsert_event("graduation", fingerprint, IndexStatus.INDEXED, 1)
    assert service.reconcile_event("graduation") is False
    assert service.queue_index("graduation") is False


def test_service_records_image_progress_and_returns_search_results(tmp_path):
    photo = _make_event(tmp_path)
    expected = {"confident": ["top"], "possible": ["maybe"]}
    calls = []

    def fake_builder(event_id, show_progress, progress_callback):
        calls.append((event_id, show_progress))
        index_dir = tmp_path / event_id / "indexed"
        index_dir.mkdir()
        (index_dir / "faces.faiss").write_bytes(b"index")
        (index_dir / "metadata.json").write_text("[]")
        progress_callback(photo, "indexed", 2, None)

    service = PhotoMatchService(
        events_dir=tmp_path,
        database_path=tmp_path / "status.sqlite3",
        index_builder=fake_builder,
        matcher=lambda image, event_id, top_k: expected,
    )
    service.reconcile_event("graduation")
    service._run_index_job("graduation")

    event = service.get_event("graduation")
    assert calls == [("graduation", False)]
    assert event.status is IndexStatus.INDEXED
    assert service.list_image_statuses("graduation")[0].face_count == 2
    result = service.search_selfie(np.zeros((1, 1, 3)), "graduation")
    assert result.confident == ["top"]
    assert result.possible == ["maybe"]


def test_queue_rejects_duplicate_work_and_enforces_worker_bounds():
    from threading import Event

    started = Event()
    release = Event()

    def blocking_worker(event_id):
        started.set()
        release.wait(timeout=2)

    manager = IndexingManager(blocking_worker, max_workers=1)
    try:
        assert manager.queue("graduation") is True
        assert started.wait(timeout=1)
        assert manager.queue("graduation") is False
    finally:
        release.set()
        manager.shutdown(wait=True)

    try:
        IndexingManager(lambda event_id: None, max_workers=4)
    except ValueError as exc:
        assert "between 1 and 3" in str(exc)
    else:
        raise AssertionError("Expected max worker validation")
