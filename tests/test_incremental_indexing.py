from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import demo_goated_index_and_search as demo_script
from src.indexing import IndexBuildOutcome, load_event_index, update_event_index
from src.services import IndexStatus, IndexingService
from src.services.indexing_manager import IndexingManager


@dataclass
class _FakeFace:
    bbox: tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray


def _add_photo(events_dir: Path, event_id: str, name: str, content=b"photo") -> Path:
    raw_dir = events_dir / event_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    photo = raw_dir / name
    photo.write_bytes(content)
    return photo.resolve()


class _FakeUpdater:
    def __init__(self):
        self.calls = []
        self.indexed_events = set()

    def __call__(self, event_id, paths, rebuild, show_progress):
        self.calls.append((event_id, [Path(path) for path in paths], rebuild))
        self.indexed_events.add(event_id)
        outcomes = [
            IndexBuildOutcome(Path(path), "indexed", face_count=1) for path in paths
        ]
        return object(), outcomes

    def exists(self, event_id):
        return event_id in self.indexed_events


def _service(tmp_path):
    updater = _FakeUpdater()
    events_dir = tmp_path / "events"
    service = IndexingService(
        events_dir=events_dir,
        database_path=tmp_path / "status.sqlite3",
        index_updater=updater,
        index_exists=updater.exists,
    )
    return service, updater, events_dir


def test_queue_processes_oldest_event_first(tmp_path):
    service, updater, events_dir = _service(tmp_path)
    _add_photo(events_dir, "newer", "one.jpg")
    _add_photo(events_dir, "older", "one.jpg")
    service.register_event("newer", "2026-07-20")
    service.register_event("older", "2026-07-01")

    service.request_index("newer")
    service.request_index("older")
    service._drain_queue()

    assert [call[0] for call in updater.calls] == ["older", "newer"]


def test_registration_rejects_unsafe_event_id_and_invalid_date(tmp_path):
    service, _, events_dir = _service(tmp_path)
    _add_photo(events_dir, "graduation", "one.jpg")

    with pytest.raises(ValueError, match="event_id"):
        service.register_event("../graduation", "2026-07-01")
    with pytest.raises(ValueError, match="ISO date"):
        service.register_event("graduation", "July 1")


def test_registration_assigns_a_stable_attendee_access_code(tmp_path):
    service, _, events_dir = _service(tmp_path)
    _add_photo(events_dir, "graduation", "one.jpg")

    service.register_event("graduation", "2026-07-01")
    first_code = service.get_event_access_code("graduation")
    service.register_event("graduation", "2026-07-02")

    assert len(first_code) == 8
    assert set(first_code) <= set("0123456789ABCDEF")
    assert service.get_event_access_code("graduation") == first_code


def test_demo_index_command_uses_incremental_service(monkeypatch):
    class FakeService:
        def __init__(self):
            self.get_calls = 0
            self.requested = False
            self.ran = False

        def get_event(self, event_id):
            self.get_calls += 1
            statuses = {
                1: SimpleNamespace(status=IndexStatus.INDEXED),
                2: SimpleNamespace(
                    status=IndexStatus.QUEUED,
                    rebuild_required=False,
                    pending_images=1,
                    total_images=5,
                ),
                3: SimpleNamespace(
                    status=IndexStatus.INDEXED,
                    indexed_images=4,
                    no_face_images=1,
                    failed_images=0,
                    error=None,
                ),
            }
            return statuses[self.get_calls]

        def request_index(self, event_id):
            self.requested = True

        def run_pending(self, show_progress):
            self.ran = show_progress

        def get_event_access_code(self, event_id):
            return "A1B2C3D4"

    service = FakeService()
    monkeypatch.setattr(demo_script, "IndexingService", lambda: service)

    assert demo_script.run_index("graduation") == 0
    assert service.requested is True
    assert service.ran is True


def test_duplicate_queue_requests_are_coalesced(tmp_path):
    service, updater, events_dir = _service(tmp_path)
    _add_photo(events_dir, "graduation", "one.jpg")
    service.register_event("graduation", "2026-07-01")

    assert service.request_index("graduation") is True
    assert service.request_index("graduation") is False
    service._drain_queue()

    assert len(updater.calls) == 1


def test_new_photo_is_appended_without_reprocessing_existing_photo(tmp_path):
    service, updater, events_dir = _service(tmp_path)
    first = _add_photo(events_dir, "graduation", "first.jpg")
    service.register_event("graduation", "2026-07-01")
    service.request_index("graduation")
    service._drain_queue()

    second = _add_photo(events_dir, "graduation", "second.jpg")
    service.request_index("graduation")
    service._drain_queue()

    assert updater.calls[0] == ("graduation", [first], True)
    assert updater.calls[1] == ("graduation", [second], False)
    assert service.get_event("graduation").indexed_images == 2


def test_startup_reconciliation_detects_manual_addition_without_reindexing_old_photo(
    tmp_path,
):
    service, updater, events_dir = _service(tmp_path)
    first = _add_photo(events_dir, "graduation", "first.jpg")
    service.register_event("graduation", "2026-07-01")
    service.request_index("graduation")
    service._drain_queue()

    second = _add_photo(events_dir, "graduation", "second.jpg")
    assert service.reconcile_registered_events() == ["graduation"]
    service._drain_queue()

    assert updater.calls == [
        ("graduation", [first], True),
        ("graduation", [second], False),
    ]


def test_changed_or_removed_photo_rebuilds_current_inventory(tmp_path):
    service, updater, events_dir = _service(tmp_path)
    first = _add_photo(events_dir, "graduation", "first.jpg", b"first")
    second = _add_photo(events_dir, "graduation", "second.jpg", b"second")
    service.register_event("graduation", "2026-07-01")
    service.request_index("graduation")
    service._drain_queue()

    first.write_bytes(b"changed-size")
    service.request_index("graduation")
    service._drain_queue()
    assert updater.calls[-1] == ("graduation", [first, second], True)

    second.unlink()
    service.request_index("graduation")
    service._drain_queue()
    assert updater.calls[-1] == ("graduation", [first], True)
    assert [row.photo_path for row in service.list_image_statuses("graduation")] == [
        str(first)
    ]


def test_failed_image_can_be_explicitly_retried(tmp_path):
    service, updater, events_dir = _service(tmp_path)
    photo = _add_photo(events_dir, "graduation", "broken.jpg")

    def fail_once(event_id, paths, rebuild, show_progress):
        updater.indexed_events.add(event_id)
        updater.calls.append((event_id, paths, rebuild))
        return object(), [IndexBuildOutcome(photo, "failed", error="bad image")]

    service._index_updater = fail_once
    service.register_event("graduation", "2026-07-01")
    service.request_index("graduation")
    service._drain_queue()
    assert service.list_image_statuses("graduation")[0].status is IndexStatus.FAILED

    service._index_updater = updater
    assert service.retry_failed("graduation") == 1
    service._drain_queue()
    assert service.list_image_statuses("graduation")[0].status is IndexStatus.INDEXED


def test_existing_index_is_rebuilt_when_sql_inventory_is_first_adopted(tmp_path):
    service, updater, events_dir = _service(tmp_path)
    photo = _add_photo(events_dir, "graduation", "one.jpg")
    updater.indexed_events.add("graduation")

    event = service.register_event("graduation", "2026-07-01")
    assert event.rebuild_required is True
    service.request_index("graduation")
    service._drain_queue()

    assert updater.calls == [("graduation", [photo], True)]


def test_readable_legacy_index_is_adopted_without_reprocessing(tmp_path):
    updater = _FakeUpdater()
    events_dir = tmp_path / "events"
    photo = _add_photo(events_dir, "graduation", "one.jpg")
    updater.indexed_events.add("graduation")
    legacy_index = SimpleNamespace(
        metadata=[
            SimpleNamespace(photo_path=str(photo)),
            SimpleNamespace(photo_path=str(photo)),
        ]
    )
    service = IndexingService(
        events_dir=events_dir,
        database_path=tmp_path / "status.sqlite3",
        index_updater=updater,
        index_exists=updater.exists,
        index_loader=lambda event_id: legacy_index,
    )

    event = service.register_event("graduation", "2026-07-01")

    assert event.status is IndexStatus.INDEXED
    assert event.rebuild_required is False
    assert event.indexed_images == 1
    assert service.list_image_statuses("graduation")[0].face_count == 2
    assert service.request_index("graduation") is False
    assert updater.calls == []


def test_one_event_failure_does_not_stop_the_queue(tmp_path):
    service, updater, events_dir = _service(tmp_path)
    _add_photo(events_dir, "broken", "one.jpg")
    _add_photo(events_dir, "healthy", "one.jpg")
    service.register_event("broken", "2026-07-01")
    service.register_event("healthy", "2026-07-02")

    original_updater = service._index_updater

    def fail_broken(event_id, paths, rebuild, show_progress):
        if event_id == "broken":
            raise RuntimeError("simulated indexing failure")
        return original_updater(event_id, paths, rebuild, show_progress)

    service._index_updater = fail_broken
    service.request_index("broken")
    service.request_index("healthy")
    service._drain_queue()

    assert service.get_event("broken").status is IndexStatus.FAILED
    assert service.get_event("healthy").status is IndexStatus.INDEXED


def test_worker_waits_for_interrupt_instead_of_polling():
    drained = Event()
    manager = IndexingManager(drained.set)
    manager.start()
    try:
        assert not drained.wait(timeout=0.05)
        manager.signal()
        assert drained.wait(timeout=1)
    finally:
        manager.shutdown()


def test_startup_recovery_restores_pending_image_work(tmp_path):
    service, _, events_dir = _service(tmp_path)
    photo = _add_photo(events_dir, "graduation", "one.jpg")
    service.register_event("graduation", "2026-07-01")

    assert service.store.recover_interrupted_events() == ["graduation"]
    job = service.store.claim_next_event()

    assert job == ("graduation", False)
    assert service.store.paths_for_processing("graduation", rebuild=False) == [
        str(photo)
    ]


def test_image_added_during_active_job_is_not_lost(tmp_path):
    events_dir = tmp_path / "events"
    first = _add_photo(events_dir, "graduation", "first.jpg")
    started = Event()
    release = Event()
    second_done = Event()
    calls = []
    indexed = set()

    def blocking_updater(event_id, paths, rebuild, show_progress):
        calls.append(([Path(path) for path in paths], rebuild))
        if len(calls) == 1:
            started.set()
            assert release.wait(timeout=2)
        indexed.add(event_id)
        if len(calls) == 2:
            second_done.set()
        return object(), [
            IndexBuildOutcome(Path(path), "indexed", face_count=1) for path in paths
        ]

    service = IndexingService(
        events_dir=events_dir,
        database_path=tmp_path / "status.sqlite3",
        index_updater=blocking_updater,
        index_exists=lambda event_id: event_id in indexed,
    )
    service.register_event("graduation", "2026-07-01")
    service.start()
    try:
        service.request_index("graduation")
        assert started.wait(timeout=1)
        second = _add_photo(events_dir, "graduation", "second.jpg")
        assert service.request_index("graduation") is False
        release.set()
        assert second_done.wait(timeout=2)
    finally:
        release.set()
        service.shutdown()

    assert calls == [([first], True), ([second], False)]


def test_index_generations_append_and_load_atomically(tmp_path, monkeypatch):
    event_root = tmp_path / "events" / "graduation"
    first = _add_photo(tmp_path / "events", "graduation", "first.jpg")
    second = _add_photo(tmp_path / "events", "graduation", "second.jpg")
    embeddings = {
        "first.jpg": np.eye(2, 512, dtype=np.float32)[0],
        "second.jpg": np.eye(2, 512, dtype=np.float32)[1],
    }

    monkeypatch.setattr(
        "src.indexing.build_index.event_dir", lambda event_id: event_root
    )
    monkeypatch.setattr(
        "src.indexing.build_index.cv2.imread", lambda path: np.zeros((8, 8, 3), dtype=np.uint8)
    )
    monkeypatch.setattr("src.indexing.build_index.preprocess_image", lambda image: image)

    current_name = {"value": "first.jpg"}

    def fake_detect(image):
        return [_FakeFace((0, 0, 4, 4), 0.99, embeddings[current_name["value"]])]

    monkeypatch.setattr("src.indexing.build_index.detect_and_embed", fake_detect)

    first_index, _ = update_event_index("graduation", [first], rebuild=True)
    current_name["value"] = "second.jpg"
    second_index, _ = update_event_index("graduation", [second], rebuild=False)
    update_event_index("graduation", [second], rebuild=False)
    loaded = load_event_index("graduation")

    assert len(first_index) == 1
    assert len(second_index) == 2
    assert len(loaded) == 3
    assert (event_root / "indexed" / "active.json").exists()
    assert len(list((event_root / "indexed" / "generations").iterdir())) == 2
