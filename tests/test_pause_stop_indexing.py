"""Operator pause / resume / stop controls for incremental indexing.

These exercise the guarantee that halting an event keeps the progress already
made: photos indexed before the halt stay indexed, and resuming processes only
the remainder rather than starting over.
"""

from __future__ import annotations

from pathlib import Path
from threading import Event

import numpy as np

from src.indexing import IndexBuildOutcome, load_event_index, update_event_index
from src.services import IndexStatus, IndexingService


def _add_photo(events_dir: Path, event_id: str, name: str, content=b"photo") -> Path:
    raw_dir = events_dir / event_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    photo = raw_dir / name
    photo.write_bytes(content)
    return photo.resolve()


class _ControllableUpdater:
    """Mimic ``update_event_index``'s per-photo loop over an injected service.

    It honors ``should_continue`` between photos and records each processed
    photo through ``progress_callback``, exactly like the real photo loop, and
    optionally triggers an operator hold after a chosen number of photos.
    """

    def __init__(self):
        self.indexed_events: set[str] = set()
        self.calls: list[tuple[str, list[Path], bool]] = []
        self._service: IndexingService | None = None
        self._hold_after: int | None = None
        self._hold_method: str | None = None

    def bind(self, service: IndexingService) -> None:
        self._service = service

    def hold(self, method: str, after: int) -> None:
        """Ask the updater to call ``service.<method>`` after N photos once."""
        self._hold_method = method
        self._hold_after = after

    def __call__(
        self,
        event_id,
        paths,
        rebuild,
        show_progress,
        progress_callback=None,
        should_continue=None,
    ):
        paths = [Path(path) for path in paths]
        self.calls.append((event_id, paths, rebuild))
        outcomes = []
        for path in paths:
            if should_continue is not None and not should_continue():
                break
            outcomes.append(IndexBuildOutcome(path, "indexed", face_count=1))
            if progress_callback is not None:
                progress_callback(path, "indexed", 1, None)
            if self._hold_after is not None and len(outcomes) == self._hold_after:
                getattr(self._service, self._hold_method)(event_id)
                self._hold_after = None  # fire once
        # The real updater always publishes, so an index exists even after a
        # partial run.
        self.indexed_events.add(event_id)
        return object(), outcomes

    def exists(self, event_id):
        return event_id in self.indexed_events


def _service(tmp_path):
    updater = _ControllableUpdater()
    events_dir = tmp_path / "events"
    service = IndexingService(
        events_dir=events_dir,
        database_path=tmp_path / "status.sqlite3",
        index_updater=updater,
        index_exists=updater.exists,
    )
    updater.bind(service)
    return service, updater, events_dir


def _register_with_photos(service, events_dir, event_id, count):
    for index in range(count):
        _add_photo(events_dir, event_id, f"photo{index}.jpg", f"photo{index}".encode())
    service.register_event(event_id, "2026-07-01")


def test_pause_keeps_indexed_progress_and_parks_the_remainder(tmp_path):
    service, updater, events_dir = _service(tmp_path)
    _register_with_photos(service, events_dir, "graduation", 4)
    updater.hold("pause_index", after=2)

    service.request_index("graduation")
    service.run_pending()

    event = service.get_event("graduation")
    assert event.status is IndexStatus.PAUSED
    assert event.indexed_images == 2
    statuses = {row.status for row in service.list_image_statuses("graduation")}
    assert statuses == {IndexStatus.INDEXED, IndexStatus.PENDING}
    assert event.pending_images == 2


def test_resume_continues_without_reprocessing_indexed_photos(tmp_path):
    service, updater, events_dir = _service(tmp_path)
    _register_with_photos(service, events_dir, "graduation", 4)
    updater.hold("pause_index", after=2)
    service.request_index("graduation")
    service.run_pending()
    paused_indexed = {
        row.photo_path
        for row in service.list_image_statuses("graduation")
        if row.status is IndexStatus.INDEXED
    }

    assert service.resume_index("graduation") is True
    service.run_pending()

    event = service.get_event("graduation")
    assert event.status is IndexStatus.INDEXED
    assert event.indexed_images == 4
    # The resume run only reprocessed the two photos left pending.
    resumed_batch = updater.calls[-1][1]
    assert len(resumed_batch) == 2
    assert paused_indexed.isdisjoint(str(path) for path in resumed_batch)


def test_stop_keeps_progress_and_is_not_auto_requeued(tmp_path):
    service, updater, events_dir = _service(tmp_path)
    _register_with_photos(service, events_dir, "graduation", 4)
    updater.hold("stop_index", after=3)
    service.request_index("graduation")
    service.run_pending()

    event = service.get_event("graduation")
    assert event.status is IndexStatus.STOPPED
    assert event.indexed_images == 3
    assert event.pending_images == 1

    # An automatic index request must respect the deliberate stop.
    calls_before = len(updater.calls)
    assert service.request_index("graduation") is False
    service.run_pending()
    assert len(updater.calls) == calls_before
    assert service.get_event("graduation").status is IndexStatus.STOPPED


def test_stopped_event_can_be_resumed_to_completion(tmp_path):
    service, updater, events_dir = _service(tmp_path)
    _register_with_photos(service, events_dir, "graduation", 4)
    updater.hold("stop_index", after=1)
    service.request_index("graduation")
    service.run_pending()
    assert service.get_event("graduation").status is IndexStatus.STOPPED

    assert service.resume_index("graduation") is True
    service.run_pending()

    event = service.get_event("graduation")
    assert event.status is IndexStatus.INDEXED
    assert event.indexed_images == 4


def test_pause_before_claim_holds_a_queued_event(tmp_path):
    service, updater, events_dir = _service(tmp_path)
    _register_with_photos(service, events_dir, "graduation", 2)
    service.request_index("graduation")  # queued, not yet drained

    assert service.pause_index("graduation") is True

    event = service.get_event("graduation")
    assert event.status is IndexStatus.PAUSED
    # The worker must not pick up a paused event.
    service.run_pending()
    assert updater.calls == []
    assert service.get_event("graduation").status is IndexStatus.PAUSED


def test_pausing_an_indexed_event_is_a_no_op(tmp_path):
    service, updater, events_dir = _service(tmp_path)
    _register_with_photos(service, events_dir, "graduation", 1)
    service.request_index("graduation")
    service.run_pending()
    assert service.get_event("graduation").status is IndexStatus.INDEXED

    assert service.pause_index("graduation") is False
    assert service.resume_index("graduation") is False
    assert service.get_event("graduation").status is IndexStatus.INDEXED


def test_resume_processes_new_photos_added_while_paused(tmp_path):
    service, updater, events_dir = _service(tmp_path)
    _register_with_photos(service, events_dir, "graduation", 3)
    updater.hold("pause_index", after=1)
    service.request_index("graduation")
    service.run_pending()
    assert service.get_event("graduation").status is IndexStatus.PAUSED

    _add_photo(events_dir, "graduation", "late.jpg", b"late")
    # Reconciling a paused event registers the new photo but must not un-hold it.
    service.reconcile_event("graduation")
    assert service.get_event("graduation").status is IndexStatus.PAUSED

    service.resume_index("graduation")
    service.run_pending()

    event = service.get_event("graduation")
    assert event.status is IndexStatus.INDEXED
    assert event.indexed_images == 4


def test_pause_from_another_thread_stops_the_running_worker(tmp_path):
    """An operator pausing on the request thread halts the background worker."""
    events_dir = tmp_path / "events"
    for index in range(4):
        _add_photo(events_dir, "graduation", f"p{index}.jpg", f"p{index}".encode())

    first_photo_started = Event()
    allow_finish = Event()
    indexed: set[str] = set()

    def blocking_updater(
        event_id,
        paths,
        rebuild,
        show_progress,
        progress_callback=None,
        should_continue=None,
    ):
        outcomes = []
        for path in paths:
            if should_continue is not None and not should_continue():
                break
            outcomes.append(IndexBuildOutcome(Path(path), "indexed", face_count=1))
            if progress_callback is not None:
                progress_callback(path, "indexed", 1, None)
            first_photo_started.set()
            # Hold inside the first photo so the test thread can request a pause
            # while the run is genuinely in flight.
            assert allow_finish.wait(timeout=2)
        indexed.add(event_id)
        return object(), outcomes

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
        assert first_photo_started.wait(timeout=2)
        assert service.pause_index("graduation") is True
        allow_finish.set()
        # Let the worker observe the pause and settle.
        for _ in range(200):
            if service.get_event("graduation").status is IndexStatus.PAUSED:
                break
            Event().wait(0.01)
    finally:
        allow_finish.set()
        service.shutdown()

    event = service.get_event("graduation")
    assert event.status is IndexStatus.PAUSED
    assert 1 <= event.indexed_images < 4
    assert event.pending_images >= 1

    assert service.resume_index("graduation") is True
    service.run_pending()
    assert service.get_event("graduation").status is IndexStatus.INDEXED
    assert service.get_event("graduation").indexed_images == 4


def test_update_event_index_publishes_partial_progress_on_stop(tmp_path, monkeypatch):
    """The real photo loop must keep faces added before a stop signal."""
    event_root = tmp_path / "events" / "graduation"
    photos = [
        _add_photo(tmp_path / "events", "graduation", f"p{i}.jpg", f"p{i}".encode())
        for i in range(4)
    ]

    class _Face:
        def __init__(self, embedding):
            self.bbox = (0, 0, 4, 4)
            self.confidence = 0.99
            self.embedding = embedding

    monkeypatch.setattr(
        "src.indexing.build_index.event_dir", lambda event_id: event_root
    )
    monkeypatch.setattr(
        "src.indexing.build_index.cv2.imread",
        lambda path: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "src.indexing.build_index.preprocess_image", lambda image: image
    )
    counter = {"n": 0}

    def fake_detect(image):
        vector = np.eye(4, 512, dtype=np.float32)[counter["n"]]
        counter["n"] += 1
        return [_Face(vector)]

    monkeypatch.setattr("src.indexing.build_index.detect_and_embed", fake_detect)

    processed = {"count": 0}

    def should_continue() -> bool:
        # Allow exactly two photos through, then request a stop.
        return processed["count"] < 2

    seen = []

    def progress(path, status, faces, error):
        seen.append(status)
        processed["count"] += 1

    index, outcomes = update_event_index(
        "graduation",
        photos,
        rebuild=True,
        progress_callback=progress,
        should_continue=should_continue,
    )

    assert len(outcomes) == 2
    assert len(index) == 2
    # The partial index is durably published and reloads with the kept faces.
    assert len(load_event_index("graduation")) == 2
