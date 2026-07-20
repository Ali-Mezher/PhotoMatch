"""Framework-independent orchestration for incremental event indexing."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from config import (
    EVENTS_DIR,
    EVENT_RAW_SUBDIR,
    INDEX_PIPELINE_VERSION,
    INDEX_STATUS_DB,
    validate_event_id,
)
from src.indexing import event_index_exists, load_event_index, update_event_index

from .indexing_manager import IndexingManager
from .models import EventSummary, ImageIndexOutcome, ImageIndexStatus, IndexStatus
from .status_store import StatusStore

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class IndexingService:
    """Track source inventory and run the minimum safe indexing work."""

    def __init__(
        self,
        events_dir: Path = EVENTS_DIR,
        database_path: Path = INDEX_STATUS_DB,
        index_updater=update_event_index,
        index_exists=event_index_exists,
        index_loader=load_event_index,
    ):
        self.events_dir = Path(events_dir).resolve()
        self.store = StatusStore(database_path)
        self._index_updater = index_updater
        self._index_exists = index_exists
        self._index_loader = index_loader
        self._manager = IndexingManager(lambda: self._drain_queue(False))

    def start(self) -> None:
        """Start the worker and resume jobs interrupted by a prior shutdown."""
        self._manager.start()
        if self.store.recover_interrupted_events():
            self._manager.signal()

    def shutdown(self, wait: bool = True) -> None:
        self._manager.shutdown(wait=wait)

    def register_event(self, event_id: str, event_date: str | date) -> EventSummary:
        event_id = validate_event_id(event_id)
        normalized_date = self._normalize_date(event_date)
        raw_dir = self._raw_dir(event_id)
        if not raw_dir.exists():
            raise FileNotFoundError(f"Event raw directory does not exist: {raw_dir}")
        self.store.register_event(event_id, normalized_date)
        self.reconcile_event(event_id)
        return self.get_event(event_id)

    def request_index(self, event_id: str, force_rebuild: bool = False) -> bool:
        """Persist an event request and interrupt the worker immediately."""
        event_id = self._require_registered_event(event_id)
        self.reconcile_event(event_id)
        if force_rebuild:
            self.store.require_rebuild(event_id)

        event = self.get_event(event_id)
        needs_work = (
            force_rebuild
            or event.rebuild_required
            or event.pending_images > 0
            or not self._event_index_exists(event_id)
        )
        if not needs_work:
            return False

        queued = self.store.queue_event(event_id)
        # A request received while this event is indexing is picked up by the
        # post-job reconciliation. Signaling is harmless and avoids a missed
        # wake-up if the active job is just completing.
        self._manager.signal()
        return queued

    def force_rebuild(self, event_id: str) -> bool:
        return self.request_index(event_id, force_rebuild=True)

    def retry_failed(self, event_id: str) -> int:
        event_id = self._require_registered_event(event_id)
        count = self.store.retry_failed_images(event_id)
        if count:
            self.request_index(event_id)
        return count

    def reconcile_event(self, event_id: str) -> bool:
        """Refresh one event's SQL inventory without scheduling by itself."""
        event_id = self._require_registered_event(event_id)
        current_files = self._event_images(event_id)
        stored = self.store.get_image_map(event_id)
        adopting_existing_index = not stored and self._event_index_exists(event_id)
        if adopting_existing_index and self._adopt_existing_index(
            event_id, current_files
        ):
            stored = self.store.get_image_map(event_id)
            adopting_existing_index = False
        current_paths = {str(path): path for path in current_files}
        changed_or_removed = False

        for photo_path_string, photo_path in current_paths.items():
            fingerprint = self._file_fingerprint(photo_path)
            existing = stored.get(photo_path_string)
            if existing is None:
                self.store.upsert_pending_image(
                    event_id, photo_path_string, fingerprint
                )
            elif existing.fingerprint != fingerprint:
                self.store.upsert_pending_image(
                    event_id, photo_path_string, fingerprint
                )
                changed_or_removed = True

        removed = sorted(set(stored) - set(current_paths))
        if removed:
            self.store.remove_images(event_id, removed)
            changed_or_removed = True

        version_changed = self.store.pipeline_version(event_id) not in {
            "",
            INDEX_PIPELINE_VERSION,
        }
        if changed_or_removed or version_changed or adopting_existing_index:
            self.store.require_rebuild(event_id)

        event = self.get_event(event_id)
        return (
            event.rebuild_required
            or event.pending_images > 0
            or not self._event_index_exists(event_id)
        )

    def list_events(self) -> list[EventSummary]:
        return self.store.list_events()

    def get_event(self, event_id: str) -> EventSummary:
        event_id = validate_event_id(event_id)
        event = self.store.get_event(event_id)
        if event is None:
            raise KeyError(f"Unknown event: {event_id}")
        return event

    def get_event_access_code(self, event_id: str) -> str:
        """Return the attendee access code assigned when the event was registered."""
        event_id = self._require_registered_event(event_id)
        access_code = self.store.get_event_access_code(event_id)
        if access_code is None:
            raise RuntimeError(f"Event '{event_id}' has no access code")
        return access_code

    def list_image_statuses(self, event_id: str) -> list[ImageIndexStatus]:
        event_id = self._require_registered_event(event_id)
        return self.store.list_images(event_id)

    def run_pending(self, show_progress: bool = False) -> None:
        """Synchronously drain queued jobs, primarily for CLI operation."""
        self._drain_queue(show_progress)

    def _drain_queue(self, show_progress: bool = False) -> None:
        while True:
            job = self.store.claim_next_event()
            if job is None:
                return
            event_id, rebuild_requested = job
            self._run_index_job(event_id, rebuild_requested, show_progress)

    def _run_index_job(
        self, event_id: str, rebuild_requested: bool, show_progress: bool = False
    ) -> None:
        rebuild = rebuild_requested or not self._event_index_exists(event_id)
        paths = [
            Path(path) for path in self.store.paths_for_processing(event_id, rebuild)
        ]

        try:
            _, outcomes = self._index_updater(
                event_id,
                paths,
                rebuild=rebuild,
                show_progress=show_progress,
            )
            self.store.record_outcomes(
                event_id,
                [
                    ImageIndexOutcome(
                        photo_path=str(outcome.photo_path),
                        status=IndexStatus(outcome.status),
                        face_count=outcome.face_count,
                        error=outcome.error,
                    )
                    for outcome in outcomes
                ],
            )
            self.store.complete_event(event_id, INDEX_PIPELINE_VERSION)
        except Exception as exc:  # noqa: BLE001 - one event must not stop the queue
            self.store.fail_event(event_id, str(exc))
            return

        # Capture files added or modified while this job was running. This is
        # event-driven: the reconciliation happens at the state transition,
        # not on a timer.
        if self.reconcile_event(event_id):
            self.store.queue_event(event_id)
            self._manager.signal()

    def _event_images(self, event_id: str) -> list[Path]:
        raw_dir = self._raw_dir(event_id)
        if not raw_dir.exists():
            return []
        return sorted(
            path.resolve()
            for path in raw_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _adopt_existing_index(
        self, event_id: str, current_files: list[Path]
    ) -> bool:
        """Seed SQL state from a legacy full index without reprocessing photos."""
        try:
            index = self._index_loader(event_id)
        except Exception:  # noqa: BLE001 - an unreadable legacy index is rebuilt
            return False

        faces_per_photo: dict[str, int] = {}
        for metadata in index.metadata:
            resolved_path = str(Path(metadata.photo_path).resolve())
            faces_per_photo[resolved_path] = faces_per_photo.get(resolved_path, 0) + 1

        for photo_path in current_files:
            photo_path_string = str(photo_path)
            face_count = faces_per_photo.get(photo_path_string, 0)
            status = IndexStatus.INDEXED if face_count else IndexStatus.NO_FACE
            self.store.upsert_completed_image(
                event_id,
                photo_path_string,
                self._file_fingerprint(photo_path),
                status,
                face_count,
            )
        self.store.complete_event(event_id, INDEX_PIPELINE_VERSION)
        return True

    def _raw_dir(self, event_id: str) -> Path:
        event_root = (self.events_dir / validate_event_id(event_id)).resolve()
        if event_root.parent != self.events_dir:
            raise ValueError("event_id resolves outside the configured events directory")
        return event_root / EVENT_RAW_SUBDIR

    def _require_registered_event(self, event_id: str) -> str:
        event_id = validate_event_id(event_id)
        if self.store.get_event(event_id) is None:
            raise KeyError(f"Unknown event: {event_id}")
        return event_id

    def _event_index_exists(self, event_id: str) -> bool:
        # The production helper uses config.EVENTS_DIR. Tests and future
        # alternate storage roots can inject a root-aware implementation.
        return bool(self._index_exists(event_id))

    @staticmethod
    def _file_fingerprint(path: Path) -> str:
        stat = path.stat()
        return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"

    @staticmethod
    def _normalize_date(value: str | date) -> str:
        if isinstance(value, date):
            return value.isoformat()
        try:
            return date.fromisoformat(value).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError("event_date must be an ISO date in YYYY-MM-DD format") from exc
