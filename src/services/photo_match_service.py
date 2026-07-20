"""Application boundary shared by the desktop UI and a future Flask UI."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from config import (
    DEFAULT_INDEX_WORKERS,
    EVENT_INDEXED_SUBDIR,
    EVENT_RAW_SUBDIR,
    EVENTS_DIR,
    PROJECT_ROOT,
)
from src.indexing import build_event_index
from src.matching import match_selfie

from .indexing_manager import IndexingManager
from .models import EventSummary, ImageIndexStatus, IndexStatus, SearchResult
from .status_store import StatusStore

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PIPELINE_VERSION = "indexing-service-v1"


class PhotoMatchService:
    """Framework-independent event indexing and selfie-search use cases."""

    def __init__(
        self,
        events_dir: Path = EVENTS_DIR,
        database_path: Path = PROJECT_ROOT / "data" / "indexing_status.sqlite3",
        max_index_workers: int = DEFAULT_INDEX_WORKERS,
        index_builder=build_event_index,
        matcher=match_selfie,
    ):
        self.events_dir = Path(events_dir)
        self.store = StatusStore(database_path)
        self._index_builder = index_builder
        self._matcher = matcher
        self._manager = IndexingManager(self._run_index_job, max_workers=max_index_workers)

    def start(self) -> None:
        """Scan quickly, then queue only missing, changed, or failed events."""
        self.events_dir.mkdir(parents=True, exist_ok=True)
        for event_id in self.discover_events():
            if self.reconcile_event(event_id):
                self.queue_index(event_id)

    def shutdown(self) -> None:
        self._manager.shutdown(wait=False)

    def discover_events(self) -> list[str]:
        if not self.events_dir.exists():
            return []
        return sorted(path.name for path in self.events_dir.iterdir() if path.is_dir())

    def list_events(self) -> list[EventSummary]:
        return self.store.list_events()

    def get_event(self, event_id: str) -> EventSummary | None:
        return self.store.get_event(event_id)

    def list_image_statuses(self, event_id: str) -> list[ImageIndexStatus]:
        return self.store.list_images(event_id)

    def reconcile_event(self, event_id: str, force: bool = False) -> bool:
        """Record an event's source inventory and return whether it needs indexing."""
        files = self._event_images(event_id)
        fingerprint = self._event_fingerprint(files)
        current = self.store.get_event(event_id)
        index_dir = self.events_dir / event_id / EVENT_INDEXED_SUBDIR
        has_index = (index_dir / "faces.faiss").exists() and (index_dir / "metadata.json").exists()
        needs_index = force or current is None
        if current is not None:
            needs_index = needs_index or current.status is not IndexStatus.INDEXED
            needs_index = needs_index or self.store.get_fingerprint(event_id) != fingerprint or not has_index

        if needs_index:
            self.store.upsert_event(event_id, fingerprint, IndexStatus.PENDING, len(files))
            paths = set()
            for path in files:
                path_string = str(path)
                paths.add(path_string)
                self.store.upsert_image(event_id, path_string, self._file_fingerprint(path), IndexStatus.PENDING)
            self.store.remove_missing_images(event_id, paths)
        return needs_index

    def queue_index(self, event_id: str, force: bool = False) -> bool:
        if not self.reconcile_event(event_id, force=force):
            return False
        event = self.store.get_event(event_id)
        if event is None:
            return False
        self.store.upsert_event(event_id, self.store.get_fingerprint(event_id) or "", IndexStatus.QUEUED, event.total_images)
        return self._manager.queue(event_id)

    def search_selfie(self, selfie_image, event_id: str, top_k: int = 200) -> SearchResult:
        results = self._matcher(selfie_image, event_id, top_k=top_k)
        return SearchResult(confident=results["confident"], possible=results["possible"])

    def _run_index_job(self, event_id: str) -> None:
        event = self.store.get_event(event_id)
        if event is None:
            return
        fingerprint = self.store.get_fingerprint(event_id) or ""
        self.store.upsert_event(event_id, fingerprint, IndexStatus.INDEXING, event.total_images)
        try:
            self._index_builder(event_id, show_progress=False, progress_callback=self._record_progress)
        except Exception as exc:  # one failed event must not stop queued events
            self.store.upsert_event(event_id, fingerprint, IndexStatus.FAILED, event.total_images, str(exc))
            return
        self.store.upsert_event(event_id, fingerprint, IndexStatus.INDEXED, event.total_images)

    def _record_progress(self, photo_path: Path, status: str, face_count: int = 0, error: str | None = None) -> None:
        event_id = photo_path.parents[1].name
        statuses = {"indexed": IndexStatus.INDEXED, "no_face": IndexStatus.NO_FACE, "failed": IndexStatus.FAILED}
        self.store.upsert_image(event_id, str(photo_path), self._file_fingerprint(photo_path), statuses[status], face_count, error)

    def _event_images(self, event_id: str) -> list[Path]:
        raw_dir = self.events_dir / event_id / EVENT_RAW_SUBDIR
        if not raw_dir.exists():
            return []
        return sorted(path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

    @staticmethod
    def _file_fingerprint(path: Path) -> str:
        stat = path.stat()
        return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"

    def _event_fingerprint(self, files: list[Path]) -> str:
        source = "|".join([PIPELINE_VERSION, *(self._file_fingerprint(path) for path in files)])
        return sha256(source.encode()).hexdigest()
