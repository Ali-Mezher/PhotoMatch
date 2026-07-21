"""Framework-independent application boundary shared by desktop and Flask."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from config import DEFAULT_INDEX_WORKERS, EVENT_RAW_SUBDIR, EVENTS_DIR, INDEX_STATUS_DB
from src.matching import match_selfie

from .indexing_service import IndexingService
from .models import EventSummary, ImageIndexStatus, SearchResult


class PhotoMatchService:
    """Compatibility facade over the authoritative incremental index service.

    Issue #23 introduced this interface for Tkinter. The Flask work subsequently
    added the safer per-image incremental implementation. Keeping this thin
    facade lets both interfaces share that implementation instead of maintaining
    two schedulers and two incompatible SQLite schemas.
    """

    def __init__(
        self,
        events_dir: Path = EVENTS_DIR,
        database_path: Path = INDEX_STATUS_DB,
        max_index_workers: int = DEFAULT_INDEX_WORKERS,
        matcher=match_selfie,
        indexing_service: IndexingService | None = None,
    ):
        if max_index_workers != 1:
            raise ValueError(
                "PhotoMatch serializes indexing and clustering; "
                "max_index_workers must be 1"
            )
        self.events_dir = Path(events_dir).resolve()
        self.indexing = indexing_service or IndexingService(
            events_dir=self.events_dir,
            database_path=database_path,
        )
        self.store = self.indexing.store
        self._matcher = matcher

    def start(self) -> None:
        """Start the worker and adopt legacy event folders into SQL once."""
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.indexing.start()
        for event_id in self.discover_events():
            if self.store.get_event(event_id) is None:
                self._register_discovered_event(event_id)
            if self.indexing.reconcile_event(event_id):
                self.indexing.request_index(event_id)

    def shutdown(self) -> None:
        self.indexing.shutdown(wait=False)

    def discover_events(self) -> list[str]:
        if not self.events_dir.exists():
            return []
        return sorted(
            path.name
            for path in self.events_dir.iterdir()
            if path.is_dir() and (path / EVENT_RAW_SUBDIR).is_dir()
        )

    def list_events(self) -> list[EventSummary]:
        return self.indexing.list_events()

    def get_event(self, event_id: str) -> EventSummary | None:
        try:
            return self.indexing.get_event(event_id)
        except (KeyError, ValueError):
            return None

    def list_image_statuses(self, event_id: str) -> list[ImageIndexStatus]:
        return self.indexing.list_image_statuses(event_id)

    def reconcile_event(self, event_id: str, force: bool = False) -> bool:
        if self.store.get_event(event_id) is None:
            self._register_discovered_event(event_id)
        if force:
            self.store.require_rebuild(event_id)
        return self.indexing.reconcile_event(event_id)

    def queue_index(self, event_id: str, force: bool = False) -> bool:
        if self.store.get_event(event_id) is None:
            self._register_discovered_event(event_id)
        return self.indexing.request_index(event_id, force_rebuild=force)

    def search_selfie(
        self, selfie_image, event_id: str, top_k: int = 200
    ) -> SearchResult:
        results = self._matcher(selfie_image, event_id, top_k=top_k)
        return SearchResult(
            confident=results["confident"], possible=results["possible"]
        )

    def _register_discovered_event(self, event_id: str) -> EventSummary:
        raw_dir = self.events_dir / event_id / EVENT_RAW_SUBDIR
        if not raw_dir.exists():
            raise FileNotFoundError(f"Event raw directory does not exist: {raw_dir}")
        # Legacy folders do not carry an explicit date. Their raw-directory
        # modification date is deterministic and preserves oldest-first queueing.
        event_date: date = datetime.fromtimestamp(raw_dir.stat().st_mtime).date()
        return self.indexing.register_event(
            event_id,
            event_date,
            display_name=event_id,
        )
