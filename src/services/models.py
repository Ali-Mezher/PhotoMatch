"""Framework-independent models used by PhotoMatch application services."""

from dataclasses import dataclass
from enum import StrEnum


class IndexStatus(StrEnum):
    """Lifecycle states for an event or one source image."""

    PENDING = "pending"
    QUEUED = "queued"
    INDEXING = "indexing"
    INDEXED = "indexed"
    NO_FACE = "no_face"
    FAILED = "failed"


@dataclass(frozen=True)
class ImageIndexStatus:
    photo_path: str
    status: IndexStatus
    face_count: int = 0
    error: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class EventSummary:
    event_id: str
    status: IndexStatus
    total_images: int
    indexed_images: int
    no_face_images: int
    failed_images: int
    updated_at: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class SearchResult:
    confident: list
    possible: list
