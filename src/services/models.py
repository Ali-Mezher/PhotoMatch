"""Data models shared by indexing services and future application layers."""

from dataclasses import dataclass
from enum import StrEnum


class IndexStatus(StrEnum):
    """Lifecycle states used for events and source images."""

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
    event_id: str = ""
    fingerprint: str = ""
    face_count: int = 0
    error: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class EventSummary:
    event_id: str
    status: IndexStatus
    event_date: str = ""
    rebuild_required: bool = False
    total_images: int = 0
    indexed_images: int = 0
    no_face_images: int = 0
    failed_images: int = 0
    pending_images: int = 0
    error: str | None = None
    updated_at: str | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class SearchResult:
    confident: list
    possible: list


@dataclass(frozen=True)
class ImageIndexOutcome:
    photo_path: str
    status: IndexStatus
    face_count: int = 0
    error: str | None = None
