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
    event_id: str
    photo_path: str
    fingerprint: str
    status: IndexStatus
    face_count: int = 0
    error: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class EventSummary:
    event_id: str
    event_date: str
    status: IndexStatus
    rebuild_required: bool
    total_images: int
    indexed_images: int
    no_face_images: int
    failed_images: int
    pending_images: int
    error: str | None = None
    updated_at: str | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class ImageIndexOutcome:
    photo_path: str
    status: IndexStatus
    face_count: int = 0
    error: str | None = None
