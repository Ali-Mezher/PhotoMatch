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
    # Operator-driven holds. Both keep whatever progress has already been
    # persisted; ``paused`` is resumed to continue, ``stopped`` is a deliberate
    # halt. Neither is claimed by the worker until explicitly resumed.
    PAUSED = "paused"
    STOPPED = "stopped"


# Event-level states an operator can place a hold on, and the held states
# themselves. Kept here so the store, service, and web layers agree.
HELD_STATUSES = frozenset({IndexStatus.PAUSED, IndexStatus.STOPPED})
HOLDABLE_STATUSES = frozenset(
    {IndexStatus.PENDING, IndexStatus.QUEUED, IndexStatus.INDEXING}
)


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


@dataclass(frozen=True)
class IndexProgress:
    event_id: str
    status: IndexStatus
    completed: int
    total: int

    @property
    def percent(self) -> int:
        if self.status is IndexStatus.INDEXED:
            return 100
        if self.total <= 0:
            return 0
        return min(100, int(self.completed * 100 / self.total))


@dataclass(frozen=True)
class SearchResult:
    """Interface-neutral wrapper around the two public match tiers."""

    confident: list
    possible: list
