"""Short-lived, server-side search result storage for the public web app."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


@dataclass(frozen=True)
class StoredPhoto:
    photo_id: str
    path: Path
    score: float
    tier: str


@dataclass(frozen=True)
class StoredSearch:
    token: str
    event_id: str
    photos: dict[str, StoredPhoto]
    created_at: float
    expires_at: float

    @property
    def confident(self) -> list[StoredPhoto]:
        return [photo for photo in self.photos.values() if photo.tier == "confident"]

    @property
    def possible(self) -> list[StoredPhoto]:
        return [photo for photo in self.photos.values() if photo.tier == "possible"]


class SearchResultStore:
    """Thread-safe in-memory store; no selfies or embeddings are retained."""

    def __init__(self, ttl_seconds: int = 15 * 60, clock=time.monotonic):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = Lock()
        self._searches: dict[str, StoredSearch] = {}

    def create(self, event_id: str, matches: dict) -> StoredSearch:
        now = self._clock()
        token = secrets.token_urlsafe(24)
        photos: dict[str, StoredPhoto] = {}
        seen_paths: set[Path] = set()

        for tier in ("confident", "possible"):
            for match in matches.get(tier, []):
                path = Path(match.photo_path).resolve()
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                photo_id = secrets.token_urlsafe(9)
                photos[photo_id] = StoredPhoto(
                    photo_id=photo_id,
                    path=path,
                    score=float(match.score),
                    tier=tier,
                )

        search = StoredSearch(
            token=token,
            event_id=event_id,
            photos=photos,
            created_at=now,
            expires_at=now + self.ttl_seconds,
        )
        with self._lock:
            self._purge_expired(now)
            self._searches[token] = search
        return search

    def get(self, token: str) -> StoredSearch | None:
        now = self._clock()
        with self._lock:
            self._purge_expired(now)
            return self._searches.get(token)

    def _purge_expired(self, now: float) -> None:
        expired = [
            token
            for token, search in self._searches.items()
            if search.expires_at <= now
        ]
        for token in expired:
            del self._searches[token]
