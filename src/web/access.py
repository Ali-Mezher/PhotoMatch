"""Short-lived event unlock grants and lightweight brute-force throttling."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class EventAccessGrant:
    token: str
    event_id: str
    expires_at: float


class EventAccessGate:
    """Keep public event grants in memory and throttle failed code attempts."""

    def __init__(
        self,
        ttl_seconds: int = 30 * 60,
        max_failures: int = 6,
        failure_window_seconds: int = 60,
        clock=time.monotonic,
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_failures <= 0 or failure_window_seconds <= 0:
            raise ValueError("failure limits must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_failures = max_failures
        self.failure_window_seconds = failure_window_seconds
        self._clock = clock
        self._lock = Lock()
        self._grants: dict[str, EventAccessGrant] = {}
        self._failures: dict[str, list[float]] = {}

    def grant(self, event_id: str) -> EventAccessGrant:
        now = self._clock()
        grant = EventAccessGrant(
            token=secrets.token_urlsafe(24),
            event_id=event_id,
            expires_at=now + self.ttl_seconds,
        )
        with self._lock:
            self._purge_expired_grants(now)
            self._grants[grant.token] = grant
        return grant

    def resolve(self, token: str) -> EventAccessGrant | None:
        now = self._clock()
        with self._lock:
            self._purge_expired_grants(now)
            return self._grants.get(token)

    def can_attempt(self, client_key: str) -> bool:
        now = self._clock()
        with self._lock:
            failures = self._recent_failures(client_key, now)
            return len(failures) < self.max_failures

    def record_failure(self, client_key: str) -> None:
        now = self._clock()
        with self._lock:
            failures = self._recent_failures(client_key, now)
            failures.append(now)
            self._failures[client_key] = failures

    def clear_failures(self, client_key: str) -> None:
        with self._lock:
            self._failures.pop(client_key, None)

    def _recent_failures(self, client_key: str, now: float) -> list[float]:
        cutoff = now - self.failure_window_seconds
        failures = [
            attempted_at
            for attempted_at in self._failures.get(client_key, [])
            if attempted_at > cutoff
        ]
        if failures:
            self._failures[client_key] = failures
        else:
            self._failures.pop(client_key, None)
        return failures

    def _purge_expired_grants(self, now: float) -> None:
        expired = [
            token
            for token, grant in self._grants.items()
            if grant.expires_at <= now
        ]
        for token in expired:
            del self._grants[token]
