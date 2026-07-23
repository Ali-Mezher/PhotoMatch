"""In-memory cooperative-cancellation signalling for the indexing worker.

The SQL store is the durable authority for event state, but a job already
running inside the worker cannot be interrupted by a database write alone: the
photo loop lives in ``update_event_index`` and never re-reads SQL mid-run. This
small thread-safe registry lets the service ask the running loop to stop after
the current photo, so already-processed photos are published and kept.
"""

from __future__ import annotations

from enum import StrEnum
from threading import Lock


class ControlAction(StrEnum):
    """What an operator asked an in-flight (or queued) job to do."""

    PAUSE = "pause"
    STOP = "stop"


class IndexControl:
    """Track pause/stop requests and whether the worker has honored them.

    A request is *pending* once ``request`` is called. When the worker's photo
    loop calls :meth:`should_stop` and sees a pending request, it is recorded as
    *triggered* so the post-run code can tell a genuine interruption apart from
    a request that arrived after the job had already finished.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._requested: dict[str, ControlAction] = {}
        self._triggered: dict[str, ControlAction] = {}

    def request(self, event_id: str, action: ControlAction) -> None:
        with self._lock:
            self._requested[event_id] = action

    def should_stop(self, event_id: str) -> ControlAction | None:
        """Return the pending action, recording it as honored if present.

        Called between photos by the running job. Returning a non-``None``
        action is the loop's signal to break.
        """
        with self._lock:
            action = self._requested.get(event_id)
            if action is not None:
                self._triggered[event_id] = action
            return action

    def triggered(self, event_id: str) -> ControlAction | None:
        """Return the action the running loop actually acted on, if any."""
        with self._lock:
            return self._triggered.get(event_id)

    def clear(self, event_id: str) -> None:
        with self._lock:
            self._requested.pop(event_id, None)
            self._triggered.pop(event_id, None)
