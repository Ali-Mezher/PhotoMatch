"""Interrupt-driven single-worker execution for indexing jobs."""

from __future__ import annotations

from collections.abc import Callable
from threading import Event, Lock, Thread


class IndexingManager:
    """Wake a single worker on demand; never poll for indexing work."""

    def __init__(self, drain_queue: Callable[[], None]):
        self._drain_queue = drain_queue
        self._wake = Event()
        self._stop = Event()
        self._lifecycle_lock = Lock()
        self._thread: Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self.running:
                return
            self._stop.clear()
            self._thread = Thread(
                target=self._run,
                name="photomatch-indexer",
                daemon=True,
            )
            self._thread.start()

    def signal(self) -> None:
        """Interrupt the idle worker after queue state has been persisted."""
        self._wake.set()

    def shutdown(self, wait: bool = True) -> None:
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return
            self._stop.set()
            self._wake.set()
        if wait:
            thread.join()
        with self._lifecycle_lock:
            if self._thread is thread and not thread.is_alive():
                self._thread = None

    def _run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                return
            self._drain_queue()
