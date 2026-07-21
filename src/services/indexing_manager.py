"""Indexing execution helpers for event-driven and bounded queue flows."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock, Thread

from config import MAX_INDEX_WORKERS


class IndexingManager:
    """Supports both interrupt-driven draining and bounded per-event queuing."""

    def __init__(
        self,
        worker: Callable[..., None],
        max_workers: int | None = None,
    ):
        self._worker = worker
        self._lifecycle_lock = Lock()
        self._jobs_lock = Lock()
        self._jobs: dict[str, Future] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._wake = Event()
        self._stop = Event()
        self._thread: Thread | None = None
        self._queue_mode = max_workers is not None

        if self._queue_mode:
            if not 1 <= max_workers <= MAX_INDEX_WORKERS:
                raise ValueError(f"max_workers must be between 1 and {MAX_INDEX_WORKERS}")
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="photomatch-index",
            )

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def queue(self, event_id: str) -> bool:
        if not self._queue_mode or self._executor is None:
            raise RuntimeError("queue() is only available when max_workers is configured")
        with self._jobs_lock:
            current = self._jobs.get(event_id)
            if current is not None and not current.done():
                return False
            self._jobs[event_id] = self._executor.submit(self._worker, event_id)
            return True

    def start(self) -> None:
        if self._queue_mode:
            return
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
        if self._queue_mode:
            return
        self._wake.set()

    def shutdown(self, wait: bool = True) -> None:
        if self._queue_mode:
            if self._executor is not None:
                self._executor.shutdown(wait=wait, cancel_futures=True)
            return
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
            self._worker()
