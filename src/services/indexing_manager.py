"""Bounded background scheduler for event indexing."""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Callable

from config import MAX_INDEX_WORKERS


class IndexingManager:
    def __init__(self, worker: Callable[[str], None], max_workers: int = 1):
        if not 1 <= max_workers <= MAX_INDEX_WORKERS:
            raise ValueError(f"max_workers must be between 1 and {MAX_INDEX_WORKERS}")
        self._worker = worker
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="photomatch-index")
        self._lock = Lock()
        self._jobs: dict[str, Future] = {}

    def queue(self, event_id: str) -> bool:
        with self._lock:
            current = self._jobs.get(event_id)
            if current is not None and not current.done():
                return False
            self._jobs[event_id] = self._executor.submit(self._worker, event_id)
            return True

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
