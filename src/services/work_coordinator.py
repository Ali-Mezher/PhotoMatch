"""One interrupt-driven worker for indexing and clustering jobs."""

from __future__ import annotations

from .admin_store import AdminStore
from .clustering_service import ClusteringService
from .indexing_manager import IndexingManager
from .indexing_service import IndexingService


class AdminWorkCoordinator:
    """Serialize expensive jobs and always drain indexing before clustering."""

    def __init__(
        self,
        indexing: IndexingService,
        clustering: ClusteringService,
        admin_store: AdminStore,
    ):
        self.indexing = indexing
        self.clustering = clustering
        self.admin_store = admin_store
        self._manager = IndexingManager(self._drain)

    def start(self) -> None:
        self._manager.start()
        indexing_waiting = bool(self.indexing.store.recover_interrupted_events())
        indexing_waiting = bool(
            self.indexing.reconcile_registered_events()
        ) or indexing_waiting
        clustering_waiting = self.admin_store.recover_interrupted_clusters()
        if indexing_waiting or clustering_waiting:
            self.signal()

    def signal(self) -> None:
        self._manager.signal()

    def shutdown(self, wait: bool = True) -> None:
        self._manager.shutdown(wait=wait)

    def _drain(self) -> None:
        while True:
            # A new index request arriving during a cluster run gets priority
            # before another cluster is claimed.
            self.indexing.run_pending(show_progress=False)
            if not self.clustering.run_next():
                return
