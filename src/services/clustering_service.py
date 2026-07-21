"""On-demand identity clustering over one immutable event index."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from src.indexing.build_index import load_event_index_snapshot

from .admin_store import AdminStore, ClusterRun


class ClusteringService:
    def __init__(self, store: AdminStore, snapshot_loader=load_event_index_snapshot):
        self.store = store
        self._snapshot_loader = snapshot_loader

    def request(self, event_id: str) -> ClusterRun:
        settings = self.store.get_settings()
        return self.store.queue_cluster(
            event_id, settings.cluster_similarity, settings.min_cluster_size
        )

    def run_pending(self) -> None:
        while self.run_next():
            pass

    def run_next(self) -> bool:
        run = self.store.claim_next_cluster()
        if run is None:
            return False
        self._run(run)
        return True

    def _run(self, run: ClusterRun) -> None:
        try:
            generation, event_index = self._snapshot_loader(run.event_id)
            vectors = event_index.index.reconstruct_n(0, len(event_index))
            vectors = np.asarray(vectors, dtype=np.float32)
            if not len(vectors):
                labels = np.empty(0, dtype=np.int64)
            else:
                labels = DBSCAN(
                    eps=1.0 - run.similarity,
                    min_samples=run.min_cluster_size,
                    metric="cosine",
                ).fit_predict(vectors)
            assignments = [
                (
                    index,
                    int(labels[index]),
                    metadata.photo_path,
                    tuple(metadata.bbox),
                    float(metadata.confidence),
                )
                for index, metadata in enumerate(event_index.metadata)
            ]
            self.store.complete_cluster(run.run_id, generation, assignments)
        except Exception as exc:  # noqa: BLE001 - persist one failed run
            self.store.fail_cluster(run.run_id, str(exc))
