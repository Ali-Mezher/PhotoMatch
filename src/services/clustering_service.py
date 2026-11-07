"""On-demand identity clustering over one immutable event index."""

from __future__ import annotations

import numpy as np

from src.clustering import cluster_index
from src.indexing.build_index import load_event_index_snapshot

from .admin_store import AdminStore, ClusterRun


class ClusteringService:
    def __init__(self, store: AdminStore, snapshot_loader=load_event_index_snapshot):
        self.store = store
        self._snapshot_loader = snapshot_loader

    def request(self, event_id: str) -> ClusterRun:
        settings = self.store.get_settings()
        return self.store.queue_cluster(
            event_id,
            settings.cluster_similarity,
            settings.min_cluster_size,
            settings.cluster_cohesion_similarity,
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
            result = cluster_index(
                event_index,
                edge_similarity=run.similarity,
                cohesion_similarity=run.cohesion_similarity,
                min_cluster_size=run.min_cluster_size,
            )
            labels = np.full(len(event_index), -1, dtype=np.int64)
            for cluster_label, cluster in enumerate(result.clusters):
                for member in cluster.members:
                    labels[member.face_index] = cluster_label
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
