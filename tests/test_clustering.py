"""Identity clustering tests against small deterministic face vectors."""

from __future__ import annotations

import numpy as np

from src.indexing.faiss_index import EventIndex, IndexedFace
from src.services.admin_store import AdminStore, RuntimeSettings
from src.services.clustering_service import ClusteringService


def test_clustering_persists_groups_noise_and_snapshot_generation(tmp_path):
    index = EventIndex(dim=3)
    index.add(
        [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.99, 0.05, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ],
        [
            IndexedFace("one.jpg", (1, 2, 10, 10), 0.99),
            IndexedFace("two.jpg", (2, 3, 10, 10), 0.98),
            IndexedFace("three.jpg", (3, 4, 10, 10), 0.97),
        ],
    )
    store = AdminStore(tmp_path / "state.sqlite3")
    store.update_settings(RuntimeSettings(cluster_similarity=0.8, min_cluster_size=2))
    service = ClusteringService(store, snapshot_loader=lambda _event: ("generation-1", index))

    requested = service.request("event-one")
    assert requested.status == "queued"
    service.run_pending()

    completed = store.get_cluster_run(requested.run_id)
    members = store.cluster_members(requested.run_id)
    assert completed.status == "complete"
    assert completed.index_generation == "generation-1"
    assert completed.cluster_count == 1
    assert completed.noise_count == 1
    assert [member.cluster_label for member in members].count(-1) == 1


def test_interrupted_cluster_is_requeued(tmp_path):
    store = AdminStore(tmp_path / "state.sqlite3")
    run = store.queue_cluster("event", 0.7, 2)
    claimed = store.claim_next_cluster()
    assert claimed.run_id == run.run_id
    assert claimed.status == "running"

    assert store.recover_interrupted_clusters()
    assert store.get_cluster_run(run.run_id).status == "queued"


def test_cluster_labels_are_optional_and_replaceable(tmp_path):
    store = AdminStore(tmp_path / "state.sqlite3")
    run = store.queue_cluster("event", 0.7, 2)
    store.set_cluster_label(run.run_id, 0, "Speaker")
    assert store.cluster_labels(run.run_id) == {0: "Speaker"}
    store.set_cluster_label(run.run_id, 0, "")
    assert store.cluster_labels(run.run_id) == {}


def test_existing_admin_database_adds_cohesion_columns(tmp_path):
    import sqlite3

    database = tmp_path / "legacy-admin.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE runtime_settings (
                singleton_id INTEGER PRIMARY KEY,
                possible_threshold REAL NOT NULL,
                confident_threshold REAL NOT NULL,
                top_k INTEGER NOT NULL,
                cluster_similarity REAL NOT NULL,
                min_cluster_size INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO runtime_settings VALUES (1, 0.5, 0.65, 200, 0.7, 2, CURRENT_TIMESTAMP);
            CREATE TABLE cluster_runs (
                run_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                status TEXT NOT NULL,
                index_generation TEXT,
                similarity REAL NOT NULL,
                min_cluster_size INTEGER NOT NULL,
                cluster_count INTEGER NOT NULL DEFAULT 0,
                noise_count INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                requested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TEXT,
                completed_at TEXT
            );
            """
        )

    store = AdminStore(database)
    assert store.get_settings().cluster_cohesion_similarity == 0.65
    assert store.queue_cluster("legacy-event", 0.72, 2, 0.74).cohesion_similarity == 0.74
