"""SQLite persistence for operator settings, audit records, and clustering."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class RuntimeSettings:
    possible_threshold: float = 0.50
    confident_threshold: float = 0.65
    top_k: int = 200
    cluster_similarity: float = 0.65
    min_cluster_size: int = 2

    def validate(self) -> "RuntimeSettings":
        if not 0 <= self.possible_threshold < self.confident_threshold <= 1:
            raise ValueError(
                "Match thresholds must satisfy 0 ≤ possible < confident ≤ 1."
            )
        if not 1 <= self.top_k <= 5000:
            raise ValueError("Search candidates must be between 1 and 5000.")
        if not 0.50 <= self.cluster_similarity <= 0.95:
            raise ValueError("Cluster similarity must be between 0.50 and 0.95.")
        if not 2 <= self.min_cluster_size <= 20:
            raise ValueError("Minimum cluster size must be between 2 and 20.")
        return self


@dataclass(frozen=True)
class ClusterRun:
    run_id: str
    event_id: str
    status: str
    index_generation: str | None
    similarity: float
    min_cluster_size: int
    cluster_count: int
    noise_count: int
    error: str | None
    requested_at: str
    completed_at: str | None


@dataclass(frozen=True)
class ClusterMember:
    run_id: str
    face_index: int
    cluster_label: int
    photo_path: str
    bbox: tuple[int, int, int, int]
    confidence: float


class AdminStore:
    """Connection-per-operation store safe for web and worker threads."""

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runtime_settings (
                    singleton_id INTEGER PRIMARY KEY CHECK(singleton_id = 1),
                    possible_threshold REAL NOT NULL,
                    confident_threshold REAL NOT NULL,
                    top_k INTEGER NOT NULL,
                    cluster_similarity REAL NOT NULL,
                    min_cluster_size INTEGER NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    event_id TEXT,
                    details TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS cluster_runs (
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

                CREATE INDEX IF NOT EXISTS idx_cluster_queue
                    ON cluster_runs(status, requested_at);
                CREATE INDEX IF NOT EXISTS idx_cluster_event
                    ON cluster_runs(event_id, requested_at DESC);

                CREATE TABLE IF NOT EXISTS cluster_members (
                    run_id TEXT NOT NULL,
                    face_index INTEGER NOT NULL,
                    cluster_label INTEGER NOT NULL,
                    photo_path TEXT NOT NULL,
                    bbox_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    PRIMARY KEY(run_id, face_index),
                    FOREIGN KEY(run_id) REFERENCES cluster_runs(run_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS cluster_labels (
                    run_id TEXT NOT NULL,
                    cluster_label INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(run_id, cluster_label),
                    FOREIGN KEY(run_id) REFERENCES cluster_runs(run_id)
                        ON DELETE CASCADE
                );
                """
            )
            defaults = RuntimeSettings()
            connection.execute(
                """
                INSERT OR IGNORE INTO runtime_settings(
                    singleton_id, possible_threshold, confident_threshold,
                    top_k, cluster_similarity, min_cluster_size
                ) VALUES (1, ?, ?, ?, ?, ?)
                """,
                (
                    defaults.possible_threshold,
                    defaults.confident_threshold,
                    defaults.top_k,
                    defaults.cluster_similarity,
                    defaults.min_cluster_size,
                ),
            )

    def get_settings(self) -> RuntimeSettings:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runtime_settings WHERE singleton_id = 1"
            ).fetchone()
        return RuntimeSettings(
            possible_threshold=row["possible_threshold"],
            confident_threshold=row["confident_threshold"],
            top_k=row["top_k"],
            cluster_similarity=row["cluster_similarity"],
            min_cluster_size=row["min_cluster_size"],
        )

    def update_settings(self, settings: RuntimeSettings) -> RuntimeSettings:
        settings.validate()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runtime_settings SET
                    possible_threshold = ?, confident_threshold = ?, top_k = ?,
                    cluster_similarity = ?, min_cluster_size = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE singleton_id = 1
                """,
                (
                    settings.possible_threshold,
                    settings.confident_threshold,
                    settings.top_k,
                    settings.cluster_similarity,
                    settings.min_cluster_size,
                ),
            )
        return settings

    def reset_settings(self) -> RuntimeSettings:
        return self.update_settings(RuntimeSettings())

    def record_audit(
        self, action: str, event_id: str | None = None, **details: object
    ) -> None:
        safe_details = {
            key: value
            for key, value in details.items()
            if key not in {"password", "secret", "photo_path"}
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO audit_log(action, event_id, details) VALUES (?, ?, ?)",
                (action, event_id, json.dumps(safe_details, default=str)),
            )

    def recent_audit(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.audit_page(limit=limit, offset=0)

    def audit_page(self, limit: int, offset: int = 0) -> list[sqlite3.Row]:
        if limit <= 0 or offset < 0:
            raise ValueError("audit pagination values are invalid")
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM audit_log ORDER BY audit_id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

    def audit_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM audit_log").fetchone()
        return int(row["total"])

    def queue_cluster(
        self, event_id: str, similarity: float, min_cluster_size: int
    ) -> ClusterRun:
        run_id = uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cluster_runs(
                    run_id, event_id, status, similarity, min_cluster_size
                ) VALUES (?, ?, 'queued', ?, ?)
                """,
                (run_id, event_id, similarity, min_cluster_size),
            )
        return self.get_cluster_run(run_id)

    def recover_interrupted_clusters(self) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE cluster_runs SET status = 'queued', started_at = NULL,
                    error = NULL WHERE status = 'running'
                """
            )
            queued = connection.execute(
                "SELECT 1 FROM cluster_runs WHERE status = 'queued' LIMIT 1"
            ).fetchone()
        return cursor.rowcount > 0 or queued is not None

    def claim_next_cluster(self) -> ClusterRun | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT run_id FROM cluster_runs WHERE status = 'queued'
                ORDER BY requested_at, run_id LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE cluster_runs SET status = 'running',
                    started_at = CURRENT_TIMESTAMP, error = NULL
                WHERE run_id = ?
                """,
                (row["run_id"],),
            )
        return self.get_cluster_run(row["run_id"])

    def complete_cluster(
        self,
        run_id: str,
        generation: str,
        assignments: list[tuple[int, int, str, tuple[int, int, int, int], float]],
    ) -> None:
        labels = {assignment[1] for assignment in assignments if assignment[1] >= 0}
        noise_count = sum(assignment[1] < 0 for assignment in assignments)
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO cluster_members(
                    run_id, face_index, cluster_label, photo_path,
                    bbox_json, confidence
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (run_id, face_index, label, path, json.dumps(bbox), confidence)
                    for face_index, label, path, bbox, confidence in assignments
                ],
            )
            connection.execute(
                """
                UPDATE cluster_runs SET status = 'complete', index_generation = ?,
                    cluster_count = ?, noise_count = ?, completed_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
                """,
                (generation, len(labels), noise_count, run_id),
            )

    def fail_cluster(self, run_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE cluster_runs SET status = 'failed', error = ?,
                    completed_at = CURRENT_TIMESTAMP WHERE run_id = ?
                """,
                (error[:1000], run_id),
            )

    def get_cluster_run(self, run_id: str) -> ClusterRun:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM cluster_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._run_from_row(row)

    def latest_cluster(self, event_id: str) -> ClusterRun | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM cluster_runs WHERE event_id = ?
                ORDER BY requested_at DESC, run_id DESC LIMIT 1
                """,
                (event_id,),
            ).fetchone()
        return self._run_from_row(row) if row else None

    def cluster_members(self, run_id: str) -> list[ClusterMember]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM cluster_members WHERE run_id = ?
                ORDER BY cluster_label, face_index
                """,
                (run_id,),
            ).fetchall()
        return [
            self._member_from_row(row)
            for row in rows
        ]

    def get_cluster_member(
        self, run_id: str, face_index: int
    ) -> ClusterMember | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM cluster_members
                WHERE run_id = ? AND face_index = ?
                """,
                (run_id, face_index),
            ).fetchone()
        return self._member_from_row(row) if row else None

    def cluster_labels(self, run_id: str) -> dict[int, str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT cluster_label, label FROM cluster_labels WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        return {row["cluster_label"]: row["label"] for row in rows}

    def set_cluster_label(self, run_id: str, cluster_label: int, label: str) -> None:
        label = label.strip()
        with self._connect() as connection:
            if label:
                connection.execute(
                    """
                    INSERT INTO cluster_labels(run_id, cluster_label, label)
                    VALUES (?, ?, ?)
                    ON CONFLICT(run_id, cluster_label) DO UPDATE SET
                        label = excluded.label, updated_at = CURRENT_TIMESTAMP
                    """,
                    (run_id, cluster_label, label[:80]),
                )
            else:
                connection.execute(
                    "DELETE FROM cluster_labels WHERE run_id = ? AND cluster_label = ?",
                    (run_id, cluster_label),
                )

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> ClusterRun:
        return ClusterRun(
            run_id=row["run_id"],
            event_id=row["event_id"],
            status=row["status"],
            index_generation=row["index_generation"],
            similarity=row["similarity"],
            min_cluster_size=row["min_cluster_size"],
            cluster_count=row["cluster_count"],
            noise_count=row["noise_count"],
            error=row["error"],
            requested_at=row["requested_at"],
            completed_at=row["completed_at"],
        )

    @staticmethod
    def _member_from_row(row: sqlite3.Row) -> ClusterMember:
        return ClusterMember(
            run_id=row["run_id"],
            face_index=row["face_index"],
            cluster_label=row["cluster_label"],
            photo_path=row["photo_path"],
            bbox=tuple(json.loads(row["bbox_json"])),
            confidence=row["confidence"],
        )
