"""SQLite persistence for incremental indexing state."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from .models import EventSummary, ImageIndexOutcome, ImageIndexStatus, IndexStatus


class StatusStore:
    """Small connection-per-operation store safe for the indexing thread."""

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
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rebuild_required INTEGER NOT NULL DEFAULT 0,
                    pipeline_version TEXT NOT NULL DEFAULT '',
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS images (
                    event_id TEXT NOT NULL,
                    photo_path TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    face_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (event_id, photo_path),
                    FOREIGN KEY (event_id) REFERENCES events(event_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_events_queue
                    ON events(status, event_date, created_at);
                CREATE INDEX IF NOT EXISTS idx_images_status
                    ON images(event_id, status);
                """
            )

    def register_event(self, event_id: str, event_date: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO events(event_id, event_date, status)
                VALUES (?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    event_date = excluded.event_date,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (event_id, event_date, IndexStatus.PENDING.value),
            )

    def get_event(self, event_id: str) -> EventSummary | None:
        with self._connect() as connection:
            row = connection.execute(
                self._event_summary_query("WHERE e.event_id = ?"), (event_id,)
            ).fetchone()
        return self._event_from_row(row) if row else None

    def list_events(self) -> list[EventSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                self._event_summary_query("") + " ORDER BY e.event_date, e.event_id"
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    @staticmethod
    def _event_summary_query(suffix: str) -> str:
        return f"""
            SELECT e.event_id, e.event_date, e.status, e.rebuild_required,
                   e.error, e.updated_at,
                   COUNT(i.photo_path) AS total_images,
                   SUM(CASE WHEN i.status = 'indexed' THEN 1 ELSE 0 END) AS indexed_images,
                   SUM(CASE WHEN i.status = 'no_face' THEN 1 ELSE 0 END) AS no_face_images,
                   SUM(CASE WHEN i.status = 'failed' THEN 1 ELSE 0 END) AS failed_images,
                   SUM(CASE WHEN i.status IN ('pending', 'queued')
                            THEN 1 ELSE 0 END) AS pending_images
            FROM events e
            LEFT JOIN images i ON i.event_id = e.event_id
            {suffix}
            GROUP BY e.event_id
        """

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> EventSummary:
        return EventSummary(
            event_id=row["event_id"],
            event_date=row["event_date"],
            status=IndexStatus(row["status"]),
            rebuild_required=bool(row["rebuild_required"]),
            total_images=row["total_images"] or 0,
            indexed_images=row["indexed_images"] or 0,
            no_face_images=row["no_face_images"] or 0,
            failed_images=row["failed_images"] or 0,
            pending_images=row["pending_images"] or 0,
            error=row["error"],
            updated_at=row["updated_at"],
        )

    def list_images(self, event_id: str) -> list[ImageIndexStatus]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM images WHERE event_id = ? ORDER BY photo_path",
                (event_id,),
            ).fetchall()
        return [self._image_from_row(row) for row in rows]

    def get_image_map(self, event_id: str) -> dict[str, ImageIndexStatus]:
        return {image.photo_path: image for image in self.list_images(event_id)}

    @staticmethod
    def _image_from_row(row: sqlite3.Row) -> ImageIndexStatus:
        return ImageIndexStatus(
            event_id=row["event_id"],
            photo_path=row["photo_path"],
            fingerprint=row["fingerprint"],
            status=IndexStatus(row["status"]),
            face_count=row["face_count"],
            error=row["error"],
            updated_at=row["updated_at"],
        )

    def upsert_pending_image(
        self, event_id: str, photo_path: str, fingerprint: str
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO images(event_id, photo_path, fingerprint, status)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(event_id, photo_path) DO UPDATE SET
                    fingerprint = excluded.fingerprint,
                    status = excluded.status,
                    face_count = 0,
                    error = NULL,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (event_id, photo_path, fingerprint, IndexStatus.PENDING.value),
            )

    def remove_images(self, event_id: str, photo_paths: Iterable[str]) -> int:
        paths = tuple(photo_paths)
        if not paths:
            return 0
        with self._connect() as connection:
            placeholders = ",".join("?" for _ in paths)
            cursor = connection.execute(
                f"DELETE FROM images WHERE event_id = ? AND photo_path IN ({placeholders})",
                (event_id, *paths),
            )
            return cursor.rowcount

    def require_rebuild(self, event_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE events
                SET rebuild_required = 1,
                    status = CASE
                        WHEN status IN ('queued', 'indexing') THEN status
                        ELSE ?
                    END,
                    error = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE event_id = ?
                """,
                (IndexStatus.PENDING.value, event_id),
            )

    def pipeline_version(self, event_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT pipeline_version FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return row["pipeline_version"] if row else None

    def queue_event(self, event_id: str) -> bool:
        """Persist a queue request unless the same event is already active."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown event: {event_id}")
            if row["status"] in {IndexStatus.QUEUED.value, IndexStatus.INDEXING.value}:
                return False
            connection.execute(
                """
                UPDATE events SET status = ?, error = NULL,
                    updated_at = CURRENT_TIMESTAMP WHERE event_id = ?
                """,
                (IndexStatus.QUEUED.value, event_id),
            )
            connection.execute(
                """
                UPDATE images SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE event_id = ? AND status = ?
                """,
                (IndexStatus.QUEUED.value, event_id, IndexStatus.PENDING.value),
            )
            return True

    def claim_next_event(self) -> tuple[str, bool] | None:
        """Atomically claim the oldest queued event."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT event_id, rebuild_required FROM events WHERE status = ?
                ORDER BY event_date ASC, created_at ASC, event_id ASC LIMIT 1
                """,
                (IndexStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            event_id = row["event_id"]
            rebuild_required = bool(row["rebuild_required"])
            connection.execute(
                """
                UPDATE events SET status = ?, rebuild_required = 0,
                    updated_at = CURRENT_TIMESTAMP
                WHERE event_id = ?
                """,
                (IndexStatus.INDEXING.value, event_id),
            )
            connection.execute(
                """
                UPDATE images SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE event_id = ? AND status = ?
                """,
                (IndexStatus.INDEXING.value, event_id, IndexStatus.QUEUED.value),
            )
            return event_id, rebuild_required

    def paths_for_processing(self, event_id: str, rebuild: bool) -> list[str]:
        with self._connect() as connection:
            if rebuild:
                rows = connection.execute(
                    """
                    SELECT photo_path FROM images
                    WHERE event_id = ? ORDER BY photo_path
                    """,
                    (event_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT photo_path FROM images
                    WHERE event_id = ? AND status = ? ORDER BY photo_path
                    """,
                    (event_id, IndexStatus.INDEXING.value),
                ).fetchall()
        return [row["photo_path"] for row in rows]

    def record_outcomes(
        self, event_id: str, outcomes: Iterable[ImageIndexOutcome]
    ) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                UPDATE images SET status = ?, face_count = ?, error = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE event_id = ? AND photo_path = ?
                """,
                [
                    (
                        outcome.status.value,
                        outcome.face_count,
                        outcome.error,
                        event_id,
                        outcome.photo_path,
                    )
                    for outcome in outcomes
                ],
            )

    def complete_event(self, event_id: str, pipeline_version: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE events SET status = ?, pipeline_version = ?, error = NULL,
                    updated_at = CURRENT_TIMESTAMP WHERE event_id = ?
                """,
                (IndexStatus.INDEXED.value, pipeline_version, event_id),
            )

    def fail_event(self, event_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE events SET status = ?, rebuild_required = 1, error = ?,
                    updated_at = CURRENT_TIMESTAMP WHERE event_id = ?
                """,
                (IndexStatus.FAILED.value, error, event_id),
            )
            connection.execute(
                """
                UPDATE images SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE event_id = ? AND status = ?
                """,
                (IndexStatus.PENDING.value, event_id, IndexStatus.INDEXING.value),
            )

    def retry_failed_images(self, event_id: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE images SET status = ?, error = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE event_id = ? AND status = ?
                """,
                (IndexStatus.PENDING.value, event_id, IndexStatus.FAILED.value),
            )
            return cursor.rowcount

    def recover_interrupted_events(self) -> list[str]:
        """Return persisted unfinished events to the queue after a restart."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE events SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE status IN (?, ?, ?)
                """,
                (
                    IndexStatus.QUEUED.value,
                    IndexStatus.PENDING.value,
                    IndexStatus.QUEUED.value,
                    IndexStatus.INDEXING.value,
                ),
            )
            connection.execute(
                """
                UPDATE images SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE status IN (?, ?)
                  AND event_id IN (
                      SELECT event_id FROM events WHERE status = ?
                  )
                """,
                (
                    IndexStatus.QUEUED.value,
                    IndexStatus.PENDING.value,
                    IndexStatus.INDEXING.value,
                    IndexStatus.QUEUED.value,
                ),
            )
            rows = connection.execute(
                """
                SELECT event_id FROM events WHERE status = ?
                ORDER BY event_date, created_at, event_id
                """,
                (IndexStatus.QUEUED.value,),
            ).fetchall()
        return [row["event_id"] for row in rows]
