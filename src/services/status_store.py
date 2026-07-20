"""Local SQLite persistence for indexing scheduler state."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import EventSummary, ImageIndexStatus, IndexStatus


class StatusStore:
    """Thread-safe-by-connection SQLite store for operational metadata only."""

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL, total_images INTEGER NOT NULL DEFAULT 0,
                    error TEXT, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS images (
                    event_id TEXT NOT NULL, photo_path TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL, face_count INTEGER NOT NULL DEFAULT 0, error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (event_id, photo_path)
                );
                """
            )

    def get_event(self, event_id: str) -> EventSummary | None:
        with self._connect() as connection:
            row = connection.execute(self._event_query("WHERE e.event_id = ?"), (event_id,)).fetchone()
        return self._event_from_row(row) if row else None

    def list_events(self) -> list[EventSummary]:
        with self._connect() as connection:
            rows = connection.execute(self._event_query("") + " ORDER BY e.event_id").fetchall()
        return [self._event_from_row(row) for row in rows]

    @staticmethod
    def _event_query(suffix: str) -> str:
        return f"""
            SELECT e.event_id, e.status, e.total_images, e.error, e.updated_at,
                   SUM(CASE WHEN i.status = 'indexed' THEN 1 ELSE 0 END) AS indexed_images,
                   SUM(CASE WHEN i.status = 'no_face' THEN 1 ELSE 0 END) AS no_face_images,
                   SUM(CASE WHEN i.status = 'failed' THEN 1 ELSE 0 END) AS failed_images
            FROM events e LEFT JOIN images i ON i.event_id = e.event_id
            {suffix} GROUP BY e.event_id
        """

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> EventSummary:
        return EventSummary(
            event_id=row["event_id"], status=IndexStatus(row["status"]),
            total_images=row["total_images"], indexed_images=row["indexed_images"] or 0,
            no_face_images=row["no_face_images"] or 0, failed_images=row["failed_images"] or 0,
            updated_at=row["updated_at"], error=row["error"],
        )

    def get_fingerprint(self, event_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute("SELECT fingerprint FROM events WHERE event_id = ?", (event_id,)).fetchone()
        return row["fingerprint"] if row else None

    def upsert_event(self, event_id: str, fingerprint: str, status: IndexStatus, total_images: int, error: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO events(event_id, fingerprint, status, total_images, error, updated_at)
                   VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(event_id) DO UPDATE SET fingerprint=excluded.fingerprint,
                   status=excluded.status, total_images=excluded.total_images,
                   error=excluded.error, updated_at=CURRENT_TIMESTAMP""",
                (event_id, fingerprint, status.value, total_images, error),
            )

    def upsert_image(self, event_id: str, photo_path: str, fingerprint: str, status: IndexStatus, face_count: int = 0, error: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO images(event_id, photo_path, fingerprint, status, face_count, error, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(event_id, photo_path) DO UPDATE SET fingerprint=excluded.fingerprint,
                   status=excluded.status, face_count=excluded.face_count,
                   error=excluded.error, updated_at=CURRENT_TIMESTAMP""",
                (event_id, photo_path, fingerprint, status.value, face_count, error),
            )

    def remove_missing_images(self, event_id: str, paths: set[str]) -> None:
        with self._connect() as connection:
            if paths:
                placeholders = ",".join("?" for _ in paths)
                connection.execute(f"DELETE FROM images WHERE event_id = ? AND photo_path NOT IN ({placeholders})", (event_id, *sorted(paths)))
            else:
                connection.execute("DELETE FROM images WHERE event_id = ?", (event_id,))

    def list_images(self, event_id: str) -> list[ImageIndexStatus]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM images WHERE event_id = ? ORDER BY photo_path", (event_id,)).fetchall()
        return [ImageIndexStatus(row["photo_path"], IndexStatus(row["status"]), row["face_count"], row["error"], row["updated_at"]) for row in rows]
