"""Event organizers: responsible people reachable for the attendee code."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.indexing import IndexBuildOutcome
from src.services import IndexingService


class _Updater:
    def __call__(self, event_id, paths, rebuild, show_progress):
        return object(), [
            IndexBuildOutcome(Path(path), "indexed", face_count=1) for path in paths
        ]

    def exists(self, event_id):
        return False


def _service(tmp_path):
    events_dir = tmp_path / "events"
    updater = _Updater()
    service = IndexingService(
        events_dir=events_dir,
        database_path=tmp_path / "status.sqlite3",
        index_updater=updater,
        index_exists=updater.exists,
    )
    (events_dir / "gala" / "raw").mkdir(parents=True)
    service.register_event("gala", "2026-07-21", "Summer Gala")
    return service


def test_add_list_and_remove_organizers(tmp_path):
    service = _service(tmp_path)

    deemah = service.add_organizer("gala", "Deemah", "deemah@example.com")
    service.add_organizer("gala", "Ali", "ali@example.com")

    organizers = service.list_organizers("gala")
    assert [(o.name, o.email) for o in organizers] == [
        ("Deemah", "deemah@example.com"),
        ("Ali", "ali@example.com"),
    ]

    assert service.remove_organizer("gala", deemah.id) is True
    assert [o.name for o in service.list_organizers("gala")] == ["Ali"]
    assert service.remove_organizer("gala", deemah.id) is False


def test_organizer_input_is_validated(tmp_path):
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="name is required"):
        service.add_organizer("gala", "   ", "who@example.com")
    with pytest.raises(ValueError, match="valid email"):
        service.add_organizer("gala", "Hessah", "not-an-email")
    with pytest.raises(ValueError, match="valid email"):
        service.add_organizer("gala", "Hessah", "spaces @example.com")

    # Values are trimmed before storage.
    stored = service.add_organizer("gala", "  Hessah  ", "  hessah@example.com ")
    assert stored.name == "Hessah"
    assert stored.email == "hessah@example.com"


def test_adding_organizer_to_unknown_event_raises(tmp_path):
    service = _service(tmp_path)
    with pytest.raises(KeyError):
        service.add_organizer("ghost", " Nobody", "nobody@example.com")


def test_deleting_event_cascades_to_organizers(tmp_path):
    service = _service(tmp_path)
    service.add_organizer("gala", "Deemah", "deemah@example.com")

    service.store.delete_event_if_idle("gala")

    # The catalog row is gone and its organizers went with it (FK cascade).
    assert service.store.list_organizers("gala") == []
