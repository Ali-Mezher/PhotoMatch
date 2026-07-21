"""Route and persistence tests for the local operator console."""

from __future__ import annotations

from io import BytesIO
import sqlite3

from PIL import Image
from src.services.admin_store import AdminStore, RuntimeSettings
from src.services.status_store import StatusStore
from src.web import create_app


def _app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "EVENTS_DIR": tmp_path / "events",
            "INDEX_STATUS_DB": tmp_path / "status.sqlite3",
            "ADMIN_USERNAME": "operator",
            "ADMIN_PASSWORD": "correct horse",
            "ADMIN_START_WORKER": False,
        }
    )


def _csrf(client):
    with client.session_transaction() as session:
        return session["csrf_token"]


def _login(client):
    client.get("/admin/login")
    return client.post(
        "/admin/login",
        data={
            "_csrf_token": _csrf(client),
            "username": "operator",
            "password": "correct horse",
        },
    )


def _png(name="photo.png"):
    stream = BytesIO()
    Image.new("RGB", (24, 20), "#28708a").save(stream, "PNG")
    stream.seek(0)
    return stream, name


def test_admin_requires_login_and_accepts_configured_credentials(tmp_path):
    client = _app(tmp_path).test_client()

    protected = client.get("/admin/")
    assert protected.status_code == 303
    assert "/admin/login" in protected.location

    response = _login(client)
    assert response.status_code == 303
    assert response.location.endswith("/admin/")
    assert client.get("/admin/").status_code == 200


def test_admin_rejects_bad_credentials_and_missing_csrf(tmp_path):
    client = _app(tmp_path).test_client()
    client.get("/admin/login")

    assert client.post(
        "/admin/login", data={"username": "operator", "password": "wrong"}
    ).status_code == 400
    response = client.post(
        "/admin/login",
        data={
            "_csrf_token": _csrf(client),
            "username": "operator",
            "password": "wrong",
        },
    )
    assert response.status_code == 401


def test_event_creation_generates_slug_code_and_display_name(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)

    response = client.post(
        "/admin/events",
        data={
            "_csrf_token": _csrf(client),
            "title": "Summer Gala 2026",
            "event_date": "2026-07-21",
        },
    )
    assert response.status_code == 303
    assert response.location.endswith("/admin/events/summer-gala-2026")

    event = app.extensions["indexing_service"].get_event("summer-gala-2026")
    assert event.display_name == "Summer Gala 2026"
    assert len(app.extensions["indexing_service"].get_event_access_code(event.event_id)) == 8
    assert (tmp_path / "events" / event.event_id / "raw").is_dir()


def test_photo_upload_is_atomic_tracks_image_and_queues_index(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)
    client.post(
        "/admin/events",
        data={
            "_csrf_token": _csrf(client),
            "title": "Test Event",
            "event_date": "2026-07-21",
        },
    )

    response = client.post(
        "/admin/events/test-event/photos",
        data={
            "_csrf_token": _csrf(client),
            "final_batch": "1",
            "photos": [_png(), (BytesIO(b"not an image"), "bad.jpg")],
        },
        headers={"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.json["imported"] == 1
    assert response.json["rejected"] == 1
    assert (tmp_path / "events" / "test-event" / "raw" / "photo.png").is_file()
    assert not list((tmp_path / "events" / "test-event" / "raw").glob("*.uploading"))

    event = app.extensions["indexing_service"].get_event("test-event")
    assert event.pending_images == 1
    assert event.status.value == "queued"


def test_duplicate_photo_does_not_overwrite_original(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)
    client.post(
        "/admin/events",
        data={"_csrf_token": _csrf(client), "title": "Duplicates", "event_date": "2026-01-01"},
    )
    url = "/admin/events/duplicates/photos"
    first = client.post(
        url,
        data={"_csrf_token": _csrf(client), "photos": [_png("same.png")]},
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    original = (tmp_path / "events" / "duplicates" / "raw" / "same.png").read_bytes()
    second = client.post(
        url,
        data={"_csrf_token": _csrf(client), "photos": [_png("same.png")]},
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert first.status_code == 200
    assert second.status_code == 400
    assert (tmp_path / "events" / "duplicates" / "raw" / "same.png").read_bytes() == original


def test_runtime_settings_validate_and_persist(tmp_path):
    store = AdminStore(tmp_path / "admin.sqlite3")
    settings = RuntimeSettings(0.55, 0.72, 350, 0.78, 3)
    store.update_settings(settings)
    assert AdminStore(tmp_path / "admin.sqlite3").get_settings() == settings

    try:
        store.update_settings(RuntimeSettings(0.8, 0.7, 200, 0.65, 2))
    except ValueError as exc:
        assert "possible" in str(exc)
    else:
        raise AssertionError("invalid thresholds should fail")


def test_security_headers_are_applied(tmp_path):
    response = _app(tmp_path).test_client().get("/admin/login")
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_existing_event_schema_migrates_display_names_without_data_loss(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                event_date TEXT NOT NULL,
                status TEXT NOT NULL,
                rebuild_required INTEGER NOT NULL DEFAULT 0,
                pipeline_version TEXT NOT NULL DEFAULT '',
                error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            "INSERT INTO events(event_id, event_date, status) VALUES ('legacy-event', '2026-01-02', 'indexed')"
        )

    event = StatusStore(database).get_event("legacy-event")
    assert event is not None
    assert event.display_name == "legacy-event"
    assert event.status.value == "indexed"
