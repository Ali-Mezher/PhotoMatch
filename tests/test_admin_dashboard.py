"""Route and persistence tests for the local operator console."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from zipfile import ZipFile

from PIL import Image
from src.services.admin_store import AdminStore, RuntimeSettings
from src.services.models import ImageIndexOutcome, IndexStatus
from src.services.status_store import StatusStore
from src.web import create_app


def _app(tmp_path, **overrides):
    config = {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "EVENTS_DIR": tmp_path / "events",
            "INDEX_STATUS_DB": tmp_path / "status.sqlite3",
            "ADMIN_USERNAME": "operator",
            "ADMIN_PASSWORD": "correct horse",
            "ADMIN_START_WORKER": False,
        }
    config.update(overrides)
    return create_app(config)


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


def _indexed_event(app, event_id="ready-event", display_name="Ready Event"):
    raw_dir = app.extensions["indexing_service"].events_dir / event_id / "raw"
    raw_dir.mkdir(parents=True)
    match = raw_dir / "match.png"
    match.write_bytes(_png()[0].getvalue())
    store = app.extensions["indexing_service"].store
    store.register_event(event_id, "2026-07-21", display_name)
    store.complete_event(event_id, "test-pipeline")
    return match


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


def test_overview_shows_five_recent_rows_and_activity_log_is_paginated(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)
    store = app.extensions["admin_store"]
    for index in range(60):
        store.record_audit(
            f"fixture_action_{index:02d}",
            "event-fixture",
            sequence=index,
        )

    overview = client.get("/admin/")
    first_page = client.get("/admin/activity?per_page=25&page=1")
    second_page = client.get("/admin/activity?per_page=25&page=2")

    assert overview.status_code == 200
    assert overview.data.count(b'class="activity-action"') == 5
    assert b"Fixture Action 59" in overview.data
    assert b"Fixture Action 54" not in overview.data
    assert b"View All" in overview.data

    assert first_page.status_code == 200
    assert first_page.data.count(b'data-label="Action"') == 25
    assert b"Rows per page" in first_page.data
    assert b"Fixture Action 59" in first_page.data
    assert b"sequence: 59" in first_page.data
    assert b"Page 1 of 3" in first_page.data

    assert second_page.status_code == 200
    assert second_page.data.count(b'data-label="Action"') == 25
    assert b"Page 2 of 3" in second_page.data


def test_activity_log_normalizes_page_size_and_redirects_overflow_page(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)

    invalid_size = client.get("/admin/activity?per_page=20")
    overflow = client.get("/admin/activity?per_page=25&page=99")

    assert invalid_size.status_code == 200
    assert b'<option value="25" selected>' in invalid_size.data
    assert overflow.status_code == 303
    assert "page=1" in overflow.location


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


def test_event_deletion_requires_exact_confirmation_and_removes_local_state(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)
    client.post(
        "/admin/events",
        data={
            "_csrf_token": _csrf(client),
            "title": "Delete Me",
            "event_date": "2026-07-21",
        },
    )
    event_root = tmp_path / "events" / "delete-me"
    (event_root / "raw" / "photo.jpg").write_bytes(b"fixture")
    (event_root / "indexed").mkdir()
    (event_root / "indexed" / "faces.faiss").write_bytes(b"index")
    url = "/admin/events/delete-me/delete"

    rejected = client.post(
        url,
        data={"_csrf_token": _csrf(client), "confirmation": "wrong-event"},
    )

    assert rejected.status_code == 400
    assert event_root.exists()
    assert app.extensions["indexing_service"].get_event("delete-me")

    deleted = client.post(
        url,
        data={"_csrf_token": _csrf(client), "confirmation": "delete-me"},
    )

    assert deleted.status_code == 303
    assert deleted.location.endswith("/admin/")
    assert not event_root.exists()
    assert app.extensions["indexing_service"].store.get_event("delete-me") is None
    assert (
        app.extensions["indexing_service"].store.get_event_access_code("delete-me")
        is None
    )
    assert client.get("/admin/events/delete-me").status_code == 404
    assert app.extensions["admin_store"].recent_audit(1)[0]["action"] == "event_deleted"


def test_event_deletion_refuses_active_indexing(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)
    client.post(
        "/admin/events",
        data={
            "_csrf_token": _csrf(client),
            "title": "Busy Event",
            "event_date": "2026-07-21",
        },
    )
    indexing = app.extensions["indexing_service"]
    indexing.store.queue_event("busy-event")
    assert indexing.store.claim_next_event()[0] == "busy-event"

    response = client.post(
        "/admin/events/busy-event/delete",
        data={"_csrf_token": _csrf(client), "confirmation": "busy-event"},
    )

    assert response.status_code == 409
    assert b"active indexing" in response.data
    assert (tmp_path / "events" / "busy-event").exists()
    assert indexing.get_event("busy-event")


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


def test_issue_23_service_schema_migrates_and_accepts_new_events(tmp_path):
    database = tmp_path / "issue23.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                total_images INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE images (
                event_id TEXT NOT NULL,
                photo_path TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                face_count INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(event_id, photo_path)
            );
            INSERT INTO events(event_id, fingerprint, status, total_images)
            VALUES ('legacy-event', 'fingerprint', 'indexed', 1);
            INSERT INTO images(
                event_id, photo_path, fingerprint, status, face_count
            ) VALUES ('legacy-event', 'legacy.jpg', 'photo-fingerprint', 'indexed', 2);
            """
        )

    store = StatusStore(database)
    legacy = store.get_event("legacy-event")
    store.register_event("new-event", "2026-07-22", "New Event")

    assert legacy is not None
    assert legacy.display_name == "legacy-event"
    assert legacy.indexed_images == 1
    assert store.list_images("legacy-event")[0].face_count == 2
    assert store.get_event("new-event").display_name == "New Event"


def test_event_inventory_searches_and_paginates_in_sql(tmp_path):
    app = _app(tmp_path)
    store = app.extensions["indexing_service"].store
    for index in range(30):
        store.register_event(
            f"event-{index:02d}",
            f"2026-07-{(index % 28) + 1:02d}",
            "Summer Gala" if index == 7 else f"Event {index:02d}",
        )

    code = store.get_event_access_code("event-07")
    formatted_code = f"{code[:4]}-{code[4:]}"
    by_name, name_count = store.search_events("summer gala", 25, 0)
    by_code, code_count = store.search_events(formatted_code, 25, 0)
    by_date, date_count = store.search_events("2026-07-08", 25, 0)
    first_page, total = store.search_events(limit=10, offset=0)
    second_page, _ = store.search_events(limit=10, offset=10)

    assert [event.event_id for event in by_name] == ["event-07"]
    assert name_count == code_count == 1
    assert [event.event_id for event in by_code] == ["event-07"]
    assert any(event.event_id == "event-07" for event in by_date)
    assert date_count >= 1
    assert total == 30
    assert len(first_page) == len(second_page) == 10
    assert {event.event_id for event in first_page}.isdisjoint(
        event.event_id for event in second_page
    )


def test_admin_overview_preserves_filter_page_size_and_global_totals(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)
    store = app.extensions["indexing_service"].store
    for index in range(12):
        store.register_event(
            f"catalog-{index:02d}", "2026-06-01", f"Catalog Event {index:02d}"
        )

    response = client.get("/admin/?q=Catalog&per_page=10&page=2")

    assert response.status_code == 200
    assert b"Page 2 of 2" in response.data
    assert b"12 matching events" in response.data
    assert b"Catalog Event 11" in response.data
    assert b'value="Catalog"' in response.data
    assert b">12</strong><span>Events" in response.data
    redirected = client.get("/admin/?q=Catalog&per_page=10&page=9")
    assert redirected.status_code == 303
    assert "page=2" in redirected.location


def test_inline_admin_search_uses_event_without_code_and_scopes_results(tmp_path):
    calls = []
    match_path = None

    def matcher(image, event_id):
        calls.append((image, event_id))
        return {
            "confident": [SimpleNamespace(photo_path=str(match_path), score=0.93)],
            "possible": [],
        }

    app = _app(tmp_path, MATCHER=matcher)
    match_path = _indexed_event(app)
    client = app.test_client()
    _login(client)

    detail = client.get("/admin/events/ready-event")
    response = client.post(
        "/admin/events/ready-event/search",
        data={"_csrf_token": _csrf(client), "selfie": _png("selfie.png")},
        content_type="multipart/form-data",
    )

    assert b"Search for a person" in detail.data
    assert b"event code" not in detail.data.lower()
    assert b"consent" not in detail.data.lower()
    assert b"No attendee code required" in detail.data
    assert b"Hang tight" in detail.data
    assert response.status_code == 303
    assert response.location.startswith("/admin/events/ready-event?search_token=")
    assert response.location.endswith("#person-search")
    assert len(calls) == 1 and calls[0][1] == "ready-event"

    results = client.get(response.location)
    assert b"Confident matches" in results.data
    assert str(match_path).encode() not in results.data
    token = response.location.split("search_token=", 1)[1].split("#", 1)[0]
    assert client.get(f"/results/{token}").status_code == 404


def test_admin_search_download_and_zip_are_limited_to_result_set(tmp_path):
    match_path = None

    def matcher(_image, _event_id):
        return {
            "confident": [SimpleNamespace(photo_path=str(match_path), score=0.9)],
            "possible": [],
        }

    app = _app(tmp_path, MATCHER=matcher)
    match_path = _indexed_event(app)
    client = app.test_client()
    _login(client)
    searched = client.post(
        "/admin/events/ready-event/search",
        data={"_csrf_token": _csrf(client), "selfie": _png("selfie.png")},
        content_type="multipart/form-data",
    )
    token = searched.location.split("search_token=", 1)[1].split("#", 1)[0]
    stored = app.config["RESULT_STORE"].get(token)
    photo_id = next(iter(stored.photos))

    download = client.get(
        f"/admin/events/ready-event/search-results/{token}/download/{photo_id}"
    )
    archive = client.post(
        f"/admin/events/ready-event/search-results/{token}/export",
        data={"_csrf_token": _csrf(client), "photo_ids": [photo_id]},
    )

    assert download.status_code == 200
    assert download.data == match_path.read_bytes()
    assert archive.status_code == 200
    with ZipFile(BytesIO(archive.data)) as zip_file:
        assert zip_file.namelist() == [match_path.name]
        assert zip_file.read(match_path.name) == match_path.read_bytes()
    assert client.get(
        f"/admin/events/other-event/search-results/{token}/download/{photo_id}"
    ).status_code == 404


def test_admin_event_page_reports_live_index_percentage(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)
    indexing = app.extensions["indexing_service"]
    raw_dir = indexing.events_dir / "progress-event" / "raw"
    raw_dir.mkdir(parents=True)
    first = raw_dir / "first.png"
    second = raw_dir / "second.png"
    first.write_bytes(_png()[0].getvalue())
    second.write_bytes(_png()[0].getvalue())
    indexing.register_event("progress-event", "2026-07-21", "Progress Event")
    indexing.request_index("progress-event")
    assert indexing.store.claim_next_event() == ("progress-event", False)
    indexing.store.start_index_progress("progress-event", 2)
    indexing.store.record_progress_outcome(
        "progress-event",
        ImageIndexOutcome(str(first.resolve()), IndexStatus.INDEXED, face_count=1),
    )

    response = client.get("/admin/events/progress-event/index-progress")
    detail = client.get("/admin/events/progress-event")

    assert response.status_code == 200
    assert response.json == {
        "status": "indexing",
        "completed": 1,
        "total": 2,
        "percent": 50,
    }
    assert b'data-index-progress' in detail.data
    assert b">50%</strong>" in detail.data
    # Live-refresh hooks so per-photo statuses and counts advance without a reload.
    assert b"data-live-summary" in detail.data
    assert b"data-live-inventory" in detail.data
    assert b"data-live-pending" in detail.data


def _queued_event(app, event_id="hold-event", photo_count=2):
    indexing = app.extensions["indexing_service"]
    raw_dir = indexing.events_dir / event_id / "raw"
    raw_dir.mkdir(parents=True)
    for index in range(photo_count):
        (raw_dir / f"p{index}.png").write_bytes(_png()[0].getvalue())
    indexing.register_event(event_id, "2026-07-21", "Hold Event")
    indexing.request_index(event_id)
    return indexing


def test_admin_can_pause_and_resume_a_queued_event(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)
    indexing = _queued_event(app)

    paused = client.post(
        "/admin/events/hold-event/pause",
        data={"_csrf_token": _csrf(client)},
    )

    assert paused.status_code == 303
    assert indexing.get_event("hold-event").status is IndexStatus.PAUSED
    detail = client.get("/admin/events/hold-event")
    assert b"/admin/events/hold-event/resume" in detail.data
    assert b"status-paused" in detail.data
    assert app.extensions["admin_store"].recent_audit(1)[0]["action"] == "index_paused"

    resumed = client.post(
        "/admin/events/hold-event/resume",
        data={"_csrf_token": _csrf(client)},
    )

    assert resumed.status_code == 303
    assert indexing.get_event("hold-event").status is IndexStatus.QUEUED
    assert app.extensions["admin_store"].recent_audit(1)[0]["action"] == "index_resumed"


def test_admin_can_stop_a_queued_event_and_keep_it_stopped(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)
    indexing = _queued_event(app, event_id="stop-event")

    stopped = client.post(
        "/admin/events/stop-event/stop",
        data={"_csrf_token": _csrf(client)},
    )

    assert stopped.status_code == 303
    assert indexing.get_event("stop-event").status is IndexStatus.STOPPED
    detail = client.get("/admin/events/stop-event")
    assert b"status-stopped" in detail.data
    assert b"/admin/events/stop-event/resume" in detail.data
    assert app.extensions["admin_store"].recent_audit(1)[0]["action"] == "index_stopped"


def test_pause_unknown_event_returns_404(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)

    response = client.post(
        "/admin/events/missing/pause",
        data={"_csrf_token": _csrf(client)},
    )

    assert response.status_code == 404


def test_admin_can_add_list_and_email_organizers(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)
    _indexed_event(app, event_id="gala", display_name="Summer Gala")

    added = client.post(
        "/admin/events/gala/organizers",
        data={"_csrf_token": _csrf(client), "name": "Deemah", "email": "deemah@example.com"},
    )
    client.post(
        "/admin/events/gala/organizers",
        data={"_csrf_token": _csrf(client), "name": "Ali", "email": "ali@example.com"},
    )

    assert added.status_code == 303
    assert added.location.endswith("#hosts")
    detail = client.get("/admin/events/gala")
    assert b"Deemah" in detail.data
    assert b"ali@example.com" in detail.data
    # The "Email attendee code" mailto addresses both organizers and carries
    # the event's access code.
    code = app.extensions["indexing_service"].get_event_access_code("gala")
    assert b"mailto:deemah@example.com" in detail.data
    assert b"ali@example.com" in detail.data
    assert code[:4].encode() in detail.data
    assert app.extensions["admin_store"].recent_audit(1)[0]["action"] == "host_added"


def test_admin_organizer_email_is_validated(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)
    _indexed_event(app, event_id="gala")

    response = client.post(
        "/admin/events/gala/organizers",
        data={"_csrf_token": _csrf(client), "name": "Hessah", "email": "nope"},
    )

    assert response.status_code == 400
    assert b"valid email" in response.data
    assert app.extensions["indexing_service"].list_organizers("gala") == []


def test_admin_can_remove_an_organizer(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)
    _indexed_event(app, event_id="gala")
    indexing = app.extensions["indexing_service"]
    organizer = indexing.add_organizer("gala", "Deemah", "deemah@example.com")

    response = client.post(
        f"/admin/events/gala/organizers/{organizer.id}/delete",
        data={"_csrf_token": _csrf(client)},
    )

    assert response.status_code == 303
    assert indexing.list_organizers("gala") == []
    assert app.extensions["admin_store"].recent_audit(1)[0]["action"] == "host_removed"


def test_admin_photo_import_has_no_total_selection_limit_and_uses_safe_batches(tmp_path):
    script = (Path(__file__).parents[1] / "src" / "web" / "static" / "admin.js").read_text(
        encoding="utf-8"
    )
    app = _app(tmp_path)
    client = app.test_client()
    _login(client)
    client.post(
        "/admin/events",
        data={
            "_csrf_token": _csrf(client),
            "title": "Unlimited Uploads",
            "event_date": "2026-07-21",
        },
    )
    detail = client.get("/admin/events/unlimited-uploads")

    assert b"Select any number" in detail.data
    assert b"20 photos per upload" not in detail.data
    assert "MAX_FILES_PER_BATCH = 10" in script
    assert "MAX_BATCH_BYTES = 200 * 1024 * 1024" in script
