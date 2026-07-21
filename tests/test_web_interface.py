from __future__ import annotations

import sqlite3
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from src.matching import NoFaceDetectedError
from src.services.status_store import (
    EVENT_ACCESS_CODE_LENGTH,
    StatusStore,
    normalize_event_access_code,
)
from src.web import PublicEvent, create_app
from src.web.access import EventAccessGate
from src.web.result_store import SearchResultStore


READY_CODE = "A1B2C3D4"
PENDING_CODE = "DEADBEEF"


def _image_bytes(color=(35, 120, 180), size=(120, 90), image_format="JPEG"):
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format=image_format)
    output.seek(0)
    return output


@pytest.fixture
def web_setup(tmp_path):
    events_dir = tmp_path / "events"
    raw_dir = events_dir / "graduation" / "raw"
    raw_dir.mkdir(parents=True)
    first = raw_dir / "first.jpg"
    second = raw_dir / "second.jpg"
    first.write_bytes(_image_bytes(color=(30, 80, 140)).getvalue())
    second.write_bytes(_image_bytes(color=(140, 70, 30)).getvalue())

    calls = []

    def matcher(image, event_id):
        calls.append((image, event_id))
        return {
            "confident": [SimpleNamespace(photo_path=str(first), score=0.91)],
            "possible": [SimpleNamespace(photo_path=str(second), score=0.58)],
        }

    events = [
        PublicEvent("graduation", "2026-07-01", True, "indexed"),
        PublicEvent("conference", "2026-07-02", False, "pending"),
    ]
    event_codes = {READY_CODE: "graduation", PENDING_CODE: "conference"}
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "EVENTS_DIR": events_dir,
            "INDEX_STATUS_DB": tmp_path / "status.sqlite3",
            "EVENT_CATALOG": lambda: events,
            "EVENT_ACCESS_RESOLVER": lambda code: event_codes.get(
                normalize_event_access_code(code)
            ),
            "EVENT_ACCESS_GATE": EventAccessGate(ttl_seconds=900),
            "MATCHER": matcher,
            "RESULT_STORE": SearchResultStore(ttl_seconds=900),
        }
    )
    return app, app.test_client(), calls, first, second


def _unlock(client, event_code=READY_CODE):
    return client.post("/event-access", data={"event_code": event_code})


def _search(client, **overrides):
    unlocked = _unlock(client)
    assert unlocked.status_code == 303
    data = {
        "consent": "yes",
        "selfie": (_image_bytes(), "selfie.jpg"),
    }
    data.update(overrides)
    return client.post(
        f'{unlocked.headers["Location"]}/search',
        data=data,
        content_type="multipart/form-data",
    )


def _stored_result(app, location):
    token = location.rsplit("/", 1)[-1]
    stored = app.config["RESULT_STORE"].get(token)
    assert stored is not None
    return stored


def test_home_hides_event_catalog_and_unlocks_only_the_matching_event(web_setup):
    _, client, _, _, _ = web_setup

    home = client.get("/")

    assert home.status_code == 200
    assert b"Enter Event Code" in home.data
    assert b"Events are not listed publicly" in home.data
    assert b"graduation" not in home.data
    assert b"conference" not in home.data
    assert b"manual photo lookup" in home.data

    unlocked = _unlock(client, "a1b2-c3d4")
    assert unlocked.status_code == 303
    assert READY_CODE not in unlocked.headers["Location"]
    assert "graduation" not in unlocked.headers["Location"]

    search_form = client.get(unlocked.headers["Location"])
    assert search_form.status_code == 200
    assert b"graduation" in search_form.data
    assert b"conference" not in search_form.data
    assert b"I understand and consent" in search_form.data
    assert b"Use Camera" in search_form.data
    assert b'capture="user"' in search_form.data


def test_invalid_unready_and_expired_event_access_are_actionable(web_setup):
    app, client, _, _, _ = web_setup

    invalid = _unlock(client, "00000000")
    unready = _unlock(client, PENDING_CODE)
    grant = app.config["EVENT_ACCESS_GATE"].grant("graduation")
    app.config["EVENT_ACCESS_GATE"] = EventAccessGate(ttl_seconds=10)
    expired = client.get(f"/event/{grant.token}")

    assert invalid.status_code == 400
    assert b"not recognized" in invalid.data
    assert unready.status_code == 409
    assert b"not ready" in unready.data
    assert expired.status_code == 410
    assert b"expired" in expired.data


def test_event_code_failures_are_throttled_and_recover_after_window(web_setup):
    app, client, _, _, _ = web_setup
    now = {"value": 100.0}
    app.config["EVENT_ACCESS_GATE"] = EventAccessGate(
        ttl_seconds=900,
        max_failures=2,
        failure_window_seconds=60,
        clock=lambda: now["value"],
    )

    assert _unlock(client, "00000000").status_code == 400
    assert _unlock(client, "00000001").status_code == 400
    blocked = _unlock(client, READY_CODE)
    assert blocked.status_code == 429
    assert b"Wait 1 minute" in blocked.data

    now["value"] = 161.0
    assert _unlock(client, READY_CODE).status_code == 303


def test_search_requires_an_unlock_token_consent_and_selfie(web_setup):
    _, client, calls, _, _ = web_setup

    direct = client.post(
        "/event/graduation/search",
        data={"event_id": "graduation", "consent": "yes"},
    )
    missing_consent = _search(client, consent="")
    unlocked = _unlock(client)
    missing_selfie = client.post(
        f'{unlocked.headers["Location"]}/search',
        data={"consent": "yes"},
        content_type="multipart/form-data",
    )

    assert direct.status_code == 410
    assert missing_consent.status_code == 400
    assert missing_selfie.status_code == 400
    assert calls == []


def test_valid_search_decodes_in_memory_and_uses_opaque_result_urls(web_setup):
    app, client, calls, first, _ = web_setup

    response = _search(client)
    assert response.status_code == 303
    assert response.headers["Location"].startswith("/results/")

    results = client.get(response.headers["Location"])
    assert results.status_code == 200
    assert b"Confident Matches" in results.data
    assert b"Possible Matches" in results.data
    assert str(first).encode() not in results.data
    assert len(calls) == 1
    image, event_id = calls[0]
    assert event_id == "graduation"
    assert isinstance(image, np.ndarray)
    assert image.shape == (90, 120, 3)
    assert _stored_result(app, response.headers["Location"]).event_id == "graduation"


def test_invalid_image_and_no_face_errors_are_actionable(web_setup):
    app, client, _, _, _ = web_setup
    invalid = _search(client, selfie=(BytesIO(b"not an image"), "selfie.jpg"))
    app.config["MATCHER"] = lambda image, event_id: (_ for _ in ()).throw(
        NoFaceDetectedError()
    )
    no_face = _search(client)

    assert invalid.status_code == 400
    assert b"not a readable image" in invalid.data
    assert no_face.status_code == 422
    assert b"No face was found" in no_face.data
    assert b"manual photo lookup" in no_face.data


def test_matching_failure_and_empty_results_have_recovery_paths(web_setup):
    app, client, _, _, _ = web_setup
    app.config["MATCHER"] = lambda image, event_id: (_ for _ in ()).throw(
        RuntimeError("internal details")
    )
    failed = _search(client)
    app.config["MATCHER"] = lambda image, event_id: {
        "confident": [],
        "possible": [],
    }
    empty_search = _search(client)
    empty_results = client.get(empty_search.headers["Location"])

    assert failed.status_code == 500
    assert b"internal details" not in failed.data
    assert b"manual photo lookup" in failed.data
    assert b"No Matches Found" in empty_results.data
    assert b"Try Another Selfie" in empty_results.data


def test_preview_is_watermarked_and_original_is_unchanged(web_setup):
    app, client, _, first, _ = web_setup
    original = first.read_bytes()
    search_response = _search(client)
    stored = _stored_result(app, search_response.headers["Location"])
    photo = stored.confident[0]

    response = client.get(f"/preview/{stored.token}/{photo.photo_id}?size=full")

    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.data != original
    assert first.read_bytes() == original
    with Image.open(BytesIO(response.data)) as preview:
        assert preview.size == (120, 90)


def test_single_and_zip_downloads_return_only_selected_originals(web_setup):
    app, client, _, first, second = web_setup
    search_response = _search(client)
    stored = _stored_result(app, search_response.headers["Location"])
    confident = stored.confident[0]
    possible = stored.possible[0]

    single = client.get(f"/download/{stored.token}/{confident.photo_id}")
    archive = client.post(
        f"/export/{stored.token}", data={"photo_ids": [possible.photo_id]}
    )

    assert single.status_code == 200
    single_bytes = single.get_data()
    single.close()
    assert single_bytes == first.read_bytes()
    assert archive.status_code == 200
    with ZipFile(BytesIO(archive.data)) as zip_file:
        assert zip_file.namelist() == [second.name]
        assert zip_file.read(second.name) == second.read_bytes()

    no_selection = client.post(f"/export/{stored.token}", data={})
    assert no_selection.status_code == 400
    assert b"Select at least 1 photo" in no_selection.data


def test_result_membership_and_event_containment_are_enforced(web_setup, tmp_path):
    app, client, _, _, _ = web_setup
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(_image_bytes().getvalue())
    app.config["MATCHER"] = lambda image, event_id: {
        "confident": [SimpleNamespace(photo_path=str(outside), score=0.99)],
        "possible": [],
    }
    search_response = _search(client)
    stored = _stored_result(app, search_response.headers["Location"])

    assert stored.photos == {}
    assert client.get(f"/preview/{stored.token}/made-up").status_code == 404
    assert client.post(
        f"/export/{stored.token}", data={"photo_ids": "made-up"}
    ).status_code == 404


def test_expired_result_tokens_cannot_preview_or_download(web_setup):
    app, client, _, first, _ = web_setup
    now = {"value": 100.0}
    store = SearchResultStore(ttl_seconds=10, clock=lambda: now["value"])
    app.config["RESULT_STORE"] = store
    stored = store.create(
        "graduation",
        {
            "confident": [SimpleNamespace(photo_path=str(first), score=0.9)],
            "possible": [],
        },
    )
    photo = stored.confident[0]
    now["value"] = 111.0

    assert client.get(f"/results/{stored.token}").status_code == 410
    assert client.get(f"/preview/{stored.token}/{photo.photo_id}").status_code == 410
    assert b"Start a New Search" in client.get(f"/results/{stored.token}").data


def test_sql_access_codes_are_unique_persistent_and_accept_grouping(tmp_path):
    database = tmp_path / "status.sqlite3"
    store = StatusStore(database)
    store.register_event("first", "2026-07-01")
    store.register_event("second", "2026-07-02")

    first_code = store.get_event_access_code("first")
    second_code = store.get_event_access_code("second")

    assert first_code is not None
    assert len(first_code) == EVENT_ACCESS_CODE_LENGTH
    assert set(first_code) <= set("0123456789ABCDEF")
    assert first_code != second_code
    assert StatusStore(database).get_event_access_code("first") == first_code
    grouped = f"{first_code[:4]}-{first_code[4:]}"
    assert store.find_event_id_by_access_code(grouped.lower()) == "first"


def test_existing_sql_database_is_backfilled_with_event_codes(tmp_path):
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
            "INSERT INTO events(event_id, event_date, status) VALUES (?, ?, ?)",
            ("legacy", "2026-06-01", "indexed"),
        )

    store = StatusStore(database)

    code = store.get_event_access_code("legacy")
    assert code is not None
    assert store.find_event_id_by_access_code(code) == "legacy"


def test_default_catalog_uses_sql_status_and_codes(tmp_path):
    events_dir = tmp_path / "events"
    ready_index = events_dir / "ready" / "indexed"
    ready_index.mkdir(parents=True)
    (ready_index / "faces.faiss").write_bytes(b"index")
    (ready_index / "metadata.json").write_text("[]", encoding="utf-8")
    (events_dir / "waiting" / "raw").mkdir(parents=True)
    updating_index = events_dir / "updating" / "indexed"
    updating_index.mkdir(parents=True)
    (updating_index / "faces.faiss").write_bytes(b"old-index")
    (updating_index / "metadata.json").write_text("[]", encoding="utf-8")
    database = tmp_path / "status.sqlite3"
    store = StatusStore(database)
    store.register_event("ready", "2026-07-01")
    store.complete_event("ready", "1")
    store.register_event("waiting", "2026-07-02")
    store.register_event("updating", "2026-07-03")

    app = create_app(
        {
            "TESTING": True,
            "EVENTS_DIR": events_dir,
            "INDEX_STATUS_DB": database,
            "MATCHER": lambda image, event_id: {},
        }
    )
    client = app.test_client()

    home = client.get("/")
    assert b"ready" not in home.data
    assert b"waiting" not in home.data
    assert _unlock(client, store.get_event_access_code("ready")).status_code == 303
    assert _unlock(client, store.get_event_access_code("waiting")).status_code == 409
    assert _unlock(client, store.get_event_access_code("updating")).status_code == 409


def test_default_catalog_rejects_an_unsafe_active_generation(tmp_path):
    events_dir = tmp_path / "events"
    index_dir = events_dir / "unsafe" / "indexed"
    index_dir.mkdir(parents=True)
    (index_dir / "active.json").write_text(
        '{"generation": "../../ready/indexed"}', encoding="utf-8"
    )
    database = tmp_path / "status.sqlite3"
    store = StatusStore(database)
    store.register_event("unsafe", "2026-07-03")

    app = create_app(
        {
            "TESTING": True,
            "EVENTS_DIR": events_dir,
            "INDEX_STATUS_DB": database,
            "MATCHER": lambda image, event_id: {},
        }
    )

    response = _unlock(app.test_client(), store.get_event_access_code("unsafe"))

    assert response.status_code == 409
    assert b"not ready" in response.data


def test_oversized_upload_is_rejected_before_matching(web_setup):
    app, client, calls, _, _ = web_setup
    app.config["MAX_CONTENT_LENGTH"] = 200

    response = _search(client)

    assert response.status_code == 413
    assert b"smaller than 12 MB" in response.data
    assert calls == []
