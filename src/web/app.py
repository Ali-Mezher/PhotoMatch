"""Flask application factory for PhotoMatch's public attendee workflow."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from flask import (
    Flask,
    abort,
    current_app,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from config import (
    EVENTS_DIR,
    EVENT_INDEXED_SUBDIR,
    EVENT_RAW_SUBDIR,
    INDEX_STATUS_DB,
)
from src.indexing.build_index import (
    ACTIVE_INDEX_FILENAME,
    GENERATIONS_DIRNAME,
)
from src.indexing.faiss_index import INDEX_FILENAME, METADATA_FILENAME
from src.matching import EventNotIndexedError, NoFaceDetectedError, match_selfie
from src.services import IndexStatus
from src.services.status_store import StatusStore

from .access import EventAccessGate
from .media import InvalidImageError, decode_selfie, render_watermarked_preview
from .result_store import SearchResultStore, StoredPhoto, StoredSearch


@dataclass(frozen=True)
class PublicEvent:
    event_id: str
    event_date: str | None
    indexed: bool
    status: str


def create_app(test_config: dict | None = None) -> Flask:
    """Create a configured Flask application for production or tests."""
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=secrets.token_hex(32),
        MAX_CONTENT_LENGTH=12 * 1024 * 1024,
        EVENTS_DIR=EVENTS_DIR,
        INDEX_STATUS_DB=INDEX_STATUS_DB,
        EVENT_ACCESS_TTL_SECONDS=30 * 60,
        EVENT_ACCESS_GATE=None,
        EVENT_ACCESS_RESOLVER=None,
        RESULT_TTL_SECONDS=15 * 60,
        MATCHER=match_selfie,
        EVENT_CATALOG=None,
        RESULT_STORE=None,
    )
    if test_config:
        app.config.update(test_config)

    if app.config["RESULT_STORE"] is None:
        app.config["RESULT_STORE"] = SearchResultStore(
            ttl_seconds=app.config["RESULT_TTL_SECONDS"]
        )
    if app.config["EVENT_ACCESS_GATE"] is None:
        app.config["EVENT_ACCESS_GATE"] = EventAccessGate(
            ttl_seconds=app.config["EVENT_ACCESS_TTL_SECONDS"]
        )

    register_routes(app)
    return app


def register_routes(app: Flask) -> None:
    @app.get("/")
    def home():
        return _render_home()

    @app.post("/event-access")
    def event_access():
        client_key = request.remote_addr or "unknown"
        gate = _event_access_gate()
        if not gate.can_attempt(client_key):
            return _render_home(
                error=(
                    "Too many event-code attempts. Wait 1 minute, then try again "
                    "or ask staff to confirm the code."
                ),
                status=429,
            )

        entered_code = request.form.get("event_code", "").strip()
        if not entered_code:
            return _render_home(
                error="Enter the event code provided by staff.", status=400
            )

        event_id = _resolve_event_access_code(entered_code)
        event = _find_event(event_id) if event_id else None
        if event is None:
            gate.record_failure(client_key)
            return _render_home(
                error="That event code was not recognized. Check the code and try again.",
                entered_code=entered_code,
                status=400,
            )
        if not event.indexed:
            gate.clear_failures(client_key)
            return _render_home(
                error=(
                    "That event is not ready for search yet. Ask staff when "
                    "indexing will be complete."
                ),
                status=409,
            )

        gate.clear_failures(client_key)
        grant = gate.grant(event.event_id)
        return redirect(url_for("unlocked_event", access_token=grant.token), code=303)

    @app.get("/event/<access_token>")
    def unlocked_event(access_token: str):
        grant = _event_access_gate().resolve(access_token)
        if grant is None:
            return _render_home(
                error="This event access has expired. Enter the event code again.",
                status=410,
            )
        event = _find_event(grant.event_id)
        if event is None or not event.indexed:
            return _render_home(
                error="That event is no longer ready for search. Ask staff for help.",
                status=409,
            )
        return _render_event_form(event, access_token)

    @app.post("/event/<access_token>/search")
    def search(access_token: str):
        grant = _event_access_gate().resolve(access_token)
        if grant is None:
            return _render_home(
                error="This event access has expired. Enter the event code again.",
                status=410,
            )
        event = _find_event(grant.event_id)
        if event is None or not event.indexed:
            return _render_home(
                error="That event is no longer ready for search. Ask staff for help.",
                status=409,
            )
        event_id = event.event_id

        if request.form.get("consent") != "yes":
            return _render_event_form(
                event,
                access_token,
                error=(
                    "Consent is required for biometric search. You can instead "
                    "ask staff for manual photo lookup."
                ),
                status=400,
            )

        upload = request.files.get("selfie")
        if upload is None or not upload.filename:
            return _render_event_form(
                event,
                access_token,
                error="Choose a selfie image, then try the search again.",
                status=400,
            )

        try:
            selfie = decode_selfie(upload)
            raw_matches = current_app.config["MATCHER"](selfie, event_id)
        except InvalidImageError as exc:
            return _render_event_form(
                event, access_token, error=str(exc), status=400
            )
        except NoFaceDetectedError:
            return _render_event_form(
                event,
                access_token,
                error=(
                    "No face was found in that selfie. Use a clear, front-facing "
                    "photo or ask staff for manual photo lookup."
                ),
                status=422,
            )
        except EventNotIndexedError:
            return _render_event_form(
                event,
                access_token,
                error="That event is no longer ready for search. Ask staff for help.",
                status=409,
            )
        except Exception:  # noqa: BLE001 - keep internal details out of public errors
            current_app.logger.exception("Photo matching failed")
            return _render_event_form(
                event,
                access_token,
                error=(
                    "The search could not be completed. Try again or ask staff "
                    "for manual photo lookup."
                ),
                status=500,
            )

        safe_matches = _filter_matches(event_id, raw_matches)
        stored = _result_store().create(event_id, safe_matches)
        return redirect(url_for("results", token=stored.token), code=303)

    @app.get("/results/<token>")
    def results(token: str):
        stored = _require_search(token)
        return render_template("results.html", search=stored)

    @app.get("/preview/<token>/<photo_id>")
    def preview(token: str, photo_id: str):
        _, photo = _require_photo(token, photo_id)
        size = request.args.get("size", "thumb")
        max_size = (1800, 1400) if size == "full" else (720, 540)
        try:
            preview_bytes = render_watermarked_preview(photo.path, max_size)
        except OSError:
            abort(404)
        response = send_file(preview_bytes, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "private, no-store"
        return response

    @app.get("/download/<token>/<photo_id>")
    def download(token: str, photo_id: str):
        _, photo = _require_photo(token, photo_id)
        return send_file(
            photo.path,
            as_attachment=True,
            download_name=photo.path.name,
            max_age=0,
        )

    @app.post("/export/<token>")
    def export(token: str):
        stored = _require_search(token)
        selected_ids = list(dict.fromkeys(request.form.getlist("photo_ids")))
        if not selected_ids:
            return (
                render_template(
                    "results.html",
                    search=stored,
                    error="Select at least 1 photo before downloading a ZIP.",
                ),
                400,
            )

        selected: list[StoredPhoto] = []
        for photo_id in selected_ids:
            _, photo = _require_photo(token, photo_id)
            selected.append(photo)

        archive = BytesIO()
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zip_file:
            for photo in selected:
                zip_file.write(photo.path, arcname=photo.path.name)
        archive.seek(0)
        return send_file(
            archive,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"photomatch-{stored.event_id}.zip",
        )

    @app.errorhandler(413)
    def upload_too_large(_error):
        access_token = (request.view_args or {}).get("access_token")
        grant = _event_access_gate().resolve(access_token) if access_token else None
        event = _find_event(grant.event_id) if grant else None
        if event is not None:
            return _render_event_form(
                event,
                access_token,
                error="That selfie is too large. Choose an image smaller than 12 MB.",
                status=413,
            )
        return _render_home(
            error="That selfie is too large. Choose an image smaller than 12 MB.",
            status=413,
        )

    @app.errorhandler(410)
    def expired_search(_error):
        return render_template("expired.html"), 410


def _render_home(
    error: str | None = None,
    entered_code: str = "",
    status: int = 200,
):
    return (
        render_template(
            "home.html",
            error=error,
            entered_code=entered_code,
            unlocked_event=None,
            access_token=None,
        ),
        status,
    )


def _render_event_form(
    event: PublicEvent,
    access_token: str,
    error: str | None = None,
    status: int = 200,
):
    return (
        render_template(
            "home.html",
            error=error,
            entered_code="",
            unlocked_event=event,
            access_token=access_token,
        ),
        status,
    )


def _events() -> list[PublicEvent]:
    provider = current_app.config.get("EVENT_CATALOG")
    if provider is not None:
        return provider()
    return _default_event_catalog(
        Path(current_app.config["EVENTS_DIR"]),
        Path(current_app.config["INDEX_STATUS_DB"]),
    )


def _find_event(event_id: str) -> PublicEvent | None:
    return next((event for event in _events() if event.event_id == event_id), None)


def _resolve_event_access_code(access_code: str) -> str | None:
    resolver = current_app.config.get("EVENT_ACCESS_RESOLVER")
    if resolver is not None:
        return resolver(access_code)
    return StatusStore(
        Path(current_app.config["INDEX_STATUS_DB"])
    ).find_event_id_by_access_code(access_code)


def _default_event_catalog(
    events_dir: Path, database_path: Path
) -> list[PublicEvent]:
    summaries = {event.event_id: event for event in StatusStore(database_path).list_events()}
    event_ids = set(summaries)
    if events_dir.exists():
        event_ids.update(path.name for path in events_dir.iterdir() if path.is_dir())

    events = []
    for event_id in event_ids:
        summary = summaries.get(event_id)
        index_files_ready = _index_ready(
            events_dir / event_id / EVENT_INDEXED_SUBDIR
        )
        indexed = index_files_ready and (
            summary is None or summary.status == IndexStatus.INDEXED
        )
        events.append(
            PublicEvent(
                event_id=event_id,
                event_date=summary.event_date if summary else None,
                indexed=indexed,
                status=(
                    summary.status.value
                    if summary
                    else IndexStatus.INDEXED.value if indexed else "not_indexed"
                ),
            )
        )
    return sorted(events, key=lambda item: (item.event_date or "9999-12-31", item.event_id))


def _index_ready(index_dir: Path) -> bool:
    active_dir = index_dir
    manifest = index_dir / ACTIVE_INDEX_FILENAME
    if manifest.exists():
        try:
            generation = json.loads(manifest.read_text(encoding="utf-8"))["generation"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return False
        if (
            not isinstance(generation, str)
            or len(generation) != 32
            or any(character not in "0123456789abcdef" for character in generation)
        ):
            return False
        active_dir = index_dir / GENERATIONS_DIRNAME / generation
    return (active_dir / INDEX_FILENAME).is_file() and (
        active_dir / METADATA_FILENAME
    ).is_file()


def _filter_matches(event_id: str, matches: dict) -> dict:
    raw_dir = _event_raw_dir(event_id)
    filtered = {"confident": [], "possible": []}
    for tier in filtered:
        for match in matches.get(tier, []):
            path = Path(match.photo_path).resolve()
            if _is_allowed_path(path, raw_dir) and path.is_file():
                filtered[tier].append(match)
            else:
                current_app.logger.warning(
                    "Ignored match outside event directory for event %s", event_id
                )
    return filtered


def _require_search(token: str) -> StoredSearch:
    stored = _result_store().get(token)
    if stored is None:
        abort(410, description="This search has expired. Start a new search.")
    return stored


def _require_photo(token: str, photo_id: str) -> tuple[StoredSearch, StoredPhoto]:
    stored = _require_search(token)
    photo = stored.photos.get(photo_id)
    if photo is None:
        abort(404)
    if not _is_allowed_path(photo.path, _event_raw_dir(stored.event_id)):
        abort(404)
    if not photo.path.is_file():
        abort(404)
    return stored, photo


def _event_raw_dir(event_id: str) -> Path:
    events_dir = Path(current_app.config["EVENTS_DIR"]).resolve()
    raw_dir = (events_dir / event_id / EVENT_RAW_SUBDIR).resolve()
    if raw_dir.parent.parent != events_dir:
        abort(404)
    return raw_dir


def _is_allowed_path(path: Path, raw_dir: Path) -> bool:
    try:
        path.relative_to(raw_dir)
    except ValueError:
        return False
    return True


def _result_store() -> SearchResultStore:
    return current_app.config["RESULT_STORE"]


def _event_access_gate() -> EventAccessGate:
    return current_app.config["EVENT_ACCESS_GATE"]
