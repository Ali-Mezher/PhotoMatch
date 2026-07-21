"""Flask application factory for PhotoMatch's public attendee workflow."""

from __future__ import annotations

import json
import os
import secrets
import atexit
from dataclasses import dataclass
from datetime import timedelta
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
from src.services.admin_store import AdminStore
from src.services.clustering_service import ClusteringService
from src.services.indexing_service import IndexingService
from src.services.status_store import StatusStore
from src.services.work_coordinator import AdminWorkCoordinator

from .access import EventAccessGate
from .admin import admin
from .liveness import (
    LivenessChallengeStore,
    LivenessVerificationError,
    LivenessVerifier,
)
from .media import InvalidImageError, decode_selfie, render_watermarked_preview
from .result_store import SearchResultStore, StoredPhoto, StoredSearch
from .search_results import (
    event_raw_dir,
    filter_matches,
    is_allowed_path,
    require_photo,
    require_search,
    result_store,
)


@dataclass(frozen=True)
class PublicEvent:
    event_id: str
    event_date: str | None
    indexed: bool
    status: str
    display_name: str | None = None


def create_app(test_config: dict | None = None) -> Flask:
    """Create a configured Flask application for production or tests."""
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("PHOTOMATCH_SECRET_KEY") or secrets.token_hex(32),
        MAX_CONTENT_LENGTH=12 * 1024 * 1024,
        EVENTS_DIR=EVENTS_DIR,
        INDEX_STATUS_DB=INDEX_STATUS_DB,
        EVENT_ACCESS_TTL_SECONDS=30 * 60,
        EVENT_ACCESS_GATE=None,
        EVENT_ACCESS_RESOLVER=None,
        LIVENESS_CHALLENGE_TTL_SECONDS=2 * 60,
        LIVENESS_CHALLENGE_STORE=None,
        LIVENESS_VERIFIER=None,
        RESULT_TTL_SECONDS=15 * 60,
        MATCHER=match_selfie,
        EVENT_CATALOG=None,
        RESULT_STORE=None,
        ADMIN_USERNAME=os.environ.get("PHOTOMATCH_ADMIN_USERNAME"),
        ADMIN_PASSWORD=os.environ.get("PHOTOMATCH_ADMIN_PASSWORD"),
        ADMIN_START_WORKER=os.environ.get("PHOTOMATCH_BACKGROUND_WORKER", "1") == "1",
        ADMIN_DEBUG=os.environ.get("PHOTOMATCH_DEBUG", "0") == "1",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("PHOTOMATCH_SECURE_COOKIE", "0") == "1",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
        INDEXING_SERVICE=None,
        ADMIN_STORE=None,
        CLUSTERING_SERVICE=None,
        WORK_COORDINATOR=None,
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
    if app.config["LIVENESS_CHALLENGE_STORE"] is None:
        app.config["LIVENESS_CHALLENGE_STORE"] = LivenessChallengeStore(
            ttl_seconds=app.config["LIVENESS_CHALLENGE_TTL_SECONDS"]
        )
    if app.config["LIVENESS_VERIFIER"] is None:
        app.config["LIVENESS_VERIFIER"] = LivenessVerifier()

    admin_store = app.config["ADMIN_STORE"] or AdminStore(
        Path(app.config["INDEX_STATUS_DB"])
    )
    indexing = app.config["INDEXING_SERVICE"] or IndexingService(
        events_dir=Path(app.config["EVENTS_DIR"]),
        database_path=Path(app.config["INDEX_STATUS_DB"]),
    )
    if app.config["CLUSTERING_SERVICE"] is None:
        from src.indexing.build_index import load_event_index_snapshot

        events_dir = Path(app.config["EVENTS_DIR"])
        clustering = ClusteringService(
            admin_store,
            snapshot_loader=lambda event_id: load_event_index_snapshot(
                event_id, events_dir
            ),
        )
    else:
        clustering = app.config["CLUSTERING_SERVICE"]
    coordinator = app.config["WORK_COORDINATOR"] or AdminWorkCoordinator(
        indexing, clustering, admin_store
    )
    app.extensions["admin_store"] = admin_store
    app.extensions["indexing_service"] = indexing
    app.extensions["clustering_service"] = clustering
    app.extensions["admin_work_coordinator"] = coordinator
    app.extensions["admin_login_gate"] = EventAccessGate(
        ttl_seconds=8 * 60 * 60,
        max_failures=5,
        failure_window_seconds=15 * 60,
    )

    app.add_template_filter(lambda value: Path(value).name, "basename")

    if app.config["MATCHER"] is match_selfie:
        def runtime_matcher(image, event_id):
            settings = admin_store.get_settings()
            return match_selfie(
                image,
                event_id,
                top_k=settings.top_k,
                possible_threshold=settings.possible_threshold,
                confident_threshold=settings.confident_threshold,
            )

        app.config["MATCHER"] = runtime_matcher

    register_routes(app)
    app.register_blueprint(admin)
    register_security_headers(app)

    if app.config["ADMIN_START_WORKER"] and not app.config.get("TESTING"):
        should_start = not app.config["ADMIN_DEBUG"] or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
        if should_start:
            coordinator.start()
            atexit.register(coordinator.shutdown)

    return app


def register_security_headers(app: Flask) -> None:
    @app.after_request
    def apply_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; "
            "style-src 'self'; script-src 'self'; connect-src 'self'; "
            "form-action 'self'; frame-ancestors 'none'; base-uri 'self'",
        )
        return response


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

        challenge_token = request.form.get("liveness_token", "")
        challenge = _liveness_challenge_store().consume(
            challenge_token, event_id, access_token
        )
        if challenge is None:
            return _render_event_form(
                event,
                access_token,
                error="The live-selfie check expired or was already used. Restart it.",
                status=400,
            )

        frame_fields = ("liveness_front", "liveness_turn", "liveness_return")
        uploads = [request.files.get(field) for field in frame_fields]
        if any(upload is None or not upload.filename for upload in uploads):
            return _render_event_form(
                event,
                access_token,
                error="Complete all 3 live-selfie steps before searching.",
                status=400,
            )

        try:
            frames = [decode_selfie(upload) for upload in uploads]
            selfie = _liveness_verifier().verify(
                frames[0], frames[1], frames[2], challenge.direction
            )
            raw_matches = current_app.config["MATCHER"](selfie, event_id)
        except InvalidImageError as exc:
            return _render_event_form(
                event,
                access_token,
                error=f"A live-selfie frame was unreadable. {exc} Restart the check.",
                status=400,
            )
        except LivenessVerificationError as exc:
            return _render_event_form(
                event, access_token, error=str(exc), status=422
            )
        except NoFaceDetectedError:
            return _render_event_form(
                event,
                access_token,
                error=(
                    "No face was found in the verified selfie. Restart the live "
                    "check or ask staff for manual photo lookup."
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
        stored = _result_store().create(event_id, safe_matches, audience="public")
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
                error=(
                    "The live-selfie capture was too large. Restart the check or "
                    "ask staff for manual photo lookup."
                ),
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
    challenge = _liveness_challenge_store().issue(event.event_id, access_token)
    return (
        render_template(
            "home.html",
            error=error,
            entered_code="",
            unlocked_event=event,
            access_token=access_token,
            liveness_challenge=challenge,
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
                display_name=summary.display_name if summary else event_id,
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
    return filter_matches(event_id, matches)


def _require_search(token: str) -> StoredSearch:
    return require_search(token, "public")


def _require_photo(token: str, photo_id: str) -> tuple[StoredSearch, StoredPhoto]:
    return require_photo(token, photo_id, "public")


def _event_raw_dir(event_id: str) -> Path:
    return event_raw_dir(event_id)


def _is_allowed_path(path: Path, raw_dir: Path) -> bool:
    return is_allowed_path(path, raw_dir)


def _result_store() -> SearchResultStore:
    return result_store()


def _event_access_gate() -> EventAccessGate:
    return current_app.config["EVENT_ACCESS_GATE"]


def _liveness_challenge_store() -> LivenessChallengeStore:
    return current_app.config["LIVENESS_CHALLENGE_STORE"]


def _liveness_verifier() -> LivenessVerifier:
    return current_app.config["LIVENESS_VERIFIER"]
