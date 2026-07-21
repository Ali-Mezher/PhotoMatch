"""Authenticated operator dashboard routes."""

from __future__ import annotations

import hmac
import secrets
from collections import defaultdict
from functools import wraps
from io import BytesIO
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from PIL import Image, ImageOps

from config import EMBEDDING_MODEL, EVENT_RAW_SUBDIR, INDEX_PIPELINE_VERSION
from src.indexing.build_index import active_index_generation
from src.services.admin_store import RuntimeSettings
from src.services.event_import import create_event, import_photos
from src.services.models import IndexStatus

from .access import EventAccessGate

admin = Blueprint("admin", __name__, url_prefix="/admin")


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("admin_authenticated"):
            return redirect(url_for("admin.login", next=request.path), code=303)
        return view(*args, **kwargs)

    return wrapped


def csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


@admin.before_request
def protect_admin_requests():
    if request.endpoint == "admin.upload_photos":
        request.max_content_length = 500 * 1024 * 1024
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        supplied = request.form.get("_csrf_token") or request.headers.get("X-CSRF-Token")
        expected = session.get("csrf_token", "")
        if not supplied or not expected or not hmac.compare_digest(supplied, expected):
            abort(400, description="The form expired. Refresh the page and try again.")


@admin.context_processor
def admin_template_context():
    return {"csrf_token": csrf_token}


@admin.get("/login")
def login():
    if session.get("admin_authenticated"):
        return redirect(url_for("admin.overview"), code=303)
    return render_template(
        "admin/login.html",
        configured=bool(
            current_app.config.get("ADMIN_USERNAME")
            and current_app.config.get("ADMIN_PASSWORD")
        ),
    )


@admin.post("/login")
def login_submit():
    client_key = request.remote_addr or "unknown"
    gate = current_app.extensions["admin_login_gate"]
    if not gate.can_attempt(client_key):
        return render_template("admin/login.html", configured=True, error="Too many attempts. Try again in 15 minutes."), 429

    username = current_app.config.get("ADMIN_USERNAME")
    password = current_app.config.get("ADMIN_PASSWORD")
    supplied_username = request.form.get("username", "")
    supplied_password = request.form.get("password", "")
    valid = bool(username and password)
    if valid:
        valid = hmac.compare_digest(supplied_username, username) and hmac.compare_digest(
            supplied_password, password
        )
    if not valid:
        gate.record_failure(client_key)
        _admin_store().record_audit("login_failed")
        return render_template(
            "admin/login.html",
            configured=bool(username and password),
            error="The username or password was not recognized.",
        ), 401

    gate.clear_failures(client_key)
    session.clear()
    session["admin_authenticated"] = True
    session["admin_username"] = username
    session["csrf_token"] = secrets.token_urlsafe(32)
    session.permanent = True
    _admin_store().record_audit("login_succeeded")
    return redirect(url_for("admin.overview"), code=303)


@admin.post("/logout")
@admin_required
def logout():
    _admin_store().record_audit("logout")
    session.clear()
    return redirect(url_for("admin.login"), code=303)


@admin.get("")
@admin.get("/")
@admin_required
def overview():
    indexing = _indexing()
    events = indexing.list_events()
    clusters = {event.event_id: _admin_store().latest_cluster(event.event_id) for event in events}
    stale_clusters = {
        event.event_id: bool(
            clusters[event.event_id]
            and clusters[event.event_id].status == "complete"
            and clusters[event.event_id].index_generation
            != active_index_generation(event.event_id, indexing.events_dir)
        )
        for event in events
    }
    totals = {
        "events": len(events),
        "photos": sum(event.total_images for event in events),
        "ready": sum(event.status == IndexStatus.INDEXED for event in events),
        "attention": sum(
            event.status == IndexStatus.FAILED or event.failed_images > 0 for event in events
        ),
    }
    return render_template(
        "admin/overview.html",
        events=events,
        clusters=clusters,
        stale_clusters=stale_clusters,
        access_codes={
            event.event_id: indexing.get_event_access_code(event.event_id)
            for event in events
        },
        totals=totals,
        audit=_admin_store().recent_audit(),
    )


@admin.get("/events/new")
@admin_required
def new_event():
    return render_template("admin/new_event.html")


@admin.post("/events")
@admin_required
def create_event_route():
    title = request.form.get("title", "")
    event_date = request.form.get("event_date", "")
    try:
        event = create_event(_indexing(), title, event_date)
    except (OSError, ValueError) as exc:
        return render_template(
            "admin/new_event.html", title=title, event_date=event_date, error=str(exc)
        ), 400
    _admin_store().record_audit("event_created", event.event_id, title=event.display_name, event_date=event.event_date)
    flash("Event created. Add photos to begin indexing.", "success")
    return redirect(url_for("admin.event_detail", event_id=event.event_id), code=303)


@admin.get("/events/<event_id>")
@admin_required
def event_detail(event_id: str):
    try:
        event = _indexing().get_event(event_id)
    except (KeyError, ValueError):
        abort(404)
    return render_template(
        "admin/event_detail.html",
        event=event,
        images=_indexing().list_image_statuses(event_id),
        access_code=_indexing().get_event_access_code(event_id),
        cluster=_admin_store().latest_cluster(event_id),
        cluster_stale=_cluster_is_stale(event_id),
    )


@admin.post("/events/<event_id>/photos")
@admin_required
def upload_photos(event_id: str):
    try:
        event = _indexing().get_event(event_id)
    except (KeyError, ValueError):
        abort(404)
    files = request.files.getlist("photos")
    if not files:
        return jsonify({"error": "Choose at least one photo."}), 400
    raw_dir = (_indexing().events_dir / event_id / EVENT_RAW_SUBDIR).resolve()
    outcomes = import_photos(raw_dir, files)
    imported = sum(outcome.imported for outcome in outcomes)
    failed = len(outcomes) - imported
    final_batch = request.form.get("final_batch", "1") == "1"
    if final_batch:
        queued = _indexing().request_index(event_id)
        if queued or imported:
            _coordinator().signal()
    _admin_store().record_audit(
        "photos_imported", event.event_id, imported=imported, rejected=failed, final_batch=final_batch
    )
    payload = {
        "imported": imported,
        "rejected": failed,
        "outcomes": [outcome.__dict__ for outcome in outcomes],
        "redirect": url_for("admin.event_detail", event_id=event_id),
    }
    if request.accept_mimetypes.best == "application/json" or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify(payload), 200 if imported else 400
    flash(f"Imported {imported} photo(s); {failed} rejected.", "success" if imported else "error")
    return redirect(payload["redirect"], code=303)


@admin.post("/events/<event_id>/index")
@admin_required
def index_event(event_id: str):
    try:
        queued = _indexing().request_index(event_id)
    except (KeyError, ValueError):
        abort(404)
    _coordinator().signal()
    _admin_store().record_audit("index_requested", event_id, queued=queued)
    flash("Indexing request queued." if queued else "The event index is already current.", "success")
    return redirect(url_for("admin.event_detail", event_id=event_id), code=303)


@admin.post("/events/<event_id>/retry")
@admin_required
def retry_event(event_id: str):
    try:
        count = _indexing().retry_failed(event_id)
    except (KeyError, ValueError):
        abort(404)
    _coordinator().signal()
    _admin_store().record_audit("index_retry_requested", event_id, photo_count=count)
    flash(f"Queued {count} failed photo(s) for another attempt.", "success")
    return redirect(url_for("admin.event_detail", event_id=event_id), code=303)


@admin.post("/events/<event_id>/cluster")
@admin_required
def cluster_event(event_id: str):
    try:
        event = _indexing().get_event(event_id)
    except (KeyError, ValueError):
        abort(404)
    if event.status != IndexStatus.INDEXED or event.pending_images or event.failed_images:
        flash("Finish indexing this event before clustering it.", "error")
        return redirect(url_for("admin.event_detail", event_id=event_id), code=303)
    run = _clustering().request(event_id)
    _coordinator().signal()
    _admin_store().record_audit("cluster_requested", event_id, run_id=run.run_id)
    return redirect(url_for("admin.cluster_review", event_id=event_id, run_id=run.run_id), code=303)


@admin.get("/events/<event_id>/clusters/<run_id>")
@admin_required
def cluster_review(event_id: str, run_id: str):
    try:
        run = _admin_store().get_cluster_run(run_id)
        event = _indexing().get_event(event_id)
    except (KeyError, ValueError):
        abort(404)
    if run.event_id != event_id:
        abort(404)
    members = _admin_store().cluster_members(run_id)
    groups = defaultdict(list)
    for member in members:
        groups[member.cluster_label].append(member)
    return render_template(
        "admin/cluster_review.html",
        event=event,
        run=run,
        groups=dict(groups),
        labels=_admin_store().cluster_labels(run_id),
    )


@admin.post("/events/<event_id>/clusters/<run_id>/labels/<int:cluster_label>")
@admin_required
def label_cluster(event_id: str, run_id: str, cluster_label: int):
    try:
        run = _admin_store().get_cluster_run(run_id)
    except KeyError:
        abort(404)
    if run.event_id != event_id or cluster_label < 0:
        abort(404)
    if not any(
        member.cluster_label == cluster_label
        for member in _admin_store().cluster_members(run_id)
    ):
        abort(404)
    label = request.form.get("label", "").strip()
    _admin_store().set_cluster_label(run_id, cluster_label, label)
    _admin_store().record_audit("cluster_label_updated", event_id, run_id=run_id, group=cluster_label, has_label=bool(label))
    flash("Group label saved.", "success")
    return redirect(url_for("admin.cluster_review", event_id=event_id, run_id=run_id), code=303)


@admin.get("/events/<event_id>/clusters/<run_id>/faces/<int:face_index>")
@admin_required
def cluster_face(event_id: str, run_id: str, face_index: int):
    try:
        run = _admin_store().get_cluster_run(run_id)
    except KeyError:
        abort(404)
    if run.event_id != event_id:
        abort(404)
    member = _admin_store().get_cluster_member(run_id, face_index)
    if member is None:
        abort(404)
    photo = Path(member.photo_path).resolve()
    raw_dir = (_indexing().events_dir / event_id / EVENT_RAW_SUBDIR).resolve()
    try:
        photo.relative_to(raw_dir)
    except ValueError:
        abort(404)
    if not photo.is_file():
        abort(404)
    try:
        with Image.open(photo) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if request.args.get("context") != "1":
                x, y, width, height = member.bbox
                pad = int(max(width, height) * 0.35)
                image = image.crop((max(0, x - pad), max(0, y - pad), min(image.width, x + width + pad), min(image.height, y + height + pad)))
            image.thumbnail((900, 700))
            output = BytesIO()
            image.save(output, format="JPEG", quality=88)
    except OSError:
        abort(404)
    output.seek(0)
    response = send_file(output, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "private, no-store"
    return response


@admin.get("/settings")
@admin_required
def settings():
    return render_template(
        "admin/settings.html",
        settings=_admin_store().get_settings(),
        model=EMBEDDING_MODEL,
        pipeline_version=INDEX_PIPELINE_VERSION,
    )


@admin.post("/settings")
@admin_required
def update_settings():
    try:
        updated = RuntimeSettings(
            possible_threshold=float(request.form["possible_threshold"]),
            confident_threshold=float(request.form["confident_threshold"]),
            top_k=int(request.form["top_k"]),
            cluster_similarity=float(request.form["cluster_similarity"]),
            min_cluster_size=int(request.form["min_cluster_size"]),
        )
        _admin_store().update_settings(updated)
    except (KeyError, ValueError) as exc:
        return render_template(
            "admin/settings.html",
            settings=updated if "updated" in locals() else _admin_store().get_settings(),
            model=EMBEDDING_MODEL,
            pipeline_version=INDEX_PIPELINE_VERSION,
            error=str(exc),
        ), 400
    _admin_store().record_audit("settings_updated")
    flash("Runtime settings saved. New searches and cluster runs will use them.", "success")
    return redirect(url_for("admin.settings"), code=303)


@admin.post("/settings/reset")
@admin_required
def reset_settings():
    _admin_store().reset_settings()
    _admin_store().record_audit("settings_reset")
    flash("Runtime settings restored to project defaults.", "success")
    return redirect(url_for("admin.settings"), code=303)


def _indexing():
    return current_app.extensions["indexing_service"]


def _clustering():
    return current_app.extensions["clustering_service"]


def _coordinator():
    return current_app.extensions["admin_work_coordinator"]


def _admin_store():
    return current_app.extensions["admin_store"]


def _cluster_is_stale(event_id: str) -> bool:
    cluster = _admin_store().latest_cluster(event_id)
    return bool(
        cluster
        and cluster.status == "complete"
        and cluster.index_generation
        != active_index_generation(event_id, _indexing().events_dir)
    )
