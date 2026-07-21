"""Shared result-token and event-photo containment checks."""

from __future__ import annotations

from pathlib import Path

from flask import abort, current_app

from config import EVENT_RAW_SUBDIR

from .result_store import SearchResultStore, StoredPhoto, StoredSearch


def filter_matches(event_id: str, matches: dict) -> dict:
    raw_dir = event_raw_dir(event_id)
    filtered = {"confident": [], "possible": []}
    for tier in filtered:
        for match in matches.get(tier, []):
            path = Path(match.photo_path).resolve()
            if is_allowed_path(path, raw_dir) and path.is_file():
                filtered[tier].append(match)
            else:
                current_app.logger.warning(
                    "Ignored match outside event directory for event %s", event_id
                )
    return filtered


def require_search(token: str, audience: str) -> StoredSearch:
    stored = result_store().get(token)
    if stored is None:
        abort(410, description="This search has expired. Start a new search.")
    if stored.audience != audience:
        abort(404)
    return stored


def require_photo(
    token: str, photo_id: str, audience: str
) -> tuple[StoredSearch, StoredPhoto]:
    stored = require_search(token, audience)
    photo = stored.photos.get(photo_id)
    if photo is None:
        abort(404)
    if not is_allowed_path(photo.path, event_raw_dir(stored.event_id)):
        abort(404)
    if not photo.path.is_file():
        abort(404)
    return stored, photo


def event_raw_dir(event_id: str) -> Path:
    events_dir = Path(current_app.config["EVENTS_DIR"]).resolve()
    raw_dir = (events_dir / event_id / EVENT_RAW_SUBDIR).resolve()
    if raw_dir.parent.parent != events_dir:
        abort(404)
    return raw_dir


def is_allowed_path(path: Path, raw_dir: Path) -> bool:
    try:
        path.relative_to(raw_dir)
    except ValueError:
        return False
    return True


def result_store() -> SearchResultStore:
    return current_app.config["RESULT_STORE"]
