"""Safe event creation and atomic local photo imports."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from config import EVENT_RAW_SUBDIR

from .indexing_service import IMAGE_EXTENSIONS, IndexingService

MAX_PHOTO_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class ImportOutcome:
    filename: str
    imported: bool
    error: str | None = None


def event_slug(title: str) -> str:
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")
    return slug[:96] or "event"


def allocate_event_id(events_dir: Path, title: str) -> str:
    base = event_slug(title)
    candidate = base
    counter = 2
    while (events_dir / candidate).exists():
        suffix = f"-{counter}"
        candidate = f"{base[:128 - len(suffix)]}{suffix}"
        counter += 1
    return candidate


def create_event(
    indexing: IndexingService, title: str, event_date: str
):
    title = title.strip()
    if not title:
        raise ValueError("Event title is required.")
    if len(title) > 160:
        raise ValueError("Event title must be 160 characters or fewer.")
    event_id = allocate_event_id(indexing.events_dir, title)
    raw_dir = indexing.events_dir / event_id / EVENT_RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=False)
    try:
        return indexing.register_event(event_id, event_date, display_name=title)
    except Exception:
        raw_dir.rmdir()
        raw_dir.parent.rmdir()
        raise


def import_photos(raw_dir: Path, files: list[FileStorage]) -> list[ImportOutcome]:
    raw_dir = raw_dir.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[ImportOutcome] = []
    for upload in files:
        original = upload.filename or "unnamed"
        filename = secure_filename(original)
        if not filename or Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
            outcomes.append(ImportOutcome(original, False, "Use a JPG, JPEG, or PNG file."))
            continue
        destination = (raw_dir / filename).resolve()
        if destination.parent != raw_dir:
            outcomes.append(ImportOutcome(original, False, "Invalid filename."))
            continue
        if destination.exists():
            outcomes.append(ImportOutcome(filename, False, "A photo with this name already exists."))
            continue

        temporary = raw_dir / f".{uuid4().hex}.uploading"
        total = 0
        try:
            with temporary.open("xb") as output:
                while chunk := upload.stream.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_PHOTO_BYTES:
                        raise ValueError("Photo exceeds the 50 MB limit.")
                    output.write(chunk)
            try:
                with Image.open(temporary) as image:
                    image.verify()
                    if image.format not in {"JPEG", "PNG"}:
                        raise ValueError("File contents are not JPG or PNG.")
                    if (
                        image.format == "JPEG" and destination.suffix.lower() == ".png"
                    ) or (
                        image.format == "PNG"
                        and destination.suffix.lower() in {".jpg", ".jpeg"}
                    ):
                        raise ValueError("The filename extension does not match the image contents.")
            except UnidentifiedImageError as exc:
                raise ValueError("File contents are not a valid image.") from exc
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise ValueError("A photo with this name already exists.") from exc
            try:
                temporary.unlink()
            except OSError:
                # The destination is already an atomically published hard link.
                # A hidden cleanup link is preferable to reporting a false import
                # failure or risking the original photo.
                pass
            outcomes.append(ImportOutcome(filename, True))
        except (OSError, ValueError) as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            outcomes.append(ImportOutcome(filename, False, str(exc)))
    return outcomes
