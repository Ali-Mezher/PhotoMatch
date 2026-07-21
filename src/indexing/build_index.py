"""
Builds (or rebuilds) the FAISS index for one event by running every raw
photo through preprocessing + detection, then saving the result.

This is the "Index" stage from the proposal's pipeline diagram, and the
manual trigger mentioned in Scope: indexing only runs when explicitly
called for an event, so unfinished/soon-to-be-replaced photos never end
up searchable by accident.
"""

import json
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
from tqdm import tqdm

from config import (
    EVENTS_DIR,
    EVENT_RAW_SUBDIR,
    EVENT_INDEXED_SUBDIR,
    event_dir,
    validate_event_id,
)
from src.preprocessing import preprocess_image
from src.detection import detect_and_embed

from .faiss_index import EventIndex, IndexedFace

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ACTIVE_INDEX_FILENAME = "active.json"
GENERATIONS_DIRNAME = "generations"
_GENERATION_PATTERN = re.compile(r"^[0-9a-f]{32}$")
ProgressCallback = Callable[[Path, str, int, str | None], None]


@dataclass(frozen=True)
class IndexBuildOutcome:
    """Result of processing one source image."""

    photo_path: Path
    status: str
    face_count: int = 0
    error: str | None = None


def _publish_index(index: EventIndex, output_dir: Path) -> None:
    """Publish an immutable index generation with an atomic manifest swap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_generation = None
    current_manifest = output_dir / ACTIVE_INDEX_FILENAME
    if current_manifest.exists():
        try:
            candidate = json.loads(current_manifest.read_text(encoding="utf-8")).get(
                "generation"
            )
            if isinstance(candidate, str) and _GENERATION_PATTERN.fullmatch(candidate):
                previous_generation = candidate
        except (OSError, AttributeError, json.JSONDecodeError):
            pass

    generation = uuid4().hex
    generation_dir = output_dir / GENERATIONS_DIRNAME / generation
    index.save(generation_dir)

    manifest = output_dir / ACTIVE_INDEX_FILENAME
    temporary_manifest = output_dir / f".{ACTIVE_INDEX_FILENAME}.{generation}.tmp"
    temporary_manifest.write_text(
        json.dumps({"generation": generation}), encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest)

    # Keep the active and immediately previous generations. A search that
    # resolved the old manifest just before publication can still finish,
    # while older full-size FAISS snapshots do not accumulate forever.
    keep = {generation, previous_generation}
    generations_dir = output_dir / GENERATIONS_DIRNAME
    for candidate in generations_dir.iterdir():
        if candidate.is_dir() and candidate.name not in keep:
            shutil.rmtree(candidate, ignore_errors=True)


def _active_index_dir(output_dir: Path) -> Path:
    """Resolve the active immutable generation, falling back to v1 layout."""
    manifest = output_dir / ACTIVE_INDEX_FILENAME
    if manifest.exists():
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            generation = raw["generation"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid active index manifest at {manifest}") from exc
        if not isinstance(generation, str) or not _GENERATION_PATTERN.fullmatch(
            generation
        ):
            raise ValueError(f"Invalid active index generation in {manifest}")
        return output_dir / GENERATIONS_DIRNAME / generation
    return output_dir


def event_index_exists(event_id: str) -> bool:
    """Return whether an event has a complete active or legacy index."""
    output_dir = event_dir(event_id) / EVENT_INDEXED_SUBDIR
    try:
        active_dir = _active_index_dir(output_dir)
    except ValueError:
        return False
    return (active_dir / "faces.faiss").exists() and (
        active_dir / "metadata.json"
    ).exists()


def active_index_generation(
    event_id: str, events_dir: Path = EVENTS_DIR
) -> str | None:
    """Return the published generation ID, or ``legacy`` for the v1 layout."""
    output_dir = Path(events_dir).resolve() / validate_event_id(event_id) / EVENT_INDEXED_SUBDIR
    try:
        active_dir = _active_index_dir(output_dir)
    except ValueError:
        return None
    if not (active_dir / "faces.faiss").is_file() or not (
        active_dir / "metadata.json"
    ).is_file():
        return None
    return active_dir.name if active_dir != output_dir else "legacy"


def _process_photos(
    index: EventIndex,
    photo_paths: list[Path],
    show_progress: bool,
    progress_callback: ProgressCallback | None = None,
) -> list[IndexBuildOutcome]:
    iterator = (
        tqdm(photo_paths, desc="Indexing photos") if show_progress else photo_paths
    )
    outcomes: list[IndexBuildOutcome] = []

    for photo_path in iterator:
        image = cv2.imread(str(photo_path))
        if image is None:
            error = "could not read image"
            outcomes.append(IndexBuildOutcome(photo_path, "failed", error=error))
            if progress_callback:
                progress_callback(photo_path, "failed", 0, error)
            continue

        try:
            cleaned = preprocess_image(image)
            faces = detect_and_embed(cleaned)
        except Exception as exc:  # noqa: BLE001 - isolate one corrupt source image
            error = str(exc)
            outcomes.append(IndexBuildOutcome(photo_path, "failed", error=error))
            if progress_callback:
                progress_callback(photo_path, "failed", 0, error)
            continue

        if not faces:
            outcomes.append(IndexBuildOutcome(photo_path, "no_face"))
            if progress_callback:
                progress_callback(photo_path, "no_face", 0, None)
            continue

        index.add(
            [face.embedding for face in faces],
            [
                IndexedFace(
                    photo_path=str(photo_path),
                    bbox=face.bbox,
                    confidence=face.confidence,
                )
                for face in faces
            ],
        )
        outcomes.append(IndexBuildOutcome(photo_path, "indexed", len(faces)))
        if progress_callback:
            progress_callback(photo_path, "indexed", len(faces), None)

    return outcomes


def update_event_index(
    event_id: str,
    photo_paths: list[Path],
    *,
    rebuild: bool = False,
    show_progress: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> tuple[EventIndex, list[IndexBuildOutcome]]:
    """Append selected photos or atomically rebuild an event's whole index."""
    output_dir = event_dir(event_id) / EVENT_INDEXED_SUBDIR
    if rebuild or not event_index_exists(event_id):
        index = EventIndex()
    else:
        index = EventIndex.load(_active_index_dir(output_dir))

    outcomes = _process_photos(
        index,
        [Path(path) for path in photo_paths],
        show_progress,
        progress_callback,
    )
    _publish_index(index, output_dir)
    return index, outcomes


def build_event_index(
    event_id: str,
    show_progress: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> EventIndex:
    """
    Index every photo in data/events/<event_id>/raw/ and save the result
    to data/events/<event_id>/indexed/.

    Args:
        event_id: the event's identifier — must match a folder under
            data/events/ containing a raw/ subfolder of photos.
        show_progress: whether to show a tqdm progress bar (useful in a
            terminal, noisy in automated runs — turn off for those).

    Returns:
        The built EventIndex, already saved to disk.

    Raises:
        FileNotFoundError: if data/events/<event_id>/raw/ doesn't exist
            or contains no images.

    Note:
        A photo that fails to load or throws during preprocessing is
        skipped with a printed warning rather than aborting the whole
        event — one corrupt file shouldn't block indexing thousands of
        others. Check the printed warnings after a run.
    """
    raw_dir = event_dir(event_id) / EVENT_RAW_SUBDIR
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"build_event_index: no raw/ folder for event '{event_id}' at {raw_dir}"
        )

    photo_paths = sorted(
        p
        for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not photo_paths:
        raise FileNotFoundError(
            f"build_event_index: no images found in {raw_dir}"
        )

    index, outcomes = update_event_index(
        event_id,
        photo_paths,
        rebuild=True,
        show_progress=show_progress,
        progress_callback=progress_callback,
    )
    failed = [outcome for outcome in outcomes if outcome.status == "failed"]
    output_dir = event_dir(event_id) / EVENT_INDEXED_SUBDIR

    if failed:
        print(f"\nSkipped {len(failed)} photo(s) while indexing '{event_id}':")
        for outcome in failed:
            print(f"  {outcome.photo_path.name}: {outcome.error}")

    print(
        f"Indexed {len(index)} face(s) from {len(photo_paths) - len(failed)} "
        f"photo(s) -> {output_dir}"
    )
    return index


def load_event_index(event_id: str) -> EventIndex:
    """
    Load a previously built index for an event.

    Raises:
        FileNotFoundError: if the event hasn't been indexed yet — callers
            (src.matching, src.interface) should catch this and show the
            student/staff a clear "this event isn't ready yet" message
            rather than a raw stack trace.
    """
    index_dir = event_dir(event_id) / EVENT_INDEXED_SUBDIR
    return EventIndex.load(_active_index_dir(index_dir))


def load_event_index_snapshot(
    event_id: str, events_dir: Path = EVENTS_DIR
) -> tuple[str, EventIndex]:
    """Load an index together with the immutable generation it represents."""
    index_dir = Path(events_dir).resolve() / validate_event_id(event_id) / EVENT_INDEXED_SUBDIR
    active_dir = _active_index_dir(index_dir)
    generation = active_dir.name if active_dir != index_dir else "legacy"
    return generation, EventIndex.load(active_dir)
