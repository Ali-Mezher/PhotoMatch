"""
Builds (or rebuilds) the FAISS index for one event by running every raw
photo through preprocessing + detection, then saving the result.

This is the "Index" stage from the proposal's pipeline diagram, and the
manual trigger mentioned in Scope: indexing only runs when explicitly
called for an event, so unfinished/soon-to-be-replaced photos never end
up searchable by accident.
"""

from collections.abc import Callable
from pathlib import Path

import cv2
from tqdm import tqdm

from config import EVENT_RAW_SUBDIR, EVENT_INDEXED_SUBDIR, event_dir
from src.preprocessing import preprocess_image
from src.detection import detect_and_embed

from .faiss_index import EventIndex, IndexedFace

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ProgressCallback = Callable[[Path, str, int, str | None], None]


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
        p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not photo_paths:
        raise FileNotFoundError(
            f"build_event_index: no images found in {raw_dir}"
        )

    index = EventIndex()

    iterator = tqdm(photo_paths, desc=f"Indexing {event_id}") if show_progress else photo_paths
    skipped = []

    for photo_path in iterator:
        image = cv2.imread(str(photo_path))
        if image is None:
            skipped.append((photo_path, "could not read image"))
            if progress_callback:
                progress_callback(photo_path, "failed", 0, "could not read image")
            continue

        try:
            cleaned = preprocess_image(image)
            faces = detect_and_embed(cleaned)
        except Exception as exc:  # noqa: BLE001 — one bad photo shouldn't kill the run
            skipped.append((photo_path, str(exc)))
            if progress_callback:
                progress_callback(photo_path, "failed", 0, str(exc))
            continue

        if not faces:
            if progress_callback:
                progress_callback(photo_path, "no_face", 0, None)
            continue  # no faces in this photo (e.g. decor/venue shot) — not an error

        embeddings = [face.embedding for face in faces]
        metadata = [
            IndexedFace(
                photo_path=str(photo_path),
                bbox=face.bbox,
                confidence=face.confidence,
            )
            for face in faces
        ]
        index.add(embeddings, metadata)
        if progress_callback:
            progress_callback(photo_path, "indexed", len(faces), None)

    output_dir = event_dir(event_id) / EVENT_INDEXED_SUBDIR
    index.save(output_dir)

    if skipped:
        print(f"\nSkipped {len(skipped)} photo(s) while indexing '{event_id}':")
        for path, reason in skipped:
            print(f"  {path.name}: {reason}")

    print(f"Indexed {len(index)} face(s) from {len(photo_paths) - len(skipped)} photo(s) -> {output_dir}")
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
    return EventIndex.load(index_dir)
