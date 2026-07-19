"""
Issue #9 (part 2) — Selfie matching against an event's index.

This is the "Match" stage from the proposal's pipeline diagram: a
student's selfie goes in, a tiered list of their photos from that one
event comes out.
"""

from dataclasses import dataclass

import numpy as np

from src.detection import detect_and_embed
from src.preprocessing import preprocess_image
from src.indexing import load_event_index

from .similarity import classify_tier

# How many nearest neighbors to pull from FAISS before tier-filtering and
# de-duplicating by photo. Generous on purpose — a popular event photo
# can contain many faces, and we want enough candidates left after
# filtering that a real match isn't cut off by a too-small top_k.
DEFAULT_SEARCH_K = 200


@dataclass
class PhotoMatch:
    """One matched photo, with the best score across any face in it."""

    photo_path: str
    score: float
    tier: str  # "confident" | "possible"


class NoFaceDetectedError(ValueError):
    """Raised when no face can be detected in the uploaded selfie."""


class EventNotIndexedError(ValueError):
    """Raised when the requested event has no built index yet."""


def match_selfie(selfie_image: np.ndarray, event_id: str, top_k: int = DEFAULT_SEARCH_K) -> dict:
    """
    Find a student's photos within one event, given a selfie.

    Args:
        selfie_image: BGR image of the student's selfie, as loaded by
            cv2.imread — raw, not yet preprocessed (this function runs
            preprocessing itself, same as the indexing path, so the
            selfie and the indexed photos go through identical steps).
        event_id: which event to search — matches must never cross
            events, per the proposal's Scope.
        top_k: how many FAISS neighbors to consider before tiering.

    Returns:
        {
            "confident": list[PhotoMatch],  # sorted by score, descending
            "possible": list[PhotoMatch],   # sorted by score, descending
        }
        A photo appears in at most one list, using its single best
        matching face if it contains several.

    Raises:
        NoFaceDetectedError: if the selfie has no detectable face —
            the interface layer should ask the student to retake it.
        EventNotIndexedError: if the event hasn't been indexed yet —
            the interface layer should show a "not ready" message.
    """
    cleaned = preprocess_image(selfie_image)
    faces = detect_and_embed(cleaned)

    if not faces:
        raise NoFaceDetectedError(
            "No face detected in the selfie — ask the student to retake it "
            "with better lighting and their face unobstructed."
        )

    # A selfie should have exactly one face; if MTCNN finds more (e.g. a
    # friend photobombing), use the most confident detection as the query.
    query_face = max(faces, key=lambda f: f.confidence)

    try:
        index = load_event_index(event_id)
    except FileNotFoundError as exc:
        raise EventNotIndexedError(
            f"Event '{event_id}' has not been indexed yet."
        ) from exc

    scores, metadata = index.search(query_face.embedding, k=top_k)

    best_per_photo: dict[str, PhotoMatch] = {}
    for score, meta in zip(scores, metadata):
        tier = classify_tier(score)
        if tier is None:
            continue

        existing = best_per_photo.get(meta.photo_path)
        if existing is None or score > existing.score:
            best_per_photo[meta.photo_path] = PhotoMatch(
                photo_path=meta.photo_path, score=score, tier=tier
            )

    confident = sorted(
        (m for m in best_per_photo.values() if m.tier == "confident"),
        key=lambda m: m.score,
        reverse=True,
    )
    possible = sorted(
        (m for m in best_per_photo.values() if m.tier == "possible"),
        key=lambda m: m.score,
        reverse=True,
    )

    return {"confident": confident, "possible": possible}
