"""
Shared configuration for PhotoMatch.

Every module should import constants from here instead of hardcoding
its own values — this is what keeps preprocessing, detection, indexing,
and matching compatible with each other across the team.
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
EVENTS_DIR = DATA_DIR / "events"          # data/events/<event_id>/*.jpg
MODELS_DIR = PROJECT_ROOT / "models"      # pretrained weights live here (gitignored)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
# Target size (width, height) faces/photos are normalized to before detection.
# Keep this the same everywhere so a preprocessed image from Ali's pipeline
# is always the right shape for Mahmood's detector.
TARGET_IMAGE_SIZE = (1024, 1024)  # max dimension, aspect ratio preserved

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
FACE_DETECTOR_BACKEND = "mtcnn"          # passed to deepface / used directly
MIN_DETECTION_CONFIDENCE = 0.90          # drop low-confidence detections
MIN_FACE_SIZE_PX = 40                    # ignore faces smaller than this (noise)

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "Facenet512"           # deepface model name
EMBEDDING_DIM = 512

# ---------------------------------------------------------------------------
# Matching / thresholds
# ---------------------------------------------------------------------------
# Cosine similarity thresholds for the two result tiers described in the
# proposal. Tune these once real evaluation data (FAR/FRR) is available —
# these are reasonable starting points, not final numbers.
CONFIDENT_MATCH_THRESHOLD = 0.65
POSSIBLE_MATCH_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Event ID convention
# ---------------------------------------------------------------------------
# data/events/<event_id>/raw/       original photos as delivered
# data/events/<event_id>/indexed/   photos that have been through the
#                                    detect -> embed -> index pipeline
EVENT_RAW_SUBDIR = "raw"
EVENT_INDEXED_SUBDIR = "indexed"

# Increment this whenever preprocessing, detection, or embedding behavior
# changes in a way that makes existing vectors stale. The indexing service
# will rebuild affected events before serving the new generation.
INDEX_PIPELINE_VERSION = "1"

INDEX_STATUS_DB = DATA_DIR / "indexing_status.sqlite3"

_EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_event_id(event_id: str) -> str:
    """Return a safe event ID or raise before it reaches filesystem paths."""
    if not isinstance(event_id, str) or not _EVENT_ID_PATTERN.fullmatch(event_id):
        raise ValueError(
            "event_id must be 1-128 characters using only letters, numbers, "
            "dots, underscores, and hyphens"
        )
    if event_id in {".", ".."}:
        raise ValueError("event_id cannot be '.' or '..'")
    return event_id


def event_dir(event_id: str) -> Path:
    """Return the data directory for a given event_id, creating it if needed."""
    event_id = validate_event_id(event_id)
    d = EVENTS_DIR / event_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Evaluation (Week 4)
# ---------------------------------------------------------------------------
# data/evaluation/<event_id>/identities/<person_name>/selfie/    one selfie
# data/evaluation/<event_id>/identities/<person_name>/matches/   copies of
#     that person's photos from data/events/<event_id>/raw/ — filenames
#     MUST match exactly, that's how ground truth links back to the index.
#
# The raw event photos themselves are NOT duplicated here — evaluation
# reuses data/events/<event_id>/raw/ directly (see EVENT_RAW_SUBDIR above),
# so an event only ever needs to be indexed once.
EVALUATION_DIR = DATA_DIR / "evaluation"
EVAL_IDENTITIES_SUBDIR = "identities"
EVAL_SELFIE_SUBDIR = "selfie"
EVAL_MATCHES_SUBDIR = "matches"

DEFAULT_EVAL_TOP_K = 10  # default k for precision/recall@k

# Manual search baseline from the proposal (5-6 hours per student) —
# used as the comparison point for the time-savings evaluation.
MANUAL_SEARCH_BASELINE_SECONDS = 5.5 * 3600


def evaluation_dir(event_id: str) -> Path:
    """Return the evaluation data directory for a given event_id."""
    d = EVALUATION_DIR / event_id
    d.mkdir(parents=True, exist_ok=True)
    return d
