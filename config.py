"""
Shared configuration for PhotoMatch.

Every module should import constants from here instead of hardcoding
its own values — this is what keeps preprocessing, detection, indexing,
and matching compatible with each other across the team.
"""

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

# ---------------------------------------------------------------------------
# Auto-clustering (Issue #20)
# ---------------------------------------------------------------------------
# Clustering is intentionally conservative: uncertain faces are left
# unclustered for staff review instead of being forced into a wrong identity.
CLUSTER_NEIGHBORS = 50
CLUSTER_EDGE_SIMILARITY = 0.75
CLUSTER_COHESION_SIMILARITY = 0.70
CLUSTER_MIN_SIZE = 2


def event_dir(event_id: str) -> Path:
    """Return the data directory for a given event_id, creating it if needed."""
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
