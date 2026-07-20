"""
Issue #13 — end-to-end retrieval benchmark: runs every identity's real
selfie through the full pipeline (preprocess -> detect -> embed ->
search) against one event's index, and measures precision/recall@k and
query time.

As a side effect, this also collects genuine/impostor similarity scores
from real photos (not hand-typed numbers) — feed those into
threshold_tuning.tune_confidence_tiers() to complete issue #14 with
real data instead of the toy scores.csv example.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from config import DEFAULT_EVAL_TOP_K
from src.detection import detect_and_embed
from src.indexing import load_event_index
from src.matching.matcher import DEFAULT_SEARCH_K
from src.preprocessing import preprocess_image

from .ground_truth import EvaluationSet, Identity, load_evaluation_set
from .retrieval_metrics import RetrievalMetrics, precision_recall_at_k


@dataclass
class IdentityResult:
    """Benchmark result for one person's selfie search."""

    name: str
    metrics: RetrievalMetrics
    query_time_seconds: float
    top_score: float | None  # None if the index returned zero results


@dataclass
class BenchmarkResult:
    """Aggregate benchmark result for one event."""

    event_id: str
    k: int
    identity_results: list[IdentityResult]
    # Every score the search returned, labeled by whether that photo
    # actually belongs to the searching identity — this is the real
    # genuine/impostor dataset for issue #14's threshold tuning.
    genuine_scores: list[float] = field(default_factory=list)
    impostor_scores: list[float] = field(default_factory=list)

    @property
    def mean_precision(self) -> float:
        return _mean(r.metrics.precision_at_k for r in self.identity_results)

    @property
    def mean_recall(self) -> float:
        return _mean(r.metrics.recall_at_k for r in self.identity_results)

    @property
    def mean_query_time_seconds(self) -> float:
        return _mean(r.query_time_seconds for r in self.identity_results)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _warm_up_models() -> None:
    """
    Trigger MTCNN/deepface's lazy model loading before timing starts, so
    the first identity's query_time_seconds isn't inflated by one-time
    model-load latency that a real deployed app would only pay once at
    startup, not per search.
    """
    import numpy as np

    dummy = np.zeros((160, 160, 3), dtype="uint8")
    try:
        detect_and_embed(dummy)
    except Exception:
        pass  # a blank image finding no faces is fine — we only care about warm-up


def _search_one_identity(
    identity: Identity, index, k: int
) -> tuple[IdentityResult, list[float], list[float]]:
    selfie = cv2.imread(str(identity.selfie_path))
    if selfie is None:
        raise ValueError(f"Could not read selfie for '{identity.name}': {identity.selfie_path}")

    start = time.perf_counter()

    cleaned = preprocess_image(selfie)
    faces = detect_and_embed(cleaned)
    if not faces:
        raise ValueError(f"No face detected in selfie for '{identity.name}'.")
    query_face = max(faces, key=lambda f: f.confidence)

    search_k = max(k, DEFAULT_SEARCH_K)
    scores, metadata = index.search(query_face.embedding, k=search_k)

    elapsed = time.perf_counter() - start

    ranked_filenames = []
    genuine_scores = []
    impostor_scores = []
    seen = set()

    for score, meta in zip(scores, metadata):
        filename = Path(meta.photo_path).name
        if filename in seen:
            continue
        seen.add(filename)
        ranked_filenames.append(filename)

        if filename in identity.ground_truth_filenames:
            genuine_scores.append(score)
        else:
            impostor_scores.append(score)

    metrics = precision_recall_at_k(ranked_filenames, identity.ground_truth_filenames, k)
    top_score = scores[0] if scores else None

    result = IdentityResult(
        name=identity.name,
        metrics=metrics,
        query_time_seconds=elapsed,
        top_score=top_score,
    )
    return result, genuine_scores, impostor_scores


def run_benchmark(event_id: str, k: int = DEFAULT_EVAL_TOP_K) -> BenchmarkResult:
    """
    Run the full precision/recall/timing benchmark for one event.

    Args:
        event_id: must already be indexed (src.indexing.build_event_index)
            and have an evaluation set set up under
            data/evaluation/<event_id>/identities/ (see ground_truth.py).
        k: how many top results to evaluate precision/recall at.

    Returns:
        BenchmarkResult with per-identity metrics/timing and the pooled
        genuine/impostor scores for threshold tuning.

    Raises:
        FileNotFoundError: if the event isn't indexed or has no
            evaluation identities set up yet.
        ValueError: if a selfie can't be read or has no detectable face
            — this is treated as a setup problem worth fixing, not a
            result to silently skip, since it usually means a bad photo
            was dropped into the evaluation folder.
    """
    eval_set: EvaluationSet = load_evaluation_set(event_id)

    try:
        index = load_event_index(event_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Event '{event_id}' has no built index yet. Run "
            f"src.indexing.build_event_index('{event_id}') first."
        ) from exc

    _warm_up_models()

    identity_results = []
    all_genuine = []
    all_impostor = []

    for identity in eval_set.identities:
        result, genuine, impostor = _search_one_identity(identity, index, k)
        identity_results.append(result)
        all_genuine.extend(genuine)
        all_impostor.extend(impostor)

    return BenchmarkResult(
        event_id=event_id,
        k=k,
        identity_results=identity_results,
        genuine_scores=all_genuine,
        impostor_scores=all_impostor,
    )
