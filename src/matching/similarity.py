"""
Issue #9 (part 1) & #10 — Similarity scoring and confidence-tier
classification.

Kept separate from matcher.py so the threshold logic (the part most
likely to get tuned once real FAR/FRR evaluation data exists in Week 4)
is isolated in one small, easily-testable place.
"""

import numpy as np

from config import CONFIDENT_MATCH_THRESHOLD, POSSIBLE_MATCH_THRESHOLD


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Cosine similarity between two face embeddings.

    Note: src.detection.embeddings already L2-normalizes every embedding
    it produces, so in practice this reduces to a plain dot product.
    This function normalizes anyway so it's correct even if it's ever
    called with a vector that wasn't produced by our pipeline (e.g. in
    a unit test with hand-built vectors).

    Args:
        embedding_a, embedding_b: 1-D vectors of the same length.

    Returns:
        Similarity in [-1, 1] — in practice [0, 1] for face embeddings
        from the same model.
    """
    embedding_a = np.asarray(embedding_a, dtype=np.float32)
    embedding_b = np.asarray(embedding_b, dtype=np.float32)
    if embedding_a.ndim != 1 or embedding_b.ndim != 1:
        raise ValueError("cosine_similarity: embeddings must be one-dimensional")
    if embedding_a.shape != embedding_b.shape:
        raise ValueError(
            "cosine_similarity: embeddings must have the same dimensions, "
            f"got {embedding_a.shape} and {embedding_b.shape}"
        )
    if not np.isfinite(embedding_a).all() or not np.isfinite(embedding_b).all():
        raise ValueError("cosine_similarity: embeddings must contain only finite values")

    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))


def classify_tier(score: float) -> str | None:
    """
    Classify a similarity score into one of the two result tiers from
    the proposal: "confident" and "possible". Everything below the
    possible-match threshold is not shown to the student at all — the
    proposal's "reduce false negatives" objective is served by the
    *possible* tier existing, not by lowering this floor to zero.

    Args:
        score: cosine similarity, typically in [0, 1].

    Returns:
        "confident", "possible", or None if the score is too low to
        surface as a match at all.

    Note:
        CONFIDENT_MATCH_THRESHOLD and POSSIBLE_MATCH_THRESHOLD live in
        config.py as starting points — Week 4 evaluation (FAR/FRR) is
        what should actually tune these, not a guess made here.
    """
    if score >= CONFIDENT_MATCH_THRESHOLD:
        return "confident"
    if score >= POSSIBLE_MATCH_THRESHOLD:
        return "possible"
    return None
