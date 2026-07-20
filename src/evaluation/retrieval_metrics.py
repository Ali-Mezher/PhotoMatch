"""
Issue #13 — precision/recall@k for one ranked retrieval result.

Kept separate from benchmark.py (which does the actual searching) so the
metric math is small, pure, and independently testable — no FAISS, no
detection models, just two lists compared against each other.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetrics:
    """Precision/recall for one identity's search, at one value of k."""

    k: int
    precision_at_k: float
    recall_at_k: float
    num_relevant: int             # total ground-truth photos for this person
    num_retrieved_relevant: int   # how many of those showed up in the top k


def precision_recall_at_k(
    ranked_filenames: list[str], relevant_filenames: set[str], k: int
) -> RetrievalMetrics:
    """
    Compute precision@k and recall@k for one search result.

    Args:
        ranked_filenames: photo filenames (not full paths) in the order
            they were returned by the search, most similar first —
            duplicates should already be removed by the caller.
        relevant_filenames: ground-truth filenames that actually belong
            to the person being searched for (from
            Identity.ground_truth_filenames).
        k: how many top results to evaluate. Must be positive.

    Returns:
        RetrievalMetrics with:
            precision_at_k = (relevant photos in top k) / k
            recall_at_k    = (relevant photos in top k) / (total relevant)

        If relevant_filenames is empty, recall_at_k is 0.0 by convention
        (this shouldn't happen in practice — ground_truth.py requires at
        least one match per identity).

    Raises:
        ValueError: if k is not positive.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    top_k = ranked_filenames[:k]
    retrieved_relevant = sum(1 for f in top_k if f in relevant_filenames)

    precision = retrieved_relevant / len(top_k) if top_k else 0.0
    recall = retrieved_relevant / len(relevant_filenames) if relevant_filenames else 0.0

    return RetrievalMetrics(
        k=k,
        precision_at_k=precision,
        recall_at_k=recall,
        num_relevant=len(relevant_filenames),
        num_retrieved_relevant=retrieved_relevant,
    )
