"""Reproducible matching and identity-clustering threshold experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.clustering import cluster_index
from src.indexing import EventIndex, IndexedFace

from .threshold_tuning import evaluate_threshold


@dataclass(frozen=True)
class ClassificationMetrics:
    threshold: float
    far: float
    frr: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class MatchingGridResult:
    possible_threshold: float
    confident_threshold: float
    possible: ClassificationMetrics
    confident: ClassificationMetrics


@dataclass(frozen=True)
class ClusteringGridResult:
    edge_similarity: float
    cohesion_similarity: float
    false_merges: int
    false_splits: int
    cluster_purity: float
    unassigned_faces: int
    cluster_count: int


def inclusive_grid(start: float, end: float, step: float) -> list[float]:
    if not all(np.isfinite([start, end, step])) or step <= 0 or start > end:
        raise ValueError("grid requires finite start <= end and a positive step")
    count = int(np.floor((end - start) / step + 1e-9))
    values = [round(start + index * step, 10) for index in range(count + 1)]
    if values[-1] < end - 1e-9:
        values.append(float(end))
    if values[0] < 0 or values[-1] > 1:
        raise ValueError("similarity thresholds must be between 0 and 1")
    return values


def classification_metrics(
    scores: Sequence[float], labels: Sequence[bool], threshold: float
) -> ClassificationMetrics:
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=bool)
    base = evaluate_threshold(score_array, label_array, threshold)
    accepted = score_array >= threshold
    true_positives = int(np.count_nonzero(accepted & label_array))
    true_negatives = int(np.count_nonzero(~accepted & ~label_array))
    false_positives = base.false_accepts
    false_negatives = base.false_rejects
    precision_denominator = true_positives + false_positives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / base.genuine_count
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return ClassificationMetrics(
        float(threshold),
        base.far,
        base.frr,
        precision,
        recall,
        f1,
        true_positives,
        true_negatives,
        false_positives,
        false_negatives,
    )


def evaluate_matching_grid(
    scores: Sequence[float],
    labels: Sequence[bool],
    possible_thresholds: Iterable[float],
    confident_thresholds: Iterable[float],
) -> list[MatchingGridResult]:
    possible_values = list(possible_thresholds)
    confident_values = list(confident_thresholds)
    results = [
        MatchingGridResult(
            possible,
            confident,
            classification_metrics(scores, labels, possible),
            classification_metrics(scores, labels, confident),
        )
        for possible in possible_values
        for confident in confident_values
        if 0 <= possible < confident <= 1
    ]
    if not results:
        raise ValueError("matching grid needs at least one possible < confident pair")
    return sorted(
        results,
        key=lambda result: (
            result.confident.false_positives,
            result.possible.false_positives,
            -result.possible.recall,
            -result.confident.recall,
            result.confident_threshold,
        ),
    )


def evaluate_clustering_grid(
    embeddings: np.ndarray,
    identities: Sequence[str],
    edge_thresholds: Iterable[float],
    cohesion_thresholds: Iterable[float],
    *,
    neighbors: int = 50,
    min_cluster_size: int = 2,
) -> list[ClusteringGridResult]:
    vectors = np.asarray(embeddings, dtype=np.float32)
    labels = [str(identity) for identity in identities]
    if vectors.ndim != 2 or not len(vectors) or len(vectors) != len(labels):
        raise ValueError("embeddings must be a non-empty 2-D array matching identities")
    if not np.isfinite(vectors).all() or any(not label for label in labels):
        raise ValueError("embeddings must be finite and identities must not be blank")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding vectors must be non-zero")
    vectors = vectors / norms

    index = EventIndex(dim=vectors.shape[1])
    index.add(
        list(vectors),
        [IndexedFace(f"face_{i}.jpg", (0, 0, 1, 1), 1.0) for i in range(len(vectors))],
    )
    results: list[ClusteringGridResult] = []
    for edge in edge_thresholds:
        for cohesion in cohesion_thresholds:
            if not 0 <= edge <= 1 or not 0 <= cohesion <= 1:
                raise ValueError("clustering thresholds must be between 0 and 1")
            clustered = cluster_index(
                index,
                neighbors=neighbors,
                edge_similarity=float(edge),
                cohesion_similarity=float(cohesion),
                min_cluster_size=min_cluster_size,
            )
            assignments = [-1] * len(vectors)
            for cluster_number, cluster in enumerate(clustered.clusters):
                for member in cluster.members:
                    assignments[member.face_index] = cluster_number
            false_merges = 0
            false_splits = 0
            for first, second in combinations(range(len(labels)), 2):
                same_cluster = assignments[first] >= 0 and assignments[first] == assignments[second]
                same_identity = labels[first] == labels[second]
                false_merges += int(same_cluster and not same_identity)
                false_splits += int(same_identity and not same_cluster)
            assigned_count = sum(assignment >= 0 for assignment in assignments)
            purity_numerator = 0
            for cluster_number in set(assignments) - {-1}:
                counts: dict[str, int] = {}
                for identity, assignment in zip(labels, assignments):
                    if assignment == cluster_number:
                        counts[identity] = counts.get(identity, 0) + 1
                purity_numerator += max(counts.values())
            purity = purity_numerator / assigned_count if assigned_count else 0.0
            results.append(
                ClusteringGridResult(
                    float(edge),
                    float(cohesion),
                    false_merges,
                    false_splits,
                    purity,
                    assignments.count(-1),
                    len(clustered.clusters),
                )
            )
    if not results:
        raise ValueError("clustering grid must not be empty")
    return sorted(
        results,
        key=lambda result: (
            result.false_merges,
            result.false_splits,
            -result.cluster_purity,
            result.unassigned_faces,
            -result.edge_similarity,
            -result.cohesion_similarity,
        ),
    )


def _threshold_result_fields() -> list[str]:
    return [
        "mode", "rank", "possible_threshold", "confident_threshold",
        "possible_far", "possible_frr", "possible_precision", "possible_recall",
        "possible_f1", "possible_tp", "possible_tn", "possible_fp", "possible_fn",
        "confident_far", "confident_frr", "confident_precision", "confident_recall",
        "confident_f1", "confident_tp", "confident_tn", "confident_fp", "confident_fn",
        "edge_similarity", "cohesion_similarity", "false_merges", "false_splits",
        "cluster_purity", "unassigned_faces", "cluster_count",
    ]


def save_threshold_artifacts(
    output_dir: Path,
    matching: Sequence[MatchingGridResult],
    clustering: Sequence[ClusteringGridResult],
) -> tuple[Path, Path]:
    """Write ranked grid data and a conservative human-readable recommendation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "threshold-results.csv"
    report_path = output_dir / "threshold-report.md"

    fields = _threshold_result_fields()
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, result in enumerate(matching, 1):
            row: dict[str, object] = {
                "mode": "matching",
                "rank": rank,
                "possible_threshold": result.possible_threshold,
                "confident_threshold": result.confident_threshold,
            }
            for prefix, metric in (
                ("possible", result.possible),
                ("confident", result.confident),
            ):
                row.update(
                    {
                        f"{prefix}_far": metric.far,
                        f"{prefix}_frr": metric.frr,
                        f"{prefix}_precision": metric.precision,
                        f"{prefix}_recall": metric.recall,
                        f"{prefix}_f1": metric.f1,
                        f"{prefix}_tp": metric.true_positives,
                        f"{prefix}_tn": metric.true_negatives,
                        f"{prefix}_fp": metric.false_positives,
                        f"{prefix}_fn": metric.false_negatives,
                    }
                )
            writer.writerow(row)
        for rank, result in enumerate(clustering, 1):
            writer.writerow(
                {
                    "mode": "clustering",
                    "rank": rank,
                    "edge_similarity": result.edge_similarity,
                    "cohesion_similarity": result.cohesion_similarity,
                    "false_merges": result.false_merges,
                    "false_splits": result.false_splits,
                    "cluster_purity": result.cluster_purity,
                    "unassigned_faces": result.unassigned_faces,
                    "cluster_count": result.cluster_count,
                }
            )

    lines = [
        "# PhotoMatch threshold recommendation",
        "",
        "These are calibration candidates, not final production values. Confirm them on an untouched event.",
        "",
        "## Recommended candidates",
        "",
    ]
    if matching:
        candidate = matching[0]
        lines.extend(
            [
                f"- Search possible-match threshold: **{candidate.possible_threshold:.3f}**",
                f"- Search confident-match threshold: **{candidate.confident_threshold:.3f}**",
                f"- Observed confident false accepts: **{candidate.confident.false_positives}**",
                f"- Observed possible false accepts: **{candidate.possible.false_positives}**",
            ]
        )
        if candidate.confident.false_positives or candidate.possible.false_positives:
            lines.append(
                "- **Warning:** No tested matching pair eliminated every observed false accept. Expand or raise the grid before deployment."
            )
    else:
        lines.append("- Search recommendation unavailable: no genuine/impostor score grid was produced.")

    if clustering:
        candidate = clustering[0]
        lines.extend(
            [
                f"- Cluster edge similarity: **{candidate.edge_similarity:.3f}**",
                f"- Cluster cohesion similarity: **{candidate.cohesion_similarity:.3f}**",
                f"- Observed cross-person false merges: **{candidate.false_merges}**",
                f"- Observed same-person false splits: **{candidate.false_splits}**",
            ]
        )
        if candidate.false_merges:
            lines.append(
                "- **DO NOT DEPLOY THIS CLUSTER SETTING:** every tested combination produced at least one cross-person merge. Label more faces or test stricter values."
            )
        else:
            lines.append(
                "- The candidate produced zero observed cross-person merges in this calibration set. This is not a guarantee; confirm it on the holdout event."
            )
    else:
        lines.append(
            "- Cluster recommendation pending: fill `face-labels.csv` and run the command again."
        )
    lines.extend(["", "## Ranked matching combinations", ""])
    if matching:
        lines.extend(
            [
                "| Rank | Possible | Confident | Confident FP | Possible FP | Possible recall |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rank, result in enumerate(matching[:25], 1):
            lines.append(
                f"| {rank} | {result.possible_threshold:.3f} | {result.confident_threshold:.3f} | "
                f"{result.confident.false_positives} | {result.possible.false_positives} | "
                f"{result.possible.recall:.2%} |"
            )
    else:
        lines.append("No matching combinations were evaluated.")
    lines.extend(["", "## Ranked clustering combinations", ""])
    if clustering:
        lines.extend(
            [
                "| Rank | Edge | Cohesion | False merges | False splits | Purity | Unassigned |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rank, result in enumerate(clustering[:25], 1):
            lines.append(
                f"| {rank} | {result.edge_similarity:.3f} | {result.cohesion_similarity:.3f} | "
                f"{result.false_merges} | {result.false_splits} | "
                f"{result.cluster_purity:.2%} | {result.unassigned_faces} |"
            )
    else:
        lines.append("No clustering combinations were evaluated yet.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, report_path
