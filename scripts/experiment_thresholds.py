"""Compare PhotoMatch search and identity-group thresholds.

Matching CSV columns: ``score,label`` where label is genuine/impostor or 1/0.
Clustering CSV columns: ``face_id,identity,e0,e1,...`` with one normalized or
unnormalized embedding per row.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.threshold_experiments import (
    evaluate_clustering_grid,
    evaluate_matching_grid,
    inclusive_grid,
    save_threshold_artifacts,
)


def _range(value: str) -> list[float]:
    try:
        start, end, step = (float(part) for part in value.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use START:END:STEP, for example 0.55:0.80:0.05") from exc
    try:
        return inclusive_grid(start, end, step)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _read_matching(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    labels: list[bool] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"score", "label"}.issubset(reader.fieldnames):
            raise ValueError("matching CSV must contain score and label columns")
        for row_number, row in enumerate(reader, 2):
            try:
                scores.append(float(row["score"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"matching row {row_number} has an invalid score") from exc
            label = row["label"].strip().lower()
            if label in {"genuine", "1", "true"}:
                labels.append(True)
            elif label in {"impostor", "0", "false"}:
                labels.append(False)
            else:
                raise ValueError(f"matching row {row_number} has an invalid label")
    return np.asarray(scores), np.asarray(labels)


def _embedding_column_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else 10**9, name)


def _read_clustering(path: Path) -> tuple[np.ndarray, list[str]]:
    identities: list[str] = []
    vectors: list[list[float]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        embedding_columns = sorted(
            [name for name in fieldnames if re.fullmatch(r"(?:e|embedding_)\d+", name)],
            key=_embedding_column_key,
        )
        if "identity" not in fieldnames or not embedding_columns:
            raise ValueError("clustering CSV must contain identity and e0,e1,... columns")
        for row_number, row in enumerate(reader, 2):
            identity = row["identity"].strip()
            if not identity:
                raise ValueError(f"clustering row {row_number} has a blank identity")
            try:
                vector = [float(row[column]) for column in embedding_columns]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"clustering row {row_number} has an invalid embedding") from exc
            identities.append(identity)
            vectors.append(vector)
    return np.asarray(vectors, dtype=np.float32), identities


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching-csv", type=Path)
    parser.add_argument("--clustering-csv", type=Path)
    parser.add_argument("--possible-range", type=_range, default=_range("0.45:0.70:0.05"))
    parser.add_argument("--confident-range", type=_range, default=_range("0.55:0.85:0.05"))
    parser.add_argument("--edge-range", type=_range, default=_range("0.60:0.85:0.05"))
    parser.add_argument("--cohesion-range", type=_range, default=_range("0.60:0.85:0.05"))
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("threshold_experiments"))
    args = parser.parse_args()
    if not args.matching_csv and not args.clustering_csv:
        parser.error("provide --matching-csv, --clustering-csv, or both")

    try:
        matching = []
        clustering = []
        if args.matching_csv:
            scores, labels = _read_matching(args.matching_csv)
            matching = evaluate_matching_grid(
                scores, labels, args.possible_range, args.confident_range
            )
        if args.clustering_csv:
            embeddings, identities = _read_clustering(args.clustering_csv)
            clustering = evaluate_clustering_grid(
                embeddings,
                identities,
                args.edge_range,
                args.cohesion_range,
                neighbors=args.neighbors,
                min_cluster_size=args.min_cluster_size,
            )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    csv_path, report_path = save_threshold_artifacts(
        args.output_dir, matching, clustering
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
