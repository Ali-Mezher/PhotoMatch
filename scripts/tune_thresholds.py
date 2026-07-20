"""Recommend PhotoMatch confidence thresholds from labeled scores.

CSV format (one comparison per row):

    score,label
    0.86,genuine
    0.42,impostor

Labels may be ``genuine``/``impostor`` or ``1``/``0``. A genuine row
compares the same person; an impostor row compares different people.

Usage:
    python scripts/tune_thresholds.py scores.csv
    python scripts/tune_thresholds.py scores.csv --confident-max-far 0.01 --possible-max-far 0.10
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import ThresholdMetrics, tune_confidence_tiers


def _read_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    labels = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"score", "label"}.issubset(reader.fieldnames):
            raise ValueError("CSV must contain score and label columns")
        for row_number, row in enumerate(reader, start=2):
            try:
                score = float(row["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"row {row_number}: invalid score") from exc
            label = row["label"].strip().lower()
            if label in {"genuine", "1", "true"}:
                genuine = True
            elif label in {"impostor", "0", "false"}:
                genuine = False
            else:
                raise ValueError(f"row {row_number}: invalid label '{row['label']}'")
            scores.append(score)
            labels.append(genuine)
    return np.asarray(scores), np.asarray(labels)


def _print_result(name: str, result: ThresholdMetrics) -> None:
    print(f"{name}: {result.threshold:.6f}")
    print(
        f"  FAR {result.far:.2%} ({result.false_accepts}/{result.impostor_count}), "
        f"FRR {result.frr:.2%} ({result.false_rejects}/{result.genuine_count})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--confident-max-far", type=float, default=0.01)
    parser.add_argument("--possible-max-far", type=float, default=0.10)
    args = parser.parse_args()

    try:
        scores, labels = _read_scores(args.csv_path)
        confident, possible = tune_confidence_tiers(
            scores,
            labels,
            confident_max_far=args.confident_max_far,
            possible_max_far=args.possible_max_far,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Evaluated {int(labels.sum())} genuine and {int((~labels).sum())} impostor scores.\n")
    _print_result("Confident threshold", confident)
    _print_result("Possible threshold", possible)
    print("\nSuggested config.py values:")
    print(f"CONFIDENT_MATCH_THRESHOLD = {confident.threshold:.6f}")
    print(f"POSSIBLE_MATCH_THRESHOLD = {possible.threshold:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
