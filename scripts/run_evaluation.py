"""
Runs the full Week 4 evaluation pipeline for one event: precision/recall
+ query time (#13), FAR/FRR threshold tuning (#14), scalability (#15),
and manual-vs-automatic time savings (#17) — then saves a combined
markdown report to data/evaluation/<event_id>/report.md.

Setup required before running (see src/evaluation/ground_truth.py for
the exact layout):

    1. Index the event normally:
       python scripts/demo_index_and_search.py index <event_id>

    2. Under data/evaluation/<event_id>/identities/, create one folder
       per person with a selfie/ and matches/ subfolder — drop their
       selfie in selfie/, and copies of their known event photos
       (same filenames as in data/events/<event_id>/raw/) in matches/.

Usage:
    python scripts/run_evaluation.py <event_id> [--k 10] [--scalability]

--scalability also runs the synthetic FAISS scaling suite (adds ~1-2
minutes) — skip it for a quick precision/recall/FAR-FRR check.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import (
    compute_time_savings,
    generate_report,
    run_benchmark,
    run_scalability_suite,
    save_report,
    tune_confidence_tiers,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_id")
    parser.add_argument("--k", type=int, default=10, help="top-k for precision/recall")
    parser.add_argument(
        "--scalability", action="store_true", help="also run the FAISS scaling suite"
    )
    parser.add_argument(
        "--confident-max-far", type=float, default=0.01,
        help="max acceptable FAR for the confident-match threshold",
    )
    parser.add_argument(
        "--possible-max-far", type=float, default=0.10,
        help="max acceptable FAR for the possible-match threshold",
    )
    args = parser.parse_args()

    print(f"Running benchmark for event '{args.event_id}' (k={args.k})...")
    try:
        benchmark = run_benchmark(args.event_id, k=args.k)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}")
        return 1

    print(f"Evaluated {len(benchmark.identity_results)} identities.")
    print(f"  Mean precision@{args.k}: {benchmark.mean_precision:.1%}")
    print(f"  Mean recall@{args.k}: {benchmark.mean_recall:.1%}")
    print(f"  Mean query time: {benchmark.mean_query_time_seconds:.2f}s")

    confident = possible = None
    all_scores = benchmark.genuine_scores + benchmark.impostor_scores
    all_labels = [True] * len(benchmark.genuine_scores) + [False] * len(benchmark.impostor_scores)
    if benchmark.genuine_scores and benchmark.impostor_scores:
        print("\nTuning FAR/FRR thresholds from real collected scores...")
        confident, possible = tune_confidence_tiers(
            all_scores, all_labels,
            confident_max_far=args.confident_max_far,
            possible_max_far=args.possible_max_far,
        )
        print(f"  Confident: {confident.threshold:.4f} (FAR {confident.far:.2%}, FRR {confident.frr:.2%})")
        print(f"  Possible:  {possible.threshold:.4f} (FAR {possible.far:.2%}, FRR {possible.frr:.2%})")
    else:
        print(
            "\nNot enough genuine/impostor scores to tune thresholds yet "
            "(need at least one of each) — add more identities and re-run."
        )

    scalability = None
    if args.scalability:
        print("\nRunning scalability suite (this can take a minute or two)...")
        scalability = run_scalability_suite()
        for s in scalability:
            print(f"  {s.num_faces:,} faces: build {s.build_seconds:.2f}s, search {s.mean_search_seconds*1000:.2f}ms")

    time_savings = compute_time_savings(benchmark.mean_query_time_seconds)
    print(
        f"\nTime savings vs. manual baseline: "
        f"{time_savings.percent_faster:.2f}% faster "
        f"({time_savings.automatic_seconds:.2f}s vs. "
        f"{time_savings.manual_baseline_seconds/3600:.1f}h)"
    )

    report = generate_report(
        benchmark, time_savings, confident=confident, possible=possible, scalability=scalability
    )
    report_path = save_report(args.event_id, report)
    print(f"\nFull report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
