"""
Evaluation package (Week 4) — everything needed to measure and validate
PhotoMatch against real data:

    ground_truth        load a hand-built evaluation set from disk
                         (data/evaluation/<event_id>/identities/...)
    retrieval_metrics    precision/recall@k
    benchmark            runs real selfies through the full pipeline,
                         producing per-identity metrics, timing, and
                         genuine/impostor scores
    threshold_tuning     FAR/FRR threshold tuning (Ahmad's original
                         Week 4 starter — unchanged)
    scalability          FAISS build/search timing at scale, and an
                         offline-operation check
    time_savings         automatic vs. manual search time comparison
    report               combines all of the above into one markdown
                         report for the final write-up
"""

from .ground_truth import EvaluationSet, Identity, load_evaluation_set
from .retrieval_metrics import RetrievalMetrics, precision_recall_at_k
from .benchmark import BenchmarkResult, IdentityResult, run_benchmark
from .threshold_tuning import (
    ThresholdMetrics,
    evaluate_threshold,
    tune_confidence_tiers,
    tune_threshold,
)
from .scalability import (
    NetworkAccessError,
    ScalabilityResult,
    benchmark_index_size,
    run_scalability_suite,
    verify_offline_operation,
)
from .time_savings import TimeSavings, compute_time_savings
from .report import generate_report, save_report

__all__ = [
    "EvaluationSet",
    "Identity",
    "load_evaluation_set",
    "RetrievalMetrics",
    "precision_recall_at_k",
    "BenchmarkResult",
    "IdentityResult",
    "run_benchmark",
    "ThresholdMetrics",
    "evaluate_threshold",
    "tune_threshold",
    "tune_confidence_tiers",
    "NetworkAccessError",
    "ScalabilityResult",
    "benchmark_index_size",
    "run_scalability_suite",
    "verify_offline_operation",
    "TimeSavings",
    "compute_time_savings",
    "generate_report",
    "save_report",
]
