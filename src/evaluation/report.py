"""
Generates a single markdown report combining retrieval accuracy (#13),
FAR/FRR thresholds (#14), scalability (#15), and time savings (#17) —
ready to paste into the Week 5 final report (#19).

Real-event robustness testing (#16) isn't a number this module can
compute on its own — it's a judgment call reviewing which photos failed
detection or matching and why (lighting, angle, occlusion, etc.). This
report DOES surface which identities scored worst, which is exactly the
list a robustness review should start from — see the "Lowest-scoring
identities" section below.
"""

from datetime import datetime, timezone
from pathlib import Path

from config import evaluation_dir
from .benchmark import BenchmarkResult
from .scalability import ScalabilityResult
from .threshold_tuning import ThresholdMetrics
from .time_savings import TimeSavings

ROBUSTNESS_REVIEW_COUNT = 3  # how many worst-scoring identities to flag


def generate_report(
    benchmark: BenchmarkResult,
    time_savings: TimeSavings,
    confident: ThresholdMetrics | None = None,
    possible: ThresholdMetrics | None = None,
    scalability: list[ScalabilityResult] | None = None,
) -> str:
    """
    Build the markdown report content. Pass None for confident/possible
    if there weren't enough genuine/impostor scores collected yet to
    tune thresholds (needs at least a couple of identities/events), and
    None for scalability if that suite hasn't been run.
    """
    lines: list[str] = []

    lines.append(f"# PhotoMatch Evaluation Report — {benchmark.event_id}")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    _add_retrieval_section(lines, benchmark)
    _add_threshold_section(lines, confident, possible)
    if scalability:
        _add_scalability_section(lines, scalability)
    _add_time_savings_section(lines, time_savings)
    _add_robustness_review_section(lines, benchmark)

    return "\n".join(lines)


def _add_retrieval_section(lines: list[str], benchmark: BenchmarkResult) -> None:
    lines.append("## Retrieval accuracy — Issue #13")
    lines.append(f"- Identities evaluated: {len(benchmark.identity_results)}")
    lines.append(f"- Mean precision@{benchmark.k}: {benchmark.mean_precision:.1%}")
    lines.append(f"- Mean recall@{benchmark.k}: {benchmark.mean_recall:.1%}")
    lines.append(f"- Mean query time: {benchmark.mean_query_time_seconds:.2f}s")
    lines.append("")
    lines.append("| Identity | Precision@k | Recall@k | Query time (s) | Top score |")
    lines.append("|---|---|---|---|---|")
    for r in benchmark.identity_results:
        top = f"{r.top_score:.3f}" if r.top_score is not None else "—"
        lines.append(
            f"| {r.name} | {r.metrics.precision_at_k:.1%} | "
            f"{r.metrics.recall_at_k:.1%} | {r.query_time_seconds:.2f} | {top} |"
        )
    lines.append("")


def _add_threshold_section(
    lines: list[str], confident: ThresholdMetrics | None, possible: ThresholdMetrics | None
) -> None:
    lines.append("## FAR / FRR thresholds — Issue #14")
    if confident is not None and possible is not None:
        lines.append(
            f"- Confident threshold: {confident.threshold:.4f} — "
            f"FAR {confident.far:.2%}, FRR {confident.frr:.2%}"
        )
        lines.append(
            f"- Possible threshold: {possible.threshold:.4f} — "
            f"FAR {possible.far:.2%}, FRR {possible.frr:.2%}"
        )
        lines.append(
            "- Update `config.py`'s `CONFIDENT_MATCH_THRESHOLD` / "
            "`POSSIBLE_MATCH_THRESHOLD` with these values once confirmed "
            "across more than one event."
        )
    else:
        lines.append(
            "- Not enough genuine/impostor scores collected yet to tune "
            "thresholds — run this benchmark across a few more identities "
            "or events first."
        )
    lines.append("")


def _add_scalability_section(lines: list[str], scalability: list[ScalabilityResult]) -> None:
    lines.append("## Scalability — Issue #15")
    lines.append("| Index size | Build time (s) | Mean search time (ms) |")
    lines.append("|---|---|---|")
    for s in scalability:
        lines.append(
            f"| {s.num_faces:,} | {s.build_seconds:.2f} | "
            f"{s.mean_search_seconds * 1000:.2f} |"
        )
    lines.append("")


def _add_time_savings_section(lines: list[str], time_savings: TimeSavings) -> None:
    lines.append("## Manual vs. automatic time savings — Issue #17")
    lines.append(
        f"- Manual baseline: {time_savings.manual_baseline_seconds / 3600:.1f} hours"
    )
    lines.append(f"- Automatic search: {time_savings.automatic_seconds:.2f} seconds")
    lines.append(f"- **{time_savings.percent_faster:.2f}% faster** than manual search")
    lines.append("")


def _add_robustness_review_section(lines: list[str], benchmark: BenchmarkResult) -> None:
    lines.append("## Starting point for robustness review — Issue #16")
    lines.append(
        "Precision/recall alone doesn't explain *why* a search underperformed "
        "— that needs a human look at the actual photos. These are the "
        f"{ROBUSTNESS_REVIEW_COUNT} lowest-scoring identities from this run; "
        "start there and note the likely cause (angle, lighting, occlusion, "
        "motion blur, group photo, etc.) for each."
    )
    lines.append("")

    worst = sorted(benchmark.identity_results, key=lambda r: r.metrics.recall_at_k)[
        :ROBUSTNESS_REVIEW_COUNT
    ]
    for r in worst:
        lines.append(
            f"- **{r.name}** — recall@{benchmark.k}: {r.metrics.recall_at_k:.1%} "
            f"({r.metrics.num_retrieved_relevant}/{r.metrics.num_relevant} found). "
            f"Cause: _fill in after reviewing the photos_."
        )
    lines.append("")


def save_report(event_id: str, content: str) -> Path:
    """Save report content to data/evaluation/<event_id>/report.md."""
    output_path = evaluation_dir(event_id) / "report.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path
