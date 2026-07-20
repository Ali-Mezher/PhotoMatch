"""
Issue #17 — manual vs. automatic search time savings.

Turns a measured automatic search time into a direct comparison against
the proposal's stated manual baseline (5-6 hours per student). This is
the headline number for the final report — the whole project's value
compressed into one comparison.
"""

from dataclasses import dataclass

from config import MANUAL_SEARCH_BASELINE_SECONDS


@dataclass(frozen=True)
class TimeSavings:
    automatic_seconds: float
    manual_baseline_seconds: float
    seconds_saved: float
    percent_faster: float


def compute_time_savings(
    automatic_seconds: float,
    manual_baseline_seconds: float = MANUAL_SEARCH_BASELINE_SECONDS,
) -> TimeSavings:
    """
    Compare a measured automatic search time against the manual baseline.

    Args:
        automatic_seconds: real measured end-to-end search time — use
            BenchmarkResult.mean_query_time_seconds from a real
            benchmark run, not an estimate.
        manual_baseline_seconds: defaults to the proposal's 5-6 hour
            estimate (midpoint: 5.5 hours). Override with a more precise
            number if the business provides one.

    Returns:
        TimeSavings with the absolute and percentage improvement.

    Raises:
        ValueError: if either input isn't a valid positive time.
    """
    if automatic_seconds < 0:
        raise ValueError("automatic_seconds cannot be negative")
    if manual_baseline_seconds <= 0:
        raise ValueError("manual_baseline_seconds must be positive")

    seconds_saved = manual_baseline_seconds - automatic_seconds
    percent_faster = (seconds_saved / manual_baseline_seconds) * 100

    return TimeSavings(
        automatic_seconds=automatic_seconds,
        manual_baseline_seconds=manual_baseline_seconds,
        seconds_saved=seconds_saved,
        percent_faster=percent_faster,
    )
