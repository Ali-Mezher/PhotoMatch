"""Tests for src/evaluation/time_savings.py."""

import pytest

from src.evaluation.time_savings import compute_time_savings


class TestComputeTimeSavings:
    def test_typical_case(self):
        result = compute_time_savings(automatic_seconds=2.0, manual_baseline_seconds=3600)
        assert result.seconds_saved == pytest.approx(3598.0)
        assert result.percent_faster == pytest.approx(99.944, abs=0.01)

    def test_uses_config_default_baseline(self):
        from config import MANUAL_SEARCH_BASELINE_SECONDS
        result = compute_time_savings(automatic_seconds=1.0)
        assert result.manual_baseline_seconds == MANUAL_SEARCH_BASELINE_SECONDS

    def test_rejects_negative_automatic_time(self):
        with pytest.raises(ValueError):
            compute_time_savings(automatic_seconds=-1.0)

    def test_rejects_non_positive_baseline(self):
        with pytest.raises(ValueError):
            compute_time_savings(automatic_seconds=1.0, manual_baseline_seconds=0)

    def test_automatic_slower_than_baseline_gives_negative_savings(self):
        # sanity check: if automatic search were somehow slower than the
        # baseline, this should show negative savings, not crash or clamp.
        result = compute_time_savings(automatic_seconds=999999, manual_baseline_seconds=3600)
        assert result.seconds_saved < 0
        assert result.percent_faster < 0
