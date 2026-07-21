"""Tests for src/evaluation/report.py."""

from src.evaluation.benchmark import BenchmarkResult, IdentityResult
from src.evaluation.retrieval_metrics import RetrievalMetrics
from src.evaluation.scalability import ScalabilityResult
from src.evaluation.threshold_tuning import ThresholdMetrics
from src.evaluation.time_savings import compute_time_savings
from src.evaluation.report import generate_report, save_report


def _sample_benchmark():
    results = [
        IdentityResult("alice", RetrievalMetrics(10, 0.8, 0.9, 5, 4), 1.2, 0.91),
        IdentityResult("bob", RetrievalMetrics(10, 0.3, 0.4, 5, 2), 1.5, 0.61),
    ]
    return BenchmarkResult(
        event_id="grad2026", k=10, identity_results=results,
        genuine_scores=[0.9, 0.85], impostor_scores=[0.2, 0.1],
    )


class TestGenerateReport:
    def test_includes_all_core_sections(self):
        bm = _sample_benchmark()
        ts = compute_time_savings(bm.mean_query_time_seconds)
        report = generate_report(bm, ts)

        assert "grad2026" in report
        assert "Issue #13" in report
        assert "Issue #14" in report
        assert "Issue #17" in report
        assert "Issue #16" in report
        assert "alice" in report
        assert "bob" in report

    def test_handles_missing_thresholds_gracefully(self):
        bm = _sample_benchmark()
        ts = compute_time_savings(bm.mean_query_time_seconds)
        report = generate_report(bm, ts, confident=None, possible=None)
        assert "Not enough genuine/impostor scores" in report

    def test_includes_thresholds_when_provided(self):
        bm = _sample_benchmark()
        ts = compute_time_savings(bm.mean_query_time_seconds)
        confident = ThresholdMetrics(0.7, 0.01, 0.05, 1, 10, 20, 100)
        possible = ThresholdMetrics(0.5, 0.08, 0.02, 8, 4, 20, 100)
        report = generate_report(bm, ts, confident=confident, possible=possible)
        assert "0.7000" in report
        assert "0.5000" in report

    def test_includes_scalability_when_provided(self):
        bm = _sample_benchmark()
        ts = compute_time_savings(bm.mean_query_time_seconds)
        scalability = [ScalabilityResult(1000, 0.05, 0.001)]
        report = generate_report(bm, ts, scalability=scalability)
        assert "Issue #15" in report
        assert "1,000" in report

    def test_omits_scalability_section_when_not_provided(self):
        bm = _sample_benchmark()
        ts = compute_time_savings(bm.mean_query_time_seconds)
        report = generate_report(bm, ts, scalability=None)
        assert "Issue #15" not in report

    def test_flags_lowest_scoring_identities_first(self):
        bm = _sample_benchmark()
        ts = compute_time_savings(bm.mean_query_time_seconds)
        report = generate_report(bm, ts)
        bob_index = report.index("**bob**")
        alice_index = report.index("**alice**")
        assert bob_index < alice_index  # bob has lower recall, should appear first


def test_save_report_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr("config.EVALUATION_DIR", tmp_path)
    path = save_report("grad2026", "# hello")
    assert path.exists()
    assert path.read_text() == "# hello"
    assert path.parent.name == "grad2026"
