"""Tests for src/evaluation/retrieval_metrics.py — pure math, no I/O."""

import pytest

from src.evaluation.retrieval_metrics import precision_recall_at_k


class TestPrecisionRecallAtK:
    def test_perfect_retrieval(self):
        m = precision_recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3)
        assert m.precision_at_k == 1.0
        assert m.recall_at_k == 1.0

    def test_no_relevant_retrieved(self):
        m = precision_recall_at_k(["x", "y", "z"], {"a", "b", "c"}, k=3)
        assert m.precision_at_k == 0.0
        assert m.recall_at_k == 0.0

    def test_partial_retrieval(self):
        m = precision_recall_at_k(["a", "x", "b", "y", "c"], {"a", "b", "c", "d"}, k=5)
        assert m.num_retrieved_relevant == 3
        assert m.precision_at_k == pytest.approx(3 / 5)
        assert m.recall_at_k == pytest.approx(3 / 4)

    def test_k_smaller_than_ranked_list_truncates(self):
        m = precision_recall_at_k(["a", "b", "x", "y", "z"], {"a", "b"}, k=2)
        assert m.num_retrieved_relevant == 2
        assert m.precision_at_k == 1.0
        assert m.recall_at_k == 1.0

    def test_k_larger_than_ranked_list(self):
        m = precision_recall_at_k(["a"], {"a", "b"}, k=10)
        assert m.num_retrieved_relevant == 1
        assert m.precision_at_k == 1.0  # precision uses len(top_k), which is 1 here
        assert m.recall_at_k == 0.5

    def test_empty_ranked_list(self):
        m = precision_recall_at_k([], {"a", "b"}, k=5)
        assert m.precision_at_k == 0.0
        assert m.recall_at_k == 0.0

    def test_rejects_non_positive_k(self):
        with pytest.raises(ValueError):
            precision_recall_at_k(["a"], {"a"}, k=0)
        with pytest.raises(ValueError):
            precision_recall_at_k(["a"], {"a"}, k=-1)
