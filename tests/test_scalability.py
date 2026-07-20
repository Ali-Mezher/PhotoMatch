"""Tests for src/evaluation/scalability.py."""

import socket

import pytest

from src.evaluation.scalability import (
    NetworkAccessError,
    benchmark_index_size,
    run_scalability_suite,
    verify_offline_operation,
)


class TestBenchmarkIndexSize:
    def test_returns_positive_timings(self):
        result = benchmark_index_size(500, num_queries=5)
        assert result.num_faces == 500
        assert result.build_seconds > 0
        assert result.mean_search_seconds > 0

    def test_rejects_non_positive_num_faces(self):
        with pytest.raises(ValueError):
            benchmark_index_size(0)
        with pytest.raises(ValueError):
            benchmark_index_size(-10)

    def test_reproducible_with_same_seed(self):
        # not a timing assertion (that varies machine to machine) — just
        # confirms the same seed builds an index of the same size both times.
        a = benchmark_index_size(200, num_queries=3, seed=7)
        b = benchmark_index_size(200, num_queries=3, seed=7)
        assert a.num_faces == b.num_faces == 200


def test_run_scalability_suite_covers_all_sizes():
    results = run_scalability_suite(sizes=(100, 500))
    assert [r.num_faces for r in results] == [100, 500]


class TestVerifyOfflineOperation:
    def test_passes_through_return_value(self):
        assert verify_offline_operation(lambda x, y: x + y, 2, 3) == 5

    def test_blocks_socket_creation(self):
        def opens_a_socket():
            return socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        with pytest.raises(NetworkAccessError):
            verify_offline_operation(opens_a_socket)

    def test_restores_socket_after_success(self):
        original = socket.socket
        verify_offline_operation(lambda: 1 + 1)
        assert socket.socket is original

    def test_restores_socket_after_blocked_attempt(self):
        original = socket.socket

        def opens_a_socket():
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        with pytest.raises(NetworkAccessError):
            verify_offline_operation(opens_a_socket)

        assert socket.socket is original
