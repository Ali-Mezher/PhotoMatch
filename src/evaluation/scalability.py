"""
Issue #15 — scalability & offline-operation validation.

Two independent things, both about whether the system holds up at real
deployment scale and stays fully local:

1. benchmark_index_size() / run_scalability_suite() — measures FAISS
   build and search time as the number of indexed faces grows toward
   the 50,000-100,000+ range from the proposal. Uses synthetic random
   embeddings on purpose: FAISS's speed depends on vector count and
   dimension, not on what the vectors represent, so this doesn't need
   tens of thousands of real labeled photos to be a valid test.

2. verify_offline_operation() — proves a given function makes no
   network connections while it runs, backing up the proposal's "fully
   local, no cloud upload" claim with an actual check rather than just
   an assertion in the report.
"""

import socket
import time
from dataclasses import dataclass

import numpy as np

from config import EMBEDDING_DIM
from src.indexing import EventIndex, IndexedFace


@dataclass(frozen=True)
class ScalabilityResult:
    """Timing result for one index size."""

    num_faces: int
    build_seconds: float
    mean_search_seconds: float


def benchmark_index_size(
    num_faces: int, num_queries: int = 20, seed: int = 0
) -> ScalabilityResult:
    """
    Build a synthetic index of `num_faces` random unit vectors and time
    both construction and average per-query search time.

    Args:
        num_faces: how many synthetic embeddings to index.
        num_queries: how many random queries to average search time over.
        seed: RNG seed, for reproducible benchmark numbers across runs.

    Returns:
        ScalabilityResult with build time and mean search time.

    Raises:
        ValueError: if num_faces isn't positive.
    """
    if num_faces <= 0:
        raise ValueError("num_faces must be positive")

    rng = np.random.default_rng(seed)

    vectors = rng.random((num_faces, EMBEDDING_DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    index = EventIndex(dim=EMBEDDING_DIM)
    metadata = [
        IndexedFace(photo_path=f"synthetic_{i}.jpg", bbox=(0, 0, 1, 1), confidence=1.0)
        for i in range(num_faces)
    ]

    start = time.perf_counter()
    index.add(list(vectors), metadata)
    build_seconds = time.perf_counter() - start

    queries = rng.random((num_queries, EMBEDDING_DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    start = time.perf_counter()
    for query in queries:
        index.search(query, k=50)
    mean_search_seconds = (time.perf_counter() - start) / num_queries

    return ScalabilityResult(
        num_faces=num_faces,
        build_seconds=build_seconds,
        mean_search_seconds=mean_search_seconds,
    )


def run_scalability_suite(
    sizes: tuple[int, ...] = (1_000, 10_000, 50_000, 100_000),
) -> list[ScalabilityResult]:
    """
    Run benchmark_index_size() across a range of index sizes, from small
    up to the proposal's target scale.

    Note: 100,000 faces takes real time and memory to build (a few
    seconds to a minute or two, and a few hundred MB, depending on the
    machine) — this is expected, not a bug.
    """
    return [benchmark_index_size(n) for n in sizes]


class NetworkAccessError(RuntimeError):
    """Raised in place of an actual network connection during a
    verify_offline_operation() check."""


def verify_offline_operation(fn, *args, **kwargs):
    """
    Run `fn(*args, **kwargs)` with socket creation blocked, to prove it
    needs no network access — backs up the "fully local, no cloud"
    claim with an actual check instead of just an assertion.

    Args:
        fn: the function to test (e.g. a lambda wrapping an indexing or
            matching call on already-downloaded models).
        *args, **kwargs: passed through to fn.

    Returns:
        Whatever fn returns, if it completed without attempting network
        access.

    Raises:
        NetworkAccessError: if fn tried to open a socket at any point —
            re-raised from inside fn's call stack, so the traceback
            still points at what tried to connect.

    Note:
        This only catches socket.socket() calls — the layer nearly
        every Python networking library (requests, urllib, gRPC)
        eventually goes through. It won't catch something reading from
        an already-open connection, but that's not a realistic failure
        mode here since nothing in this codebase keeps a connection
        open between calls. Good enough to confirm no outbound
        HTTP/DNS/API calls are being made.

    Important:
        Only run this AFTER model weights are already downloaded once
        (deepface/mtcnn download weights on first use). Testing a
        completely fresh install with this will correctly report a
        network access attempt for the download itself — that's
        expected and not a bug in the pipeline.
    """
    original_socket = socket.socket

    def _blocked_socket(*args, **kwargs):
        raise NetworkAccessError(
            "Network access was attempted during an operation that must "
            "run fully offline. If this is the first run, model weights "
            "may not be downloaded yet — run the operation once online "
            "first, then retry this check."
        )

    socket.socket = _blocked_socket
    try:
        return fn(*args, **kwargs)
    finally:
        socket.socket = original_socket
