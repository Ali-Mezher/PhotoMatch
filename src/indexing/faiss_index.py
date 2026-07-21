"""
Issue #8 — Per-event FAISS vector index.

Stores face embeddings for one event so a student's selfie can be
compared against every face in that event only (never across events —
see the "Scope by event" objective in the proposal).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from config import EMBEDDING_DIM

INDEX_FILENAME = "faces.faiss"
METADATA_FILENAME = "metadata.json"


@dataclass
class IndexedFace:
    """Metadata for one face stored in the index, alongside its vector."""

    photo_path: str
    bbox: tuple[int, int, int, int]
    confidence: float


class EventIndex:
    """
    Wraps a FAISS flat index for one event's faces.

    Uses IndexFlatIP (inner product) rather than IndexFlatL2, because
    src.detection.embeddings already L2-normalizes every embedding —
    for normalized vectors, inner product IS cosine similarity, and
    flat IP search gives exact (not approximate) nearest neighbors,
    which is worth the extra compute at the scale of one event
    (thousands, not millions, of faces).
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[IndexedFace] = []

    def add(self, embeddings: list[np.ndarray], metadata: list[IndexedFace]) -> None:
        """
        Add one or more face embeddings to the index.

        Args:
            embeddings: list of 1-D float32 vectors, length == self.dim.
            metadata: parallel list of IndexedFace, one per embedding.

        Raises:
            ValueError: if embeddings and metadata lengths don't match.
        """
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"add: got {len(embeddings)} embeddings but {len(metadata)} metadata entries"
            )
        if not embeddings:
            return

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"add: expected embeddings with shape (n, {self.dim}), "
                f"got {vectors.shape}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("add: embeddings must contain only finite values")

        # Keep cosine search correct even when an upstream caller provides
        # embeddings that have not already been normalized.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("add: zero-length embeddings cannot be indexed")
        vectors = np.ascontiguousarray(vectors / norms, dtype=np.float32)
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(
        self, query_embedding: np.ndarray, k: int = 50
    ) -> tuple[list[float], list[IndexedFace]]:
        """
        Find the k most similar faces in this event to a query embedding
        (typically a student's selfie embedding).

        Args:
            query_embedding: 1-D float32 vector, length == self.dim.
            k: max number of results to return.

        Returns:
            (scores, metadata) — parallel lists, sorted by score
            descending. scores are cosine similarities in [-1, 1] (in
            practice [0, 1] for face embeddings). Fewer than k results
            are returned if the index has fewer than k faces.
        """
        if k <= 0:
            raise ValueError("search: k must be greater than zero")
        if len(self.metadata) == 0:
            return [], []

        k = min(k, len(self.metadata))
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.dim:
            raise ValueError(
                f"search: expected an embedding with shape ({self.dim},), "
                f"got {query.shape}"
            )
        if not np.isfinite(query).all():
            raise ValueError("search: embedding must contain only finite values")
        norm = np.linalg.norm(query)
        if norm == 0:
            raise ValueError("search: a zero-length embedding cannot be searched")
        query = np.ascontiguousarray((query / norm).reshape(1, -1), dtype=np.float32)
        scores, indices = self.index.search(query, k)

        result_scores = []
        result_metadata = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 when fewer than k results exist
                continue
            result_scores.append(float(score))
            result_metadata.append(self.metadata[idx])

        return result_scores, result_metadata

    def save(self, directory: Path) -> None:
        """
        Persist the index and metadata to disk under `directory`, as
        `faces.faiss` and `metadata.json`.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / INDEX_FILENAME))

        with open(directory / METADATA_FILENAME, "w") as f:
            json.dump([asdict(m) for m in self.metadata], f)

    def embeddings(self) -> np.ndarray:
        """Return stored normalized embeddings in the same order as metadata.

        This is used by per-event clustering. The index remains the source of
        truth; vectors are reconstructed only when an operation needs the full
        event graph rather than a single nearest-neighbor query.
        """
        if not self.metadata:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.ascontiguousarray(
            self.index.reconstruct_n(0, len(self.metadata)), dtype=np.float32
        )

    @classmethod
    def load(cls, directory: Path) -> "EventIndex":
        """
        Load a previously saved index from `directory`.

        Raises:
            FileNotFoundError: if the index or metadata file is missing —
                callers (e.g. src.matching) should catch this and treat
                it as "this event hasn't been indexed yet".
        """
        directory = Path(directory)
        index_path = directory / INDEX_FILENAME
        metadata_path = directory / METADATA_FILENAME

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"No saved index found at {directory}")

        faiss_index = faiss.read_index(str(index_path))

        instance = cls(dim=faiss_index.d)
        instance.index = faiss_index

        with open(metadata_path) as f:
            raw_metadata = json.load(f)
        instance.metadata = [IndexedFace(**m) for m in raw_metadata]

        if faiss_index.ntotal != len(instance.metadata):
            raise ValueError(
                "Saved index is inconsistent: "
                f"{faiss_index.ntotal} vectors but {len(instance.metadata)} metadata entries"
            )

        return instance

    def __len__(self) -> int:
        return len(self.metadata)
