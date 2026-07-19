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

        vectors = np.vstack(embeddings).astype(np.float32)
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
        if len(self.metadata) == 0:
            return [], []

        k = min(k, len(self.metadata))
        query = query_embedding.astype(np.float32).reshape(1, -1)
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

        return instance

    def __len__(self) -> int:
        return len(self.metadata)
