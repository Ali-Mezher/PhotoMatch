"""Conservative, event-scoped face clustering using a sparse FAISS graph."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np

from config import (
    CLUSTER_COHESION_SIMILARITY,
    CLUSTER_EDGE_SIMILARITY,
    CLUSTER_MIN_SIZE,
    CLUSTER_NEIGHBORS,
    EVENT_INDEXED_SUBDIR,
    event_dir,
)
from src.indexing import EventIndex, IndexedFace, load_event_index

CLUSTERS_FILENAME = "clusters.json"
CLUSTER_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ClusterMember:
    """One face included in a candidate identity cluster."""

    face_index: int
    photo_path: str
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class FaceCluster:
    """A staff-review candidate group, not an automatic identity label."""

    cluster_id: str
    representative: ClusterMember
    members: list[ClusterMember]
    mean_similarity_to_representative: float


@dataclass(frozen=True)
class ClusterResult:
    """Clusters and intentionally unassigned faces for one event."""

    clusters: list[FaceCluster]
    unclustered: list[ClusterMember]
    neighbors: int
    edge_similarity: float
    cohesion_similarity: float
    min_cluster_size: int


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def cluster_index(
    index: EventIndex,
    *,
    neighbors: int = CLUSTER_NEIGHBORS,
    edge_similarity: float = CLUSTER_EDGE_SIMILARITY,
    cohesion_similarity: float = CLUSTER_COHESION_SIMILARITY,
    min_cluster_size: int = CLUSTER_MIN_SIZE,
) -> ClusterResult:
    """Cluster one event's faces without comparing faces across events.

    A pair becomes a graph edge only when it is a mutual k-nearest-neighbor
    match and clears the direct similarity threshold. Components are then
    checked against their medoid; weakly attached faces remain unclustered.
    """
    _validate_parameters(neighbors, edge_similarity, cohesion_similarity, min_cluster_size)
    embeddings = index.embeddings()
    if len(embeddings) != len(index.metadata):
        raise ValueError("Index embeddings and metadata are inconsistent")
    if not len(embeddings):
        return ClusterResult([], [], neighbors, edge_similarity, cohesion_similarity, min_cluster_size)

    member_lookup = [_member(i, metadata) for i, metadata in enumerate(index.metadata)]
    if len(embeddings) == 1:
        return ClusterResult([], member_lookup, neighbors, edge_similarity, cohesion_similarity, min_cluster_size)

    k = min(neighbors + 1, len(embeddings))
    graph = faiss.IndexFlatIP(index.dim)
    graph.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    scores, neighbor_ids = graph.search(np.ascontiguousarray(embeddings, dtype=np.float32), k)

    neighbor_scores: list[dict[int, float]] = []
    for face_index, (row_scores, row_ids) in enumerate(zip(scores, neighbor_ids)):
        neighbor_scores.append(
            {
                int(candidate): float(score)
                for score, candidate in zip(row_scores, row_ids)
                if candidate != -1 and candidate != face_index
            }
        )

    union_find = _UnionFind(len(embeddings))
    for first, candidates in enumerate(neighbor_scores):
        for second, score in candidates.items():
            if second <= first or score < edge_similarity:
                continue
            reverse_score = neighbor_scores[second].get(first)
            if reverse_score is not None and reverse_score >= edge_similarity:
                union_find.union(first, second)

    components: dict[int, list[int]] = {}
    for face_index in range(len(embeddings)):
        components.setdefault(union_find.find(face_index), []).append(face_index)

    clusters: list[FaceCluster] = []
    unclustered_indices: set[int] = set()
    for members in components.values():
        retained, representative_index, similarities = _retain_cohesive_members(
            members, embeddings, cohesion_similarity
        )
        unclustered_indices.update(set(members) - set(retained))
        if len(retained) < min_cluster_size:
            unclustered_indices.update(retained)
            continue
        cluster_members = [member_lookup[index] for index in retained]
        clusters.append(
            FaceCluster(
                cluster_id="",
                representative=member_lookup[representative_index],
                members=cluster_members,
                mean_similarity_to_representative=float(np.mean(similarities)),
            )
        )

    clusters.sort(key=lambda cluster: (-len(cluster.members), cluster.representative.face_index))
    numbered_clusters = [
        FaceCluster(
            cluster_id=f"cluster_{number:04d}",
            representative=cluster.representative,
            members=cluster.members,
            mean_similarity_to_representative=cluster.mean_similarity_to_representative,
        )
        for number, cluster in enumerate(clusters, start=1)
    ]
    return ClusterResult(
        numbered_clusters,
        [member_lookup[index] for index in sorted(unclustered_indices)],
        neighbors,
        edge_similarity,
        cohesion_similarity,
        min_cluster_size,
    )


def cluster_event(event_id: str, **parameters) -> tuple[ClusterResult, Path]:
    """Cluster an indexed event and persist a local staff-review artifact."""
    result = cluster_index(load_event_index(event_id), **parameters)
    destination = event_dir(event_id) / EVENT_INDEXED_SUBDIR / CLUSTERS_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CLUSTER_FORMAT_VERSION,
        "event_id": event_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "neighbors": result.neighbors,
            "edge_similarity": result.edge_similarity,
            "cohesion_similarity": result.cohesion_similarity,
            "min_cluster_size": result.min_cluster_size,
        },
        "cluster_count": len(result.clusters),
        "unclustered_count": len(result.unclustered),
        "clusters": [asdict(cluster) for cluster in result.clusters],
        "unclustered": [asdict(member) for member in result.unclustered],
    }
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return result, destination


def load_cluster_artifact(event_id: str) -> dict:
    """Load a previously generated clustering artifact for an event.

    The artifact is deliberately the cache boundary for the desktop UI: once
    staff have reviewed a set of candidate groups, pressing Cluster again
    should present that same set instead of silently recalculating it.
    """
    source = event_dir(event_id) / EVENT_INDEXED_SUBDIR / CLUSTERS_FILENAME
    if not source.exists():
        raise FileNotFoundError(f"No clustering results exist for event '{event_id}'.")

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Saved clustering results for '{event_id}' are invalid.") from exc

    if payload.get("format_version") != CLUSTER_FORMAT_VERSION:
        raise ValueError(f"Saved clustering results for '{event_id}' use an unsupported format.")
    if payload.get("event_id") != event_id:
        raise ValueError("Saved clustering results belong to a different event.")
    if not isinstance(payload.get("clusters"), list):
        raise ValueError(f"Saved clustering results for '{event_id}' are incomplete.")
    return payload


def cluster_event_if_needed(event_id: str) -> tuple[dict, bool]:
    """Return saved clusters, regenerating them when defaults have changed."""
    try:
        payload = load_cluster_artifact(event_id)
        if payload.get("configuration") == _default_configuration():
            return payload, False
    except FileNotFoundError:
        pass
    cluster_event(event_id)
    return load_cluster_artifact(event_id), True


def _default_configuration() -> dict:
    return {
        "neighbors": CLUSTER_NEIGHBORS,
        "edge_similarity": CLUSTER_EDGE_SIMILARITY,
        "cohesion_similarity": CLUSTER_COHESION_SIMILARITY,
        "min_cluster_size": CLUSTER_MIN_SIZE,
    }


def _member(face_index: int, metadata: IndexedFace) -> ClusterMember:
    return ClusterMember(face_index, metadata.photo_path, tuple(metadata.bbox), metadata.confidence)


def _retain_cohesive_members(
    members: list[int], embeddings: np.ndarray, cohesion_similarity: float
) -> tuple[list[int], int, np.ndarray]:
    if len(members) == 1:
        return members, members[0], np.array([1.0], dtype=np.float32)
    component_vectors = embeddings[members]
    similarity_matrix = component_vectors @ component_vectors.T
    representative_position = int(np.argmax(similarity_matrix.mean(axis=1)))
    representative_index = members[representative_position]
    similarities = similarity_matrix[representative_position]
    retained_positions = np.flatnonzero(similarities >= cohesion_similarity)
    retained = [members[int(position)] for position in retained_positions]
    return retained, representative_index, similarities[retained_positions]


def _validate_parameters(neighbors: int, edge_similarity: float, cohesion_similarity: float, min_cluster_size: int) -> None:
    if neighbors <= 0:
        raise ValueError("neighbors must be greater than zero")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least two")
    for name, value in (("edge_similarity", edge_similarity), ("cohesion_similarity", cohesion_similarity)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
