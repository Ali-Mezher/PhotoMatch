"""Local evaluation artifacts for matching scores and human face labels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.indexing import EventIndex

from .benchmark import BenchmarkResult
from .ground_truth import EvaluationSet


FACE_LABEL_FIELDS = (
    "face_id",
    "photo_filename",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "crop_filename",
    "possible_identities",
    "identity",
)


@dataclass(frozen=True)
class FaceLabelWorkspace:
    worksheet_path: Path
    crops_dir: Path
    face_count: int
    labeled_count: int


@dataclass(frozen=True)
class LabeledEmbeddingExport:
    csv_path: Path
    embeddings: np.ndarray
    identities: tuple[str, ...]
    face_ids: tuple[str, ...]
    unlabeled_count: int


def save_matching_scores(benchmark: BenchmarkResult, output_dir: Path) -> Path:
    """Save every deduplicated query-to-photo score with its ground truth."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "matching-scores.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "score",
                "label",
                "identity",
                "selfie_filename",
                "candidate_filename",
            ),
        )
        writer.writeheader()
        for observation in benchmark.score_observations:
            writer.writerow(
                {
                    "score": f"{observation.score:.10f}",
                    "label": "genuine" if observation.genuine else "impostor",
                    "identity": observation.identity,
                    "selfie_filename": observation.selfie_filename,
                    "candidate_filename": observation.candidate_filename,
                }
            )
    return path


def _row_key(row: dict[str, str]) -> tuple[str, int, int, int, int] | None:
    try:
        return (
            row["photo_filename"],
            int(row["bbox_x"]),
            int(row["bbox_y"]),
            int(row["bbox_w"]),
            int(row["bbox_h"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _existing_labels(path: Path) -> dict[tuple[str, int, int, int, int], str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        labels = {}
        for row in csv.DictReader(handle):
            key = _row_key(row)
            identity = (row.get("identity") or "").strip()
            if key is not None and identity:
                labels[key] = identity
        return labels


def _crop_face(image: np.ndarray, bbox: Sequence[int]) -> np.ndarray:
    x, y, width, height = (int(value) for value in bbox)
    image_height, image_width = image.shape[:2]
    left = max(0, x)
    top = max(0, y)
    right = min(image_width, x + width)
    bottom = min(image_height, y + height)
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid indexed face box: {(x, y, width, height)}")
    return image[top:bottom, left:right]


def prepare_face_label_workspace(
    evaluation_set: EvaluationSet,
    index: EventIndex,
    output_dir: Path,
) -> FaceLabelWorkspace:
    """Create face crops and a worksheet without guessing anyone's identity.

    Existing human-entered labels are preserved when the same photo/bounding box
    remains in the active index. ``possible_identities`` is only a reminder of
    who is known to appear somewhere in that photograph; it is never treated as
    the face's ground-truth identity.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worksheet_path = output_dir / "face-labels.csv"
    crops_dir = output_dir / "face-label-crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    existing = _existing_labels(worksheet_path)

    identities_by_photo: dict[str, set[str]] = {}
    for identity in evaluation_set.identities:
        for filename in identity.ground_truth_filenames:
            identities_by_photo.setdefault(filename, set()).add(identity.name)

    image_cache: dict[Path, np.ndarray] = {}
    rows: list[dict[str, str | int]] = []
    labeled_count = 0
    for position, metadata in enumerate(index.metadata):
        photo_filename = Path(metadata.photo_path).name
        if photo_filename not in identities_by_photo:
            continue
        bbox = tuple(int(value) for value in metadata.bbox)
        key = (photo_filename, *bbox)
        identity = existing.get(key, "")
        labeled_count += int(bool(identity))

        photo_path = Path(metadata.photo_path)
        if not photo_path.is_file():
            photo_path = evaluation_set.raw_dir / photo_filename
        resolved_photo = photo_path.resolve()
        if resolved_photo not in image_cache:
            image = cv2.imread(str(resolved_photo))
            if image is None:
                raise ValueError(f"Could not read indexed photo for labeling: {photo_path}")
            image_cache[resolved_photo] = image

        crop_name = f"face-{position:06d}.jpg"
        crop_path = crops_dir / crop_name
        crop = _crop_face(image_cache[resolved_photo], bbox)
        if not cv2.imwrite(str(crop_path), crop):
            raise OSError(f"Could not write evaluation face crop: {crop_path}")

        rows.append(
            {
                "face_id": f"face-{position:06d}",
                "photo_filename": photo_filename,
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_w": bbox[2],
                "bbox_h": bbox[3],
                "crop_filename": str(Path("face-label-crops") / crop_name),
                "possible_identities": ";".join(
                    sorted(identities_by_photo.get(photo_filename, set()))
                ),
                "identity": identity,
            }
        )

    with worksheet_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FACE_LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return FaceLabelWorkspace(
        worksheet_path=worksheet_path,
        crops_dir=crops_dir,
        face_count=len(rows),
        labeled_count=labeled_count,
    )


def export_labeled_embeddings(
    index: EventIndex,
    worksheet_path: Path,
    output_dir: Path,
    *,
    allowed_identities: set[str],
) -> LabeledEmbeddingExport:
    """Join human-confirmed face labels to index vectors and write a grid CSV."""
    worksheet_path = Path(worksheet_path)
    if not worksheet_path.is_file():
        raise FileNotFoundError(f"Face-label worksheet not found: {worksheet_path}")

    vectors = index.embeddings()
    selected_vectors: list[np.ndarray] = []
    identities: list[str] = []
    face_ids: list[str] = []
    seen_positions: set[int] = set()

    with worksheet_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(FACE_LABEL_FIELDS).issubset(reader.fieldnames):
            raise ValueError("face-labels.csv has missing or changed columns")
        rows = list(reader)

    for row_number, row in enumerate(rows, 2):
        identity = (row.get("identity") or "").strip()
        if not identity:
            continue
        if identity not in allowed_identities:
            raise ValueError(
                f"face-labels.csv row {row_number} has unknown identity '{identity}'"
            )
        face_id = row["face_id"].strip()
        if not face_id.startswith("face-"):
            raise ValueError(
                f"face-labels.csv row {row_number} has invalid face_id '{face_id}'"
            )
        try:
            position = int(face_id.removeprefix("face-"))
        except ValueError as exc:
            raise ValueError(
                f"face-labels.csv row {row_number} has invalid face_id '{face_id}'"
            ) from exc
        if position < 0 or position >= len(index.metadata) or position in seen_positions:
            raise ValueError(
                f"face-labels.csv row {row_number} does not uniquely match the active index"
            )
        metadata = index.metadata[position]
        expected_key = (Path(metadata.photo_path).name, *tuple(metadata.bbox))
        if _row_key(row) != expected_key:
            raise ValueError(
                "face-labels.csv no longer matches the active event index; "
                "regenerate the worksheet and review its preserved labels"
            )
        seen_positions.add(position)
        selected_vectors.append(vectors[position])
        identities.append(identity)
        face_ids.append(face_id)

    counts = {identity: identities.count(identity) for identity in set(identities)}
    if len(counts) < 2 or any(count < 2 for count in counts.values()):
        raise ValueError(
            "Clustering evaluation needs at least two labeled identities with "
            "at least two face crops each."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "labeled-embeddings.csv"
    embedding_columns = [f"e{index}" for index in range(vectors.shape[1])]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("face_id", "identity", "photo_filename", *embedding_columns),
        )
        writer.writeheader()
        for face_id, identity, vector in zip(face_ids, identities, selected_vectors):
            position = int(face_id.removeprefix("face-"))
            row = {
                "face_id": face_id,
                "identity": identity,
                "photo_filename": Path(index.metadata[position].photo_path).name,
            }
            row.update(
                {column: f"{float(value):.10f}" for column, value in zip(embedding_columns, vector)}
            )
            writer.writerow(row)

    return LabeledEmbeddingExport(
        csv_path=csv_path,
        embeddings=np.asarray(selected_vectors, dtype=np.float32),
        identities=tuple(identities),
        face_ids=tuple(face_ids),
        unlabeled_count=len(rows) - len(selected_vectors),
    )
