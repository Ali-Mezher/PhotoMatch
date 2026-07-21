"""
Detection pipeline — combines face detection and embedding extraction
into one entry point. This is what src/indexing calls per photo.
"""

from dataclasses import dataclass

import numpy as np

from .face_detector import FaceDetector, Detection
from .embeddings import EmbeddingExtractor
from src.preprocessing import crop_face_region

__all__ = [
    "FaceDetector",
    "Detection",
    "EmbeddingExtractor",
    "FaceEmbedding",
    "detect_faces",
    "embed_detection",
    "detect_and_embed",
]


@dataclass
class FaceEmbedding:
    """One detected-and-embedded face, ready for indexing."""

    bbox: tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray


# Module-level instances so weights load once per process, not once per
# photo — loading MTCNN/deepface per call would make indexing an entire
# event take hours instead of minutes.
_detector = FaceDetector()
_extractor = EmbeddingExtractor()


def detect_faces(image: np.ndarray) -> list[Detection]:
    """Detect faces with the process-wide detector instance."""
    return _detector.detect(image)


def embed_detection(image: np.ndarray, detection: Detection) -> np.ndarray:
    """Embed one detection with the process-wide embedding model."""
    face_crop = crop_face_region(image, detection.bbox, margin=0.2)
    return _extractor.get_embedding(face_crop)


def detect_and_embed(image: np.ndarray) -> list[FaceEmbedding]:
    """
    Full detect -> crop -> embed pipeline for one photo. This is what
    src/indexing should call per photo when building an event's index,
    and what src/matching should call once on a student's selfie.

    Args:
        image: preprocessed BGR photo (run through
            src.preprocessing.preprocess_image first).

    Returns:
        List of FaceEmbedding, one per face detected in the photo that
        passed confidence/size filtering AND successfully embedded.
        Empty list if no faces found — this is expected for some event
        photos (venue shots, decor) and should not raise.
    """
    detections = detect_faces(image)

    results = []
    for detection in detections:
        try:
            embedding = embed_detection(image, detection)
        except ValueError:
            # Bad crop or deepface couldn't process it — skip this face
            # rather than failing the whole photo.
            continue

        results.append(
            FaceEmbedding(
                bbox=detection.bbox,
                confidence=detection.confidence,
                embedding=embedding,
            )
        )

    return results
