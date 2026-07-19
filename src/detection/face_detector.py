"""
Issue #6 — Face detector (MTCNN).

Wraps the `mtcnn` package so the rest of the pipeline (indexing, matching)
depends on a stable in-house interface rather than a third-party library's
exact API — if we ever swap MTCNN for RetinaFace, only this file changes.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from config import MIN_DETECTION_CONFIDENCE, MIN_FACE_SIZE_PX


@dataclass
class Detection:
    """One detected face within a photo."""

    bbox: tuple[int, int, int, int]  # (x, y, width, height) in pixels
    confidence: float
    keypoints: dict  # eyes, nose, mouth corners — useful for face alignment later


class FaceDetector:
    """
    Thin wrapper around MTCNN. Lazily loads the underlying model on first
    use so importing this module doesn't pay the model-load cost if
    detection isn't actually needed yet (e.g. during preprocessing-only
    unit tests).
    """

    def __init__(
        self,
        min_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_face_size: int = MIN_FACE_SIZE_PX,
    ):
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self._detector = None  # loaded lazily, see _ensure_loaded()

    def _ensure_loaded(self):
        if self._detector is None:
            from mtcnn import MTCNN  # deferred import — heavy, TF-backed

            self._detector = MTCNN()

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Detect all faces in a photo.

        Args:
            image: BGR image, uint8 (as loaded by cv2.imread). MTCNN
                expects RGB, so this method handles the conversion —
                callers should always pass BGR for consistency with the
                rest of the pipeline.

        Returns:
            List of Detection objects, filtered by min_confidence and
            min_face_size. Empty list if no faces found — callers should
            handle that case (e.g. skip the photo, log it) rather than
            assume at least one face.
        """
        if image is None or image.size == 0:
            raise ValueError("FaceDetector.detect: received an empty image")

        self._ensure_loaded()

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_results = self._detector.detect_faces(rgb_image)

        detections = []
        for result in raw_results:
            x, y, w, h = result["box"]
            # MTCNN can return small negative coords for boxes near edges.
            x, y = max(0, x), max(0, y)

            confidence = result["confidence"]
            if confidence < self.min_confidence:
                continue
            if w < self.min_face_size or h < self.min_face_size:
                continue

            detections.append(
                Detection(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    keypoints=result.get("keypoints", {}),
                )
            )

        return detections
