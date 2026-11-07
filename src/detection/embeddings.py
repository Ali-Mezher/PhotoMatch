"""
Issue #7 — Face embeddings (FaceNet / ArcFace).

Wraps `deepface` to turn a cropped face image into a fixed-length vector
that src/indexing stores and src/matching compares against. Kept behind
the same kind of thin wrapper as face_detector.py, for the same reason:
swapping the embedding model later should mean changing this file only.
"""

import os
import sys

import numpy as np

from config import EMBEDDING_MODEL, EMBEDDING_DIM, MODELS_DIR
from .device import configure_compute_device, run_tensorflow_operation


class EmbeddingExtractor:
    """
    Thin wrapper around deepface's `represent()` for generating face
    embeddings. Lazily imports deepface (heavy, TF-backed) on first use.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._deepface = None  # loaded lazily, see _ensure_loaded()

    def _ensure_loaded(self):
        if self._deepface is None:
            # DeepFace stores weights under ``DEEPFACE_HOME/.deepface``.
            # Point it at PhotoMatch's persistent, gitignored model directory
            # before importing the library. This makes Docker's model volume
            # useful and prevents a container replacement from losing weights.
            os.environ.setdefault("DEEPFACE_HOME", str(MODELS_DIR.resolve()))

            # DeepFace logs emoji while downloading model weights. Windows
            # PowerShell may expose a cp1252 stream, which otherwise raises a
            # UnicodeEncodeError before the download even starts.
            for stream in (sys.stdout, sys.stderr):
                reconfigure = getattr(stream, "reconfigure", None)
                if reconfigure is not None:
                    reconfigure(errors="replace")

            configure_compute_device()
            try:
                from deepface import DeepFace  # deferred import — heavy, TF-backed
            except (ImportError, ValueError) as exc:
                raise RuntimeError(
                    "DeepFace could not initialize. Install the dependencies in "
                    "requirements.txt (including a tf-keras version compatible "
                    "with TensorFlow)."
                ) from exc

            self._deepface = DeepFace

    def prepare_model(self) -> None:
        """Download or load the configured model before offline operation."""
        self._ensure_loaded()
        run_tensorflow_operation(
            lambda: self._deepface.build_model(self.model_name)
        )

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate an embedding vector for a single cropped face.

        Args:
            face_image: BGR image of a cropped face region, uint8 — use
                src.preprocessing.crop_face_region() to produce this from
                a full photo + detector bounding box.

        Returns:
            1-D float32 numpy array of length EMBEDDING_DIM
            (512 for Facenet512), L2-normalized so cosine similarity
            reduces to a plain dot product downstream in src/matching.

        Raises:
            ValueError: if the face image is empty or deepface fails to
                find a face in it (e.g. a bad crop).
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("get_embedding: received an empty face image")

        self._ensure_loaded()

        # detector_backend="skip" because we already ran our own MTCNN
        # detector upstream — no need for deepface to re-detect.
        result = run_tensorflow_operation(
            lambda: self._deepface.represent(
                img_path=face_image,
                model_name=self.model_name,
                detector_backend="skip",
                enforce_detection=False,
            )
        )

        embedding = np.array(result[0]["embedding"], dtype=np.float32)

        if embedding.shape != (EMBEDDING_DIM,):
            raise RuntimeError(
                f"{self.model_name} returned embedding shape {embedding.shape}; "
                f"expected ({EMBEDDING_DIM},)"
            )

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def get_embeddings_batch(self, face_images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Convenience wrapper for embedding multiple faces (e.g. all faces
        detected in one group photo). Not parallelized in v1 — deepface
        model reuse across calls already avoids reloading weights each
        time since _deepface is cached on self.

        Args:
            face_images: list of cropped BGR face images.

        Returns:
            List of embedding vectors, same order as input. A face that
            fails to embed is skipped (not returned as None) — callers
            should compare list length against input if they need to
            know which ones failed.
        """
        embeddings = []
        for face_image in face_images:
            try:
                embeddings.append(self.get_embedding(face_image))
            except ValueError:
                continue
        return embeddings
