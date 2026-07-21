"""
Tests for src/detection. Structural tests only — they check that our
wrapper classes behave correctly (lazy loading, filtering, dataclass
shape) without actually running MTCNN/deepface inference, since those
require the full model weights and are slow. Once real sample photos are
available in data/events/, add an integration test that runs the full
detect_and_embed() pipeline on one and sanity-checks the output.

Run with: pytest tests/test_detection.py -v
"""

import numpy as np
import pytest

from src.detection.face_detector import FaceDetector, Detection
from src.detection.embeddings import EmbeddingExtractor


class TestFaceDetector:
    def test_lazy_loading_does_not_load_model_on_init(self):
        detector = FaceDetector()
        assert detector._detector is None

    def test_detect_rejects_empty_image(self):
        detector = FaceDetector()
        with pytest.raises(ValueError):
            detector.detect(np.array([]))

    def test_detection_dataclass_shape(self):
        d = Detection(bbox=(10, 10, 50, 50), confidence=0.95, keypoints={})
        assert d.bbox == (10, 10, 50, 50)
        assert 0.0 <= d.confidence <= 1.0

    def test_detect_converts_to_rgb_and_filters_results(self):
        class FakeMTCNN:
            def __init__(self):
                self.image = None

            def detect_faces(self, image):
                self.image = image
                return [
                    {
                        "box": [-5, -3, 60, 70],
                        "confidence": 0.96,
                        "keypoints": {"nose": (20, 30)},
                    },
                    {"box": [10, 10, 60, 70], "confidence": 0.50},
                    {"box": [10, 10, 20, 20], "confidence": 0.99},
                ]

        detector = FaceDetector(min_confidence=0.90, min_face_size=40)
        detector._detector = FakeMTCNN()
        bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_image[0, 0] = [10, 20, 30]

        results = detector.detect(bgr_image)

        assert detector._detector.image[0, 0].tolist() == [30, 20, 10]
        assert len(results) == 1
        assert results[0].bbox == (0, 0, 60, 70)
        assert results[0].keypoints == {"nose": (20, 30)}


class TestEmbeddingExtractor:
    def test_lazy_loading_does_not_load_model_on_init(self):
        extractor = EmbeddingExtractor()
        assert extractor._deepface is None

    def test_get_embedding_rejects_empty_image(self):
        extractor = EmbeddingExtractor()
        with pytest.raises(ValueError):
            extractor.get_embedding(np.array([]))

    def test_get_embeddings_batch_skips_failures(self, monkeypatch):
        extractor = EmbeddingExtractor()

        def fake_get_embedding(face_image):
            if face_image.sum() == 0:
                raise ValueError("simulated bad crop")
            return np.ones(512, dtype=np.float32)

        monkeypatch.setattr(extractor, "get_embedding", fake_get_embedding)

        good_face = np.ones((100, 100, 3), dtype=np.uint8)
        bad_face = np.zeros((100, 100, 3), dtype=np.uint8)

        results = extractor.get_embeddings_batch([good_face, bad_face, good_face])
        assert len(results) == 2  # bad_face's ValueError should be swallowed

    def test_get_embedding_normalizes_expected_vector(self):
        class FakeDeepFace:
            @staticmethod
            def represent(**kwargs):
                return [{"embedding": [2.0] * 512}]

        extractor = EmbeddingExtractor()
        extractor._deepface = FakeDeepFace
        embedding = extractor.get_embedding(np.ones((50, 50, 3), dtype=np.uint8))

        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32
        assert np.linalg.norm(embedding) == pytest.approx(1.0)

    def test_get_embedding_rejects_wrong_vector_dimension(self):
        class FakeDeepFace:
            @staticmethod
            def represent(**kwargs):
                return [{"embedding": [1.0] * 128}]

        extractor = EmbeddingExtractor()
        extractor._deepface = FakeDeepFace

        with pytest.raises(RuntimeError, match=r"expected \(512,\)"):
            extractor.get_embedding(np.ones((50, 50, 3), dtype=np.uint8))


# --- Manual integration check (not run by pytest automatically) ---------
#
# Once you have a real sample photo in data/events/test_event/raw/:
#
#   import cv2
#   from src.preprocessing import preprocess_image
#   from src.detection import detect_and_embed
#
#   img = cv2.imread("data/events/test_event/raw/sample.jpg")
#   img = preprocess_image(img)
#   faces = detect_and_embed(img)
#   print(f"Found {len(faces)} faces")
#   for f in faces:
#       print(f.bbox, f.confidence, f.embedding.shape)
