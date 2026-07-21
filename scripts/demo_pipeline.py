"""
Manual smoke-test for the detect -> preprocess -> embed pipeline on one
real photo. Not part of the automated test suite (needs mtcnn + deepface
installed, and a real image) — run by hand while developing:

    python scripts/demo_pipeline.py path/to/photo.jpg

This is meant to be replaced by src/indexing's real event-indexing script
once that module exists — it's here so Ali and Mahmood can sanity-check
their modules work together before Week 3 (indexing) starts.
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import preprocess_image
from src.detection import detect_and_embed


def main(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    print(f"Loaded {image_path} — shape {image.shape}")

    cleaned = preprocess_image(image)
    print("Preprocessing done.")

    faces = detect_and_embed(cleaned)
    print(f"Detected {len(faces)} face(s):")
    for i, face in enumerate(faces):
        print(
            f"  [{i}] bbox={face.bbox} "
            f"confidence={face.confidence:.3f} "
            f"embedding_shape={face.embedding.shape}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/demo_pipeline.py path/to/photo.jpg")
        sys.exit(1)

    main(sys.argv[1])
