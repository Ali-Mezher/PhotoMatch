"""
Preprocessing pipeline — combines color/geometry, intensity, filtering,
and morphology/segmentation into one entry point that src/detection
calls before running face detection.
"""

import numpy as np

from .color_geometry import correct_white_balance, correct_orientation
from .intensity import auto_contrast_clahe, estimate_exposure
from .filtering import denoise, sharpen
from .morphology_segmentation import crop_face_region, clean_mask, segment_foreground

__all__ = [
    "correct_white_balance",
    "correct_orientation",
    "auto_contrast_clahe",
    "estimate_exposure",
    "denoise",
    "sharpen",
    "crop_face_region",
    "clean_mask",
    "segment_foreground",
    "preprocess_image",
]


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline applied to a raw event photo before it
    goes to the face detector. Order matters: correct color/exposure
    first (so the noise estimate downstream is meaningful), denoise
    before sharpening (so you don't sharpen noise), and skip auto-contrast
    when the photo is already well exposed.

    Args:
        image: raw BGR image as loaded by cv2.imread.

    Returns:
        Preprocessed BGR image, ready for src/detection.
    """
    if image is None or image.size == 0:
        raise ValueError("preprocess_image: received an empty image")

    result = correct_white_balance(image)
    result = correct_orientation(result)

    if estimate_exposure(result) != "normal":
        result = auto_contrast_clahe(result)

    result = denoise(result, strength=8)
    result = sharpen(result, amount=0.6)

    return result
