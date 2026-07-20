"""
Preprocessing pipeline — combines color/geometry, intensity, filtering,
and morphology/segmentation into one entry point that src/detection
calls before running face detection.
"""

import numpy as np

from .color_geometry import normalize_white_balance, correct_geometry
from .intensity import apply_clahe, is_low_light
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


def correct_white_balance(image: np.ndarray) -> np.ndarray:
    """Validate and apply the project's gray-world white balancing."""
    if image is None or image.size == 0:
        raise ValueError("correct_white_balance: received an empty image")
    return normalize_white_balance(image)


def correct_orientation(image: np.ndarray) -> np.ndarray:
    """Apply the existing geometry correction used by the preprocessing stage."""
    return correct_geometry(image)


def auto_contrast_clahe(image: np.ndarray) -> np.ndarray:
    """Compatibility entry point for CLAHE contrast enhancement."""
    return apply_clahe(image)


def estimate_exposure(image: np.ndarray) -> str:
    """Classify exposure for the integrated pipeline's CLAHE decision."""
    return "underexposed" if is_low_light(image) else "normal"


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

    result = denoise(result)
    result = sharpen(result)

    return result
