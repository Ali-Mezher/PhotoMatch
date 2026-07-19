"""
Issue #3 — Intensity transformation (brightness & contrast).

Adjusts exposure so faces are detectable in both bright outdoor shots
and dim indoor venue shots, which is where MTCNN misses the most faces.
"""

import cv2
import numpy as np


def adjust_brightness_contrast(
    image: np.ndarray, alpha: float = 1.0, beta: float = 0.0
) -> np.ndarray:
    """
    Linear intensity transform: output = alpha * input + beta.

    Args:
        image: BGR or grayscale image, uint8.
        alpha: contrast gain. 1.0 = unchanged, >1.0 = more contrast.
        beta: brightness offset, added after scaling. Positive = brighter.

    Returns:
        Adjusted image, same shape/dtype, values clipped to [0, 255].
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def auto_contrast_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Adaptive contrast correction using CLAHE (Contrast Limited Adaptive
    Histogram Equalization), applied on the luminance channel only so
    colors don't shift. This is the main function to reach for on dim
    indoor event photos — it's local, so it doesn't blow out already-
    bright areas of the same image the way global equalization would.

    Args:
        image: BGR image, uint8.
        clip_limit: contrast limiting threshold. Higher = more aggressive.

    Returns:
        BGR image with adaptive contrast applied.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_equalized = clahe.apply(l_channel)

    merged = cv2.merge([l_equalized, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def estimate_exposure(image: np.ndarray) -> str:
    """
    Rough classification of an image as under/over/well exposed, based on
    mean luminance. Useful for deciding whether to run auto_contrast_clahe
    before detection, or logging why a photo was hard to detect faces in.

    Args:
        image: BGR image, uint8.

    Returns:
        One of "underexposed", "overexposed", "normal".
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = gray.mean()

    if mean_intensity < 80:
        return "underexposed"
    if mean_intensity > 180:
        return "overexposed"
    return "normal"
