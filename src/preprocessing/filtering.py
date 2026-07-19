"""
Issue #4 — Convolution filtering (denoise & sharpen).

Cleans up motion blur and sensor noise from event photos (low light,
fast-moving subjects at weddings/graduations) before face detection runs.
"""

import cv2
import numpy as np


def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Remove sensor noise using Non-Local Means denoising, which preserves
    edges (like facial features) better than a simple Gaussian blur.

    Args:
        image: BGR image, uint8.
        strength: filter strength — higher removes more noise but can
            also soften fine detail. 10 is a reasonable default for
            typical event-photo ISO noise.

    Returns:
        Denoised BGR image.
    """
    return cv2.fastNlMeansDenoisingColored(
        image, None, h=strength, hColor=strength, templateWindowSize=7, searchWindowSize=21
    )


def sharpen(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """
    Sharpen an image via unsharp masking: blur the image, then push the
    original away from the blurred version to boost edge contrast. Helps
    recover detail lost to slight motion blur or lens softness.

    Args:
        image: BGR image, uint8.
        amount: sharpening strength. 0 = no-op, ~1.0 = moderate, >2.0 = aggressive.

    Returns:
        Sharpened BGR image.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened


def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Simple Gaussian smoothing — mostly useful as a preprocessing step
    before edge detection / segmentation, not for the main photo cleanup
    path (use denoise() for that).

    Args:
        image: BGR or grayscale image.
        kernel_size: must be odd; larger = more blur.

    Returns:
        Blurred image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
