import cv2
import numpy as np


def erode(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Shrink white regions in a binary mask — removes small noise blobs."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask, kernel, iterations=iterations)


def dilate(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Expand white regions in a binary mask — fills small gaps."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)


def opening(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Erode then dilate — removes small noise while preserving shape."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def closing(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Dilate then erode — fills small holes inside a region."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def skin_mask(image: np.ndarray) -> np.ndarray:
    """Build a binary mask of skin-colored regions using HSV color space.

    Skin tone has a consistent hue range in HSV regardless of lighting,
    making it more reliable than RGB for detecting face regions.
    Returns a binary mask: 255 where skin is detected, 0 elsewhere.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Remove noise then fill gaps in a binary mask.

    Opening removes small isolated blobs, closing fills holes inside
    detected regions. Run in sequence for the cleanest result.
    """
    mask = opening(mask, kernel_size=5)
    mask = closing(mask, kernel_size=11)
    return mask


def preprocess(image: np.ndarray) -> np.ndarray:
    """Generate a cleaned skin-region mask from an event photo.

    Returns the original image with non-skin regions darkened,
    helping the face detector focus on likely face areas.
    """
    mask = skin_mask(image)
    mask = clean_mask(mask)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(image, mask_3ch)