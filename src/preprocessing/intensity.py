import cv2
import numpy as np


def adjust_brightness_contrast(
    image: np.ndarray, alpha: float = 1.0, beta: float = 0
) -> np.ndarray:
    """Scale contrast by alpha and shift brightness by beta.

    alpha > 1 increases contrast, alpha < 1 reduces it.
    beta > 0 brightens, beta < 0 darkens.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Enhances local contrast without over-amplifying noise — better than
    global histogram equalization for mixed indoor/outdoor event photos.
    Applied to the L channel in LAB color space to avoid hue shifts.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def is_low_light(image: np.ndarray, threshold: int = 80) -> bool:
    """Return True if the image mean brightness is below the threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < threshold


def preprocess(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Intensity transformation pipeline.

    Applies CLAHE for local contrast enhancement. For dim images (mean
    brightness < 80), also applies a brightness boost to ensure faces
    are detectable in low-light indoor event shots.
    """
    if is_low_light(image):
        image = adjust_brightness_contrast(image, alpha=1.2, beta=30)
    image = apply_clahe(image, clip_limit=clip_limit, tile_size=tile_size)
    return image
