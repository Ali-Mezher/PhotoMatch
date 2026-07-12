import cv2
import numpy as np


def normalize_white_balance(image: np.ndarray) -> np.ndarray:
    """Gray world white balance: scales each channel so its mean equals the
    overall gray mean. Corrects color casts from mixed indoor/outdoor lighting."""
    result = image.astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / avg_b), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / avg_g), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / avg_r), 0, 255)
    return result.astype(np.uint8)


def correct_geometry(image: np.ndarray, k1: float = -0.1, k2: float = 0.0) -> np.ndarray:
    """Correct radial lens distortion (barrel/pincushion).

    k1 < 0 corrects barrel distortion (common in wide-angle event lenses).
    k1 > 0 corrects pincushion distortion.
    Uses an estimated camera matrix when calibration data is unavailable.
    """
    h, w = image.shape[:2]
    f = max(w, h)
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array(
        [[f, 0, cx],
         [0, f, cy],
         [0, 0,  1]], dtype=np.float32
    )
    dist_coeffs = np.array([k1, k2, 0.0, 0.0], dtype=np.float32)
    return cv2.undistort(image, camera_matrix, dist_coeffs)


def preprocess(image: np.ndarray, k1: float = -0.1, k2: float = 0.0) -> np.ndarray:
    """Full color & geometry correction pipeline.

    Applies white balance normalization followed by radial distortion correction.
    Input and output are BGR uint8 numpy arrays (OpenCV format).
    """
    image = normalize_white_balance(image)
    image = correct_geometry(image, k1=k1, k2=k2)
    return image
