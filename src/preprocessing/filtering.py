import cv2
import numpy as np


def denoise(image: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Remove noise from the image before face detection.

    'bilateral' — blurs noise while keeping edges sharp (best for faces).
    'gaussian'  — simpler, faster, but slightly softens edges too.
    """
    if method == "bilateral":
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'bilateral' or 'gaussian'.")


def sharpen(image: np.ndarray) -> np.ndarray:
    """Sharpen edges using a convolution kernel.

    The kernel boosts the center pixel and subtracts its neighbors,
    making edges more defined. Helps detection on slightly soft photos.
    """
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def preprocess(image: np.ndarray, denoise_method: str = "bilateral") -> np.ndarray:
    """Full filtering pipeline: denoise first, then sharpen.

    Order matters — sharpening after denoising avoids amplifying grain.
    """
    image = denoise(image, method=denoise_method)
    image = sharpen(image)
    return image
