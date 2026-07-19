"""
Issue #2 — Color & geometry correction.

Normalizes color balance and corrects orientation/geometric distortion
across event photos shot in mixed lighting (indoor venue lighting,
outdoor daylight, flash) and at inconsistent angles.
"""

import cv2
import numpy as np


def correct_white_balance(image: np.ndarray) -> np.ndarray:
    """
    Normalize color balance using the Gray World assumption: scale each
    color channel so the average pixel value is neutral gray. Cheap and
    effective for mixed-lighting event photos without needing a reference
    card.

    Args:
        image: BGR image (as loaded by cv2.imread), dtype uint8.

    Returns:
        BGR image with white balance corrected, same shape/dtype.
    """
    if image is None or image.size == 0:
        raise ValueError("correct_white_balance: received an empty image")

    result = image.astype(np.float32)
    b, g, r = cv2.split(result)

    b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
    gray_mean = (b_mean + g_mean + r_mean) / 3.0

    # Avoid divide-by-zero on pathological (near-black) images.
    b = b * (gray_mean / max(b_mean, 1e-6))
    g = g * (gray_mean / max(g_mean, 1e-6))
    r = r * (gray_mean / max(r_mean, 1e-6))

    balanced = cv2.merge([b, g, r])
    return np.clip(balanced, 0, 255).astype(np.uint8)


def correct_orientation(image: np.ndarray) -> np.ndarray:
    """
    Auto-rotate an image to right-side-up based on EXIF-independent content
    analysis is out of scope for v1 — most event photos already carry
    correct EXIF orientation that cv2.imread/PIL handle on load. This is a
    placeholder for manual rotation correction if a batch of photos comes
    in sideways.

    Args:
        image: BGR image.

    Returns:
        The image, rotated 0 degrees (no-op) for now.

    Note:
        If a future event's photos consistently need rotation, call
        rotate_image(image, angle) below instead of extending this
        function — keep detection heuristics and manual fixes separate.
    """
    return image


def rotate_image(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Rotate an image by a fixed angle around its center, expanding the
    canvas so no content is cropped.

    Args:
        image: BGR image.
        angle_degrees: positive = counter-clockwise.

    Returns:
        Rotated BGR image.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    # Compute the new bounding box so rotated corners aren't clipped.
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
