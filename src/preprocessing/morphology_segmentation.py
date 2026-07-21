"""
Issue #5 — Morphological operations & segmentation.

Cleans up detection masks (remove noise, fill gaps) and isolates face
regions from cluttered group-photo backgrounds prior to embedding
extraction. These functions operate on binary masks and on cropped
regions, not full photos — they run *after* src/detection has produced
bounding boxes.
"""

import cv2
import numpy as np


def clean_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Clean up a binary mask with morphological opening (erosion then
    dilation) to remove small noise specks, followed by closing (dilation
    then erosion) to fill small gaps.

    Args:
        mask: single-channel binary mask (0/255), uint8.
        kernel_size: size of the structuring element. Larger removes more
            noise but can erode small real features — keep this small.

    Returns:
        Cleaned binary mask, same shape.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def crop_face_region(
    image: np.ndarray, bbox: tuple[int, int, int, int], margin: float = 0.2
) -> np.ndarray:
    """
    Extract a face region from a full photo given a detector bounding box,
    with a margin added so the embedding model gets some context around
    the face (hairline, chin) rather than a tight crop.

    Args:
        image: full BGR photo.
        bbox: (x, y, width, height) in pixels, as returned by the detector.
        margin: fraction of the box size to pad on each side. 0.2 = 20%.

    Returns:
        Cropped BGR face region. Smaller than requested if the margin
        would go outside the image bounds — it's clipped, not padded
        with black, since embedding models handle real content better
        than padding.
    """
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]

    pad_x = int(w * margin)
    pad_y = int(h * margin)

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(img_w, x + w + pad_x)
    y1 = min(img_h, y + h + pad_y)

    return image[y0:y1, x0:x1]


def segment_foreground(image: np.ndarray) -> np.ndarray:
    """
    Rough foreground/background segmentation using Otsu thresholding on
    the saturation channel — useful for isolating a subject from a
    cluttered background (e.g. a crowd) before running detection on a
    tighter region. This is a classical-CV fallback, not a replacement
    for the detector itself.

    Args:
        image: BGR image.

    Returns:
        Binary mask (0/255) where 255 marks likely-foreground pixels.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return clean_mask(mask, kernel_size=5)
