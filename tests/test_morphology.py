import numpy as np
import pytest
from src.preprocessing.morphology import (
    erode, dilate, opening, closing, skin_mask, clean_mask, preprocess
)


def make_mask(size=100) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    return mask


def make_skin_image(size=(100, 100)) -> np.ndarray:
    # BGR approximation of a skin tone
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    image[:, :] = [90, 130, 200]
    return image


def make_bgr_image(size=(100, 100)) -> np.ndarray:
    return np.full((size[0], size[1], 3), 128, dtype=np.uint8)


def test_erode_shrinks_white_region():
    mask = make_mask()
    result = erode(mask, kernel_size=5, iterations=2)
    assert np.sum(result) < np.sum(mask)


def test_dilate_expands_white_region():
    mask = make_mask()
    result = dilate(mask, kernel_size=5, iterations=2)
    assert np.sum(result) > np.sum(mask)


def test_opening_output_shape():
    mask = make_mask()
    result = opening(mask)
    assert result.shape == mask.shape


def test_closing_output_shape():
    mask = make_mask()
    result = closing(mask)
    assert result.shape == mask.shape


def test_opening_removes_small_noise():
    mask = make_mask()
    # Add a small noise blob
    mask[5:8, 5:8] = 255
    result = opening(mask, kernel_size=7)
    # The small blob should be gone
    assert np.sum(result[5:8, 5:8]) == 0


def test_skin_mask_output_shape_and_dtype():
    image = make_skin_image()
    result = skin_mask(image)
    assert result.shape == (100, 100)
    assert result.dtype == np.uint8


def test_clean_mask_output_shape():
    mask = make_mask()
    result = clean_mask(mask)
    assert result.shape == mask.shape


def test_preprocess_output_shape_and_dtype():
    image = make_bgr_image()
    result = preprocess(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8