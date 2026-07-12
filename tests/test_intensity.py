import cv2
import numpy as np
import pytest
from src.preprocessing.intensity import (
    adjust_brightness_contrast,
    apply_clahe,
    is_low_light,
    preprocess,
)


def make_image(brightness: int, size=(100, 100)) -> np.ndarray:
    return np.full((size[0], size[1], 3), brightness, dtype=np.uint8)


def test_brightness_contrast_increases_brightness():
    image = make_image(100)
    result = adjust_brightness_contrast(image, alpha=1.0, beta=50)
    assert np.mean(result) > np.mean(image)


def test_brightness_contrast_no_overflow():
    image = make_image(255)
    result = adjust_brightness_contrast(image, alpha=2.0, beta=100)
    assert result.max() <= 255


def test_brightness_contrast_output_dtype():
    image = make_image(120)
    result = adjust_brightness_contrast(image, alpha=1.5, beta=10)
    assert result.dtype == np.uint8


def test_clahe_output_shape_and_dtype():
    image = make_image(80)
    result = apply_clahe(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_clahe_brightens_dark_image():
    image = make_image(40)
    result = apply_clahe(image)
    assert np.mean(result) >= np.mean(image)


def test_is_low_light_dark_image():
    image = make_image(50)
    assert is_low_light(image) is True


def test_is_low_light_bright_image():
    image = make_image(180)
    assert is_low_light(image) is False


def test_preprocess_output_shape_and_dtype():
    image = make_image(100)
    result = preprocess(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_preprocess_boosts_dark_image():
    dark = make_image(40)
    result = preprocess(dark)
    assert np.mean(result) > np.mean(dark)
