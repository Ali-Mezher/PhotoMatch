import numpy as np
import pytest
from src.preprocessing.color_geometry import normalize_white_balance, correct_geometry, preprocess


def make_image(r, g, b, size=(100, 100)):
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    image[:, :] = [b, g, r]  # BGR
    return image


def test_white_balance_neutral_image_unchanged():
    # A perfectly gray image should be unchanged
    image = make_image(128, 128, 128)
    result = normalize_white_balance(image)
    np.testing.assert_array_equal(result, image)


def test_white_balance_removes_color_cast():
    # Strong red cast: after normalization all channels should be closer together
    image = make_image(200, 100, 100)
    result = normalize_white_balance(image)
    r = np.mean(result[:, :, 2])
    g = np.mean(result[:, :, 1])
    b = np.mean(result[:, :, 0])
    # Channels should be much closer than the original 200 / 100 / 100 split
    assert max(r, g, b) - min(r, g, b) < 10


def test_white_balance_output_dtype():
    image = make_image(150, 100, 80)
    result = normalize_white_balance(image)
    assert result.dtype == np.uint8


def test_white_balance_no_overflow():
    image = make_image(255, 50, 50)
    result = normalize_white_balance(image)
    assert result.max() <= 255
    assert result.min() >= 0


def test_correct_geometry_output_shape():
    image = make_image(128, 128, 128, size=(480, 640))
    result = correct_geometry(image)
    assert result.shape == image.shape


def test_correct_geometry_zero_distortion_unchanged():
    # k1=0, k2=0 means no distortion — output should equal input
    image = make_image(100, 150, 200, size=(200, 200))
    result = correct_geometry(image, k1=0.0, k2=0.0)
    np.testing.assert_array_equal(result, image)


def test_preprocess_output_shape_and_dtype():
    image = make_image(180, 120, 90, size=(300, 400))
    result = preprocess(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
