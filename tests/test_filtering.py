import numpy as np
import pytest
from src.preprocessing.filtering import denoise, sharpen, preprocess


def make_noisy_image(size=(100, 100)) -> np.ndarray:
    base = np.full((size[0], size[1], 3), 128, dtype=np.uint8)
    noise = np.random.randint(-30, 30, base.shape, dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_image(brightness: int, size=(100, 100)) -> np.ndarray:
    return np.full((size[0], size[1], 3), brightness, dtype=np.uint8)


def test_denoise_bilateral_output_shape():
    image = make_noisy_image()
    result = denoise(image, method="bilateral")
    assert result.shape == image.shape


def test_denoise_gaussian_output_shape():
    image = make_noisy_image()
    result = denoise(image, method="gaussian")
    assert result.shape == image.shape


def test_denoise_bilateral_reduces_noise():
    image = make_noisy_image()
    result = denoise(image, method="bilateral")
    assert np.std(result) < np.std(image)


def test_denoise_invalid_method_raises():
    image = make_image(128)
    with pytest.raises(ValueError):
        denoise(image, method="invalid")


def test_sharpen_output_shape_and_dtype():
    image = make_image(128)
    result = sharpen(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_sharpen_flat_image_unchanged():
    # A flat image has no edges so sharpening should not change it
    image = make_image(100)
    result = sharpen(image)
    np.testing.assert_array_equal(result, image)

def test_sharpen_output_shape_and_dtype():
    image = make_image(128)
    result = sharpen(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_sharpen_flat_image_unchanged():
    # A perfectly flat image has no edges — sharpening should not change it
    image = make_image(100)
    result = sharpen(image)
    np.testing.assert_array_equal(result, image)


def test_preprocess_output_shape_and_dtype():
    image = make_noisy_image()
    result = preprocess(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_preprocess_reduces_noise():
    image = make_noisy_image()
    result = preprocess(image)
    assert np.std(result) < np.std(image)