"""
Tests for src/preprocessing. Uses synthetic numpy arrays instead of real
event photos — real photos are gitignored and shouldn't be a test
dependency anyway. Run with: pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pytest

from src.preprocessing import (
    preprocess_image,
    correct_white_balance,
    auto_contrast_clahe,
    estimate_exposure,
    denoise,
    sharpen,
    crop_face_region,
    clean_mask,
    segment_foreground,
)


@pytest.fixture
def sample_image():
    """A small synthetic BGR image with some noise — stands in for a photo."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(200, 300, 3), dtype=np.uint8)


@pytest.fixture
def dim_image():
    """A synthetic underexposed image, to exercise the CLAHE branch."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 60, size=(200, 300, 3), dtype=np.uint8)


class TestColorGeometry:
    def test_white_balance_preserves_shape_and_dtype(self, sample_image):
        result = correct_white_balance(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_white_balance_rejects_empty_image(self):
        with pytest.raises(ValueError):
            correct_white_balance(np.array([]))


class TestIntensity:
    def test_estimate_exposure_labels_dim_image(self, dim_image):
        assert estimate_exposure(dim_image) == "underexposed"

    def test_clahe_preserves_shape(self, sample_image):
        result = auto_contrast_clahe(sample_image)
        assert result.shape == sample_image.shape


class TestFiltering:
    def test_denoise_preserves_shape(self, sample_image):
        result = denoise(sample_image)
        assert result.shape == sample_image.shape

    def test_sharpen_preserves_shape(self, sample_image):
        result = sharpen(sample_image)
        assert result.shape == sample_image.shape


class TestMorphologySegmentation:
    def test_crop_face_region_within_bounds(self, sample_image):
        crop = crop_face_region(sample_image, bbox=(50, 50, 40, 40), margin=0.2)
        assert crop.shape[0] > 0 and crop.shape[1] > 0
        assert crop.shape[0] <= sample_image.shape[0]
        assert crop.shape[1] <= sample_image.shape[1]

    def test_crop_face_region_clips_at_edges(self, sample_image):
        # bbox near the bottom-right corner — margin would go out of bounds
        h, w = sample_image.shape[:2]
        crop = crop_face_region(sample_image, bbox=(w - 30, h - 30, 40, 40), margin=0.5)
        assert crop.shape[0] > 0 and crop.shape[1] > 0

    def test_segment_foreground_returns_binary_mask(self, sample_image):
        mask = segment_foreground(sample_image)
        assert set(np.unique(mask)).issubset({0, 255})


class TestPipeline:
    def test_preprocess_image_end_to_end(self, sample_image):
        result = preprocess_image(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_preprocess_image_rejects_none(self):
        with pytest.raises(ValueError):
            preprocess_image(None)
