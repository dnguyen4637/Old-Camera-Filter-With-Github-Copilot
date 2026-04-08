"""
test_old_camera_filter.py
--------------------------
Unit tests for the retro CCD camera filter effects.

Run with:
    python -m pytest test_old_camera_filter.py -v
"""

import numpy as np
import pytest

from old_camera_filter import (
    apply_ccd_filter,
    apply_film_grain,
    apply_halation,
    color_grade,
    crush_blacks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid(h: int, w: int, b: float, g: float, r: float) -> np.ndarray:
    """Return a float32 BGR image filled with a single colour."""
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def _random(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# apply_halation
# ---------------------------------------------------------------------------

class TestApplyHalation:
    def test_output_shape_unchanged(self):
        img = _random()
        out = apply_halation(img)
        assert out.shape == img.shape

    def test_output_dtype_float32(self):
        assert apply_halation(_random()).dtype == np.float32

    def test_output_clipped_to_unit_range(self):
        img = _random()
        out = apply_halation(img)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_dark_image_unchanged(self):
        """Pixels well below the threshold contribute no bloom — output ≈ input."""
        img = _solid(32, 32, 0.1, 0.1, 0.1)
        out = apply_halation(img, threshold=0.90, intensity=1.0)
        np.testing.assert_allclose(out, img, atol=1e-4)

    def test_bright_image_gets_brighter(self):
        """A very bright image should have bloom added, making it at least as bright."""
        img = _solid(32, 32, 0.95, 0.95, 0.95)
        out = apply_halation(img, threshold=0.90, intensity=0.5)
        # Output is clipped to 1, but it must not be darker than the input.
        assert out.mean() >= img.mean() - 1e-6

    def test_zero_intensity_no_change(self):
        img = _random()
        out = apply_halation(img, intensity=0.0)
        np.testing.assert_allclose(out, img, atol=1e-6)


# ---------------------------------------------------------------------------
# crush_blacks
# ---------------------------------------------------------------------------

class TestCrushBlacks:
    def test_output_shape_unchanged(self):
        assert crush_blacks(_random()).shape == _random().shape

    def test_output_clipped_to_unit_range(self):
        out = crush_blacks(_random())
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_shadows_are_darker(self):
        """Crushed output should be ≤ input in the shadow region."""
        img = _solid(32, 32, 0.05, 0.05, 0.05)   # deep shadow
        out = crush_blacks(img, toe_start=0.10, toe_strength=1.6)
        assert out.mean() <= img.mean() + 1e-6

    def test_bright_pixels_largely_preserved(self):
        """Bright pixels (above toe_start) should not be strongly darkened."""
        img = _solid(32, 32, 0.8, 0.8, 0.8)
        out = crush_blacks(img, toe_start=0.10)
        # Allow tiny floating-point drift; bright areas should be close to input.
        np.testing.assert_allclose(out, img, atol=1e-5)


# ---------------------------------------------------------------------------
# apply_film_grain
# ---------------------------------------------------------------------------

class TestApplyFilmGrain:
    def test_output_shape_unchanged(self):
        assert apply_film_grain(_random()).shape == _random().shape

    def test_output_clipped_to_unit_range(self):
        out = apply_film_grain(_random())
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_zero_intensity_no_change(self):
        img = _random()
        out = apply_film_grain(img, intensity=0.0)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_grain_is_luminance_only(self):
        """
        Because the same scalar grain map is added to all three channels,
        the per-pixel channel *differences* must be the same before and after
        (tested on a mid-grey image so np.clip cannot alter individual channels
        differently near the 0/1 boundaries).
        """
        # Mid-grey (0.5) image: grain of ±0.05 keeps every value in [0.45, 0.55]
        img = _solid(32, 32, 0.5, 0.5, 0.5)
        # Give channels distinct base values that won't clip
        img[:, :, 0] = 0.45
        img[:, :, 1] = 0.50
        img[:, :, 2] = 0.55
        out = apply_film_grain(img, intensity=0.03, seed=42)
        delta_before = img[:, :, 0] - img[:, :, 1]
        delta_after  = out[:, :, 0] - out[:, :, 1]
        np.testing.assert_allclose(delta_before, delta_after, atol=1e-5)

    def test_reproducible_with_seed(self):
        img = _random()
        a = apply_film_grain(img, seed=99)
        b = apply_film_grain(img, seed=99)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        img = _random()
        a = apply_film_grain(img, seed=1)
        b = apply_film_grain(img, seed=2)
        assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# color_grade
# ---------------------------------------------------------------------------

class TestColorGrade:
    def test_output_shape_unchanged(self):
        assert color_grade(_random()).shape == _random().shape

    def test_output_clipped_to_unit_range(self):
        out = color_grade(_random())
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_zero_tints_no_change(self):
        img = _random()
        out = color_grade(img, midtone_warmth=0.0, shadow_teal=0.0)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_midtone_warmth_increases_red(self):
        """Mid-tone region should see a net increase in the R channel."""
        # A mid-grey is squarely in the mid-tone zone
        img = _solid(32, 32, 0.5, 0.5, 0.5)
        out = color_grade(img, midtone_warmth=0.10, shadow_teal=0.0)
        assert out[:, :, 2].mean() >= img[:, :, 2].mean() - 1e-6  # R lifted

    def test_shadow_teal_increases_blue(self):
        """Deep shadows should see a net increase in the B channel."""
        img = _solid(32, 32, 0.02, 0.02, 0.02)
        out = color_grade(img, midtone_warmth=0.0, shadow_teal=0.10)
        assert out[:, :, 0].mean() >= img[:, :, 0].mean() - 1e-6  # B lifted


# ---------------------------------------------------------------------------
# apply_ccd_filter (integration)
# ---------------------------------------------------------------------------

class TestApplyCCDFilter:
    def test_accepts_uint8_returns_uint8(self):
        img_u8 = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
        out = apply_ccd_filter(img_u8)
        assert out.dtype == np.uint8

    def test_output_shape_unchanged(self):
        img_u8 = (np.random.default_rng(1).random((48, 72, 3)) * 255).astype(np.uint8)
        out = apply_ccd_filter(img_u8)
        assert out.shape == img_u8.shape

    def test_output_values_in_uint8_range(self):
        img_u8 = (np.random.default_rng(2).random((64, 64, 3)) * 255).astype(np.uint8)
        out = apply_ccd_filter(img_u8)
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_non_trivial_output(self):
        """The filter must actually change the image (not a no-op)."""
        img_u8 = (np.random.default_rng(3).random((64, 64, 3)) * 255).astype(np.uint8)
        out = apply_ccd_filter(img_u8)
        assert not np.array_equal(out, img_u8)
