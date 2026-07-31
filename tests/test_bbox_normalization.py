import pytest

from run_figure_refine import normalize_bbox_to_fraction


def test_normalize_bbox_keeps_fractional_coordinates():
    bbox = normalize_bbox_to_fraction((0.1, 0.2, 0.8, 0.9), (1000, 2000))
    assert bbox == pytest.approx((0.1, 0.2, 0.8, 0.9))


def test_normalize_bbox_converts_pixel_coordinates():
    bbox = normalize_bbox_to_fraction((100, 400, 800, 1600), (1000, 2000))
    assert bbox == pytest.approx((0.1, 0.2, 0.8, 0.8))


def test_normalize_bbox_converts_mixed_fraction_and_pixel_coordinates():
    bbox = normalize_bbox_to_fraction((0.081, 589.0, 0.912, 794.0), (1275, 1000))
    assert bbox == pytest.approx((0.081, 0.589, 0.912, 0.794))


def test_normalize_bbox_rejects_invalid_coordinates():
    with pytest.raises(ValueError, match="out-of-range"):
        normalize_bbox_to_fraction((0.9, 0.2, 0.1, 0.8), (1000, 2000))
