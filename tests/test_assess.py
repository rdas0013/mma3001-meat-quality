"""Tests for meatqa.assess using a fake model whose output is known exactly."""
import numpy as np
import pytest
from PIL import Image

from meatqa import assess


class FakeScaler:
    """Identity scaler."""

    def transform(self, X):
        return X


class FakeModel:
    """Model that always predicts the same value."""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.array([self.value])


def bundle(value):
    return {"scaler": FakeScaler(), "model": FakeModel(value)}


@pytest.fixture
def good_image(tmp_path):
    p = tmp_path / "ok.png"
    Image.new("RGB", assess.EXPECTED_SIZE, (200, 80, 80)).save(p)
    return p


def test_prediction_above_threshold_fails(good_image):
    r = assess.assess_image(good_image, bundle(0.10), threshold=0.04)
    assert r["decision"] == "fail" and r["predicted_area_fraction"] == pytest.approx(0.10)


def test_prediction_below_threshold_passes(good_image):
    assert assess.assess_image(good_image, bundle(0.02), threshold=0.04)["decision"] == "pass"


def test_negative_prediction_is_clipped_to_zero(good_image):
    r = assess.assess_image(good_image, bundle(-0.01))
    assert r["predicted_area_fraction"] == 0.0 and r["decision"] == "pass"


def test_result_has_expected_keys(good_image):
    r = assess.assess_image(good_image, bundle(0.05))
    assert set(r) == {"image", "predicted_area_fraction", "decision", "threshold"}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        assess.assess_image(tmp_path / "nope.png", bundle(0.0))


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "x.bmp"
    Image.new("RGB", assess.EXPECTED_SIZE).save(p)
    with pytest.raises(ValueError):
        assess.assess_image(p, bundle(0.0))


def test_corrupt_file_raises(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"this is not an image")
    with pytest.raises(ValueError):
        assess.assess_image(p, bundle(0.0))


def test_wrong_size_raises(tmp_path):
    p = tmp_path / "small.png"
    Image.new("RGB", (100, 100)).save(p)
    with pytest.raises(ValueError):
        assess.assess_image(p, bundle(0.0))
