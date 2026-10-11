"""Check the fast feature extractor matches the original exactly."""
import numpy as np
import pytest

from meatqa import features
from meatqa.data import list_images
from pathlib import Path


def test_fast_matches_original_on_synthetic_image(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(540, 720, 3), dtype=np.uint8)
    p = tmp_path / "noise.png"
    Image.fromarray(arr).save(p)
    assert np.allclose(features.extract_features(p), features.extract_features_fast(p), atol=1e-12)


@pytest.mark.skipif(not Path("data/valid/images").is_dir(), reason="dataset not present")
def test_fast_matches_original_on_real_images():
    for p in list_images(Path("data"), "valid")[:10]:
        assert np.allclose(features.extract_features(p), features.extract_features_fast(p), atol=1e-12)


def test_fast_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        features.extract_features_fast(tmp_path / "nope.jpg")
