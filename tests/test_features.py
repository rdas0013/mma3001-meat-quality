"""Tests for meatqa.features using images whose answers are known by hand."""
import numpy as np
import pytest
from PIL import Image

from meatqa import features


def test_feature_vector_length(tmp_path):
    p = tmp_path / "x.jpg"
    Image.new("RGB", (720, 540), (200, 40, 40)).save(p)
    assert features.extract_features(p).shape == (features.N_FEATURES,)


def test_histograms_each_sum_to_one(tmp_path):
    p = tmp_path / "x.jpg"
    Image.new("RGB", (720, 540), (200, 40, 40)).save(p)
    f = features.extract_features(p)
    h, s = features.HUE_BINS, features.SAT_BINS
    assert f[:h].sum() == pytest.approx(1.0)
    assert f[h:h + s].sum() == pytest.approx(1.0)
    assert f[h + s:h + s + features.VAL_BINS].sum() == pytest.approx(1.0)


def test_flat_image_has_zero_edges(tmp_path):
    p = tmp_path / "x.png"  # PNG so there is no compression noise
    Image.new("RGB", (720, 540), (200, 40, 40)).save(p)
    f = features.extract_features(p)
    assert np.all(f[-features.GRID ** 2:] == 0)


def test_vertical_edge_gives_edges_only_where_expected(tmp_path):
    # Left half black, right half white: the edge sits at the centre column,
    # so with a 4x4 grid only the two middle columns of cells can see it.
    arr = np.zeros((540, 720, 3), dtype=np.uint8)
    arr[:, 360:] = 255
    p = tmp_path / "x.png"
    Image.fromarray(arr).save(p)
    cells = features.extract_features(p)[-features.GRID ** 2:].reshape(4, 4)
    assert np.all(cells[:, 0] == 0) and np.all(cells[:, 3] == 0)
    assert np.all(cells[:, 1:3].sum(axis=1) > 0)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        features.extract_features(tmp_path / "nope.jpg")
