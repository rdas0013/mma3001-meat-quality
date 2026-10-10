"""Tests for meatqa.targets using answers worked out by hand."""
import pytest

from meatqa import targets


def test_area_fraction_hand_computed(tmp_path):
    f = tmp_path / "a.txt"
    # 0.5*0.2 = 0.10 and 0.1*0.1 = 0.01, so the total is 0.11
    f.write_text("1 0.5 0.5 0.5 0.2\n3 0.2 0.2 0.1 0.1\n")
    assert targets.defect_area_fraction(f) == pytest.approx(0.11)


def test_largest_defect_hand_computed(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("1 0.5 0.5 0.5 0.2\n3 0.2 0.2 0.1 0.1\n")
    assert targets.largest_defect_fraction(f) == pytest.approx(0.10)


def test_empty_label_file_gives_zero(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("")
    assert targets.defect_area_fraction(f) == 0.0
    assert targets.largest_defect_fraction(f) == 0.0


def test_bad_line_raises(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("1 0.5 0.5\n")
    with pytest.raises(ValueError):
        targets.read_boxes(f)
