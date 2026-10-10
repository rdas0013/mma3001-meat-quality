"""Tests for meatqa.data using a small fake dataset built in a temp folder."""
import pytest

from meatqa import data


def make_dataset(root, files):
    """Create ``{(split, stem): label_text}`` as empty .jpg plus .txt files."""
    for (split, stem), label in files.items():
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        (root / split / "images" / f"{stem}.jpg").write_bytes(b"")
        (root / split / "labels" / f"{stem}.txt").write_text(label)


def test_source_photo_name_strips_roboflow_suffix():
    assert data.source_photo_name("a_jpg.rf.abc123.jpg") == "a_jpg"


def test_source_photo_name_without_suffix_is_unchanged():
    assert data.source_photo_name("plain.jpg") == "plain.jpg"


def test_read_label_classes_hand_counted(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("1 0.5 0.5 0.1 0.1\n3 0.2 0.2 0.1 0.1\n1 0.9 0.9 0.1 0.1\n")
    assert data.read_label_classes(f) == [1, 3, 1]


def test_read_label_classes_empty_file_means_no_boxes(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("")
    assert data.read_label_classes(f) == []


def test_read_label_classes_rejects_bad_line(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("1 0.5 0.5\n")
    with pytest.raises(ValueError):
        data.read_label_classes(f)


def test_read_label_classes_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.read_label_classes(tmp_path / "nope.txt")


def test_count_boxes_per_class(tmp_path):
    make_dataset(tmp_path, {
        ("train", "a"): "0 .5 .5 .1 .1\n1 .5 .5 .1 .1\n",
        ("train", "b"): "1 .5 .5 .1 .1\n",
    })
    counts = data.count_boxes_per_class(tmp_path, "train", ["loose", "error"])
    assert counts["loose"] == 1 and counts["error"] == 2


def test_unknown_split_raises(tmp_path):
    with pytest.raises(ValueError):
        data.list_images(tmp_path, "validation")


def test_filtered_train_list_removes_only_shared_names(tmp_path):
    make_dataset(tmp_path, {
        ("train", "shared_jpg.rf.t1"): "",
        ("train", "shared_jpg.rf.t2"): "",
        ("train", "own_jpg.rf.t3"): "",
        ("valid", "shared_jpg.rf.v1"): "",
        ("test", "other_jpg.rf.s1"): "",
    })
    kept, excluded = data.build_filtered_train_list(tmp_path)
    assert [p.stem for p in kept] == ["own_jpg.rf.t3"]
    assert len(excluded) == 2
    assert data.find_shared_sources(tmp_path) == {"shared_jpg": ["train", "valid"]}
