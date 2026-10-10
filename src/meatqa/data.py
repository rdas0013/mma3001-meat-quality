"""Dataset utilities for the pork rasher YOLO dataset.

Labels are YOLO detection format: one line per box,
``class_id x_centre y_centre width height`` (coordinates are fractions of
the image size).
"""
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def source_photo_name(filename: str) -> str:
    """Return the source-photo name for a Roboflow export filename.

    Roboflow names augmented copies ``<original>.rf.<hash>.jpg``, so the text
    before ``.rf.`` identifies the source photo. This is only a proxy: two
    different photos can share a name.

    Args:
        filename: Image file name, e.g. ``"frame_6510_jpg.rf.ab12.jpg"``.

    Returns:
        The source name, e.g. ``"frame_6510_jpg"``. A name without ``.rf.``
        is returned unchanged.

    Example:
        >>> source_photo_name("frame_6510_jpg.rf.ab12.jpg")
        'frame_6510_jpg'
    """
    return filename.split(".rf.")[0]


def list_images(root: Path, split: str) -> list[Path]:
    """List the image files of one split in sorted order.

    Args:
        root: Dataset folder containing train/, valid/ and test/.
        split: One of ``"train"``, ``"valid"`` or ``"test"``.

    Returns:
        Sorted list of image paths.

    Raises:
        ValueError: If ``split`` is not a known split name.
        FileNotFoundError: If the split's images folder does not exist.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    folder = Path(root) / split / "images"
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing images folder: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for(image_path: Path) -> Path:
    """Return the label file that belongs to an image.

    Args:
        image_path: Path like ``<split>/images/<name>.jpg``.

    Returns:
        Path like ``<split>/labels/<name>.txt``.
    """
    image_path = Path(image_path)
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")


def read_label_classes(label_path: Path) -> list[int]:
    """Read the class id of every box in a YOLO label file.

    Args:
        label_path: Path to a ``.txt`` label file. An empty file means the
            image has no annotated boxes.

    Returns:
        List of class ids, one per box (empty if there are no boxes).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a non-empty line does not have exactly 5 fields.
    """
    label_path = Path(label_path)
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    classes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Bad label line in {label_path}: {line!r}")
        classes.append(int(parts[0]))
    return classes


def count_boxes_per_class(root: Path, split: str, class_names: list[str]) -> Counter:
    """Count annotated boxes per class in one split.

    Args:
        root: Dataset folder.
        split: Split name.
        class_names: Class names indexed by class id.

    Returns:
        Counter mapping class name to number of boxes.

    Raises:
        IndexError: If a label uses a class id outside ``class_names``.
    """
    counts = Counter()
    for image in list_images(root, split):
        for class_id in read_label_classes(label_path_for(image)):
            counts[class_names[class_id]] += 1
    return counts


def find_shared_sources(root: Path) -> dict[str, list[str]]:
    """Find source names that appear in more than one split.

    Args:
        root: Dataset folder.

    Returns:
        Dict mapping each shared source name to the splits it appears in.
    """
    seen = defaultdict(set)
    for split in SPLITS:
        for image in list_images(root, split):
            seen[source_photo_name(image.name)].add(split)
    return {n: sorted(s) for n, s in sorted(seen.items()) if len(s) > 1}


def build_filtered_train_list(root: Path) -> tuple[list[Path], list[Path]]:
    """Split training images into kept and excluded by source-name overlap.

    A training image is excluded if its source name also appears in valid or
    test. This is a conservative guard against leakage; valid and test are
    never modified.

    Args:
        root: Dataset folder.

    Returns:
        Tuple ``(kept, excluded)`` of training image paths.
    """
    eval_names = {
        source_photo_name(p.name)
        for split in ("valid", "test")
        for p in list_images(root, split)
    }
    kept, excluded = [], []
    for image in list_images(root, "train"):
        (excluded if source_photo_name(image.name) in eval_names else kept).append(image)
    return kept, excluded
