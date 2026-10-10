"""Severity targets computed from YOLO label files."""
from pathlib import Path


def read_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read every box from a YOLO label file.

    Args:
        label_path: Path to a ``.txt`` label file. An empty file means no boxes.

    Returns:
        List of ``(class_id, x_centre, y_centre, width, height)`` tuples, with
        coordinates as fractions of the image size.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a non-empty line does not have exactly 5 fields.
    """
    label_path = Path(label_path)
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Bad label line in {label_path}: {line!r}")
        boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def defect_area_fraction(label_path: Path) -> float:
    """Return the total box area as a fraction of the image area.

    Each box contributes ``width * height`` (both already fractions). Overlapping
    boxes are double-counted, so this is a proxy for severity, not a measured area.

    Args:
        label_path: Path to a YOLO label file.

    Returns:
        Sum of box areas; 0.0 for an image with no boxes.
    """
    return sum(w * h for _, _, _, w, h in read_boxes(label_path))


def largest_defect_fraction(label_path: Path) -> float:
    """Return the area fraction of the single largest box.

    Args:
        label_path: Path to a YOLO label file.

    Returns:
        ``max(width * height)`` over the boxes; 0.0 for an image with no boxes.
    """
    return max((w * h for _, _, _, w, h in read_boxes(label_path)), default=0.0)
