"""Test whether images sharing a source name across splits are the same photo."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def thumbnail_vector(path: Path, size: int = 32) -> np.ndarray:
    """Convert an image to a brightness-invariant vector.

    The image is shrunk to a small grayscale thumbnail and standardised
    (zero mean, unit variance), so exposure changes from augmentation do
    not affect the comparison.

    Args:
        path: Path to an image file.
        size: Thumbnail width and height in pixels.

    Returns:
        A flattened, standardised array of length ``size * size``.
    """
    img = Image.open(path).convert("L").resize((size, size))
    v = np.asarray(img, dtype=float).ravel()
    return (v - v.mean()) / (v.std() + 1e-9)


def main(root: Path) -> None:
    """Print the best train-set match for each evaluation image with a shared name.

    Args:
        root: Dataset folder containing train/, valid/ and test/.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for split in ["train", "valid", "test"]:
        for p in (root / split / "images").glob("*"):
            groups[p.name.split(".rf.")[0]][split].append(p)

    for name, d in sorted(groups.items()):
        if len(d) < 2 or "train" not in d:
            continue
        train_vecs = [thumbnail_vector(p) for p in d["train"]]
        for split in ("valid", "test"):
            for p in d.get(split, []):
                v = thumbnail_vector(p)
                # Correlation of standardised vectors: 1.0 means identical layout.
                best = max(float(v @ t) / len(v) for t in train_vecs)
                print(f"{name} | {split} | best match to train: {best:.2f}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data"))
