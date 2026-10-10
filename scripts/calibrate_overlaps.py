"""Calibrate the thumbnail-correlation score and list best train matches."""
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def thumbnail_vector(path: Path, size: int = 32) -> np.ndarray:
    """Convert an image to a standardised grayscale thumbnail vector.

    Args:
        path: Path to an image file.
        size: Thumbnail width and height in pixels.

    Returns:
        Flattened array of length ``size * size`` with zero mean and unit variance.
    """
    img = Image.open(path).convert("L").resize((size, size))
    v = np.asarray(img, dtype=float).ravel()
    return (v - v.mean()) / (v.std() + 1e-9)


def corr(a: Path, b: Path) -> float:
    """Return the correlation of two images' thumbnail vectors (1.0 = same layout)."""
    va, vb = thumbnail_vector(a), thumbnail_vector(b)
    return float(va @ vb) / len(va)


def main(root: Path) -> None:
    """Print reference score distributions, then best train matches for shared names.

    Args:
        root: Dataset folder containing train/, valid/ and test/.
    """
    rng = random.Random(0)  # fixed seed so the result is repeatable
    groups = defaultdict(lambda: defaultdict(list))
    for split in ["train", "valid", "test"]:
        for p in (root / split / "images").glob("*"):
            groups[p.name.split(".rf.")[0]][split].append(p)

    # Reference 1: two train copies of one source name (assumed same photo).
    pairs = [(d["train"][0], d["train"][1]) for d in groups.values() if len(d.get("train", [])) >= 2]
    same = [corr(a, b) for a, b in rng.sample(pairs, min(150, len(pairs)))]

    # Reference 2: random train images with different source names.
    train_imgs = [(n, p) for n, d in groups.items() for p in d.get("train", [])]
    diff = []
    while len(diff) < 300:
        (n1, p1), (n2, p2) = rng.sample(train_imgs, 2)
        if n1 != n2:
            diff.append(corr(p1, p2))

    print(f"SAME photo (n={len(same)}): min {min(same):.2f}, median {np.median(same):.2f}")
    print(f"DIFFERENT photos (n={len(diff)}): median {np.median(diff):.2f}, "
          f"95th pct {np.percentile(diff, 95):.2f}, max {max(diff):.2f}")

    print("\nShared-name evaluation images:")
    for name, d in sorted(groups.items()):
        if "train" not in d or len(d) < 2:
            continue
        for split in ("valid", "test"):
            for p in d.get(split, []):
                scores = [(corr(p, t), t.name) for t in d["train"]]
                best, best_name = max(scores)
                print(f"{split} {p.name} | best {best:.2f} | train file: {best_name}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data"))
