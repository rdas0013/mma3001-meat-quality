"""Extract features and targets for every image and save them to one file."""
import sys
import time
from pathlib import Path

import numpy as np

from meatqa.data import build_filtered_train_list, label_path_for, list_images
from meatqa.features import extract_features
from meatqa.targets import defect_area_fraction


def build(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Compute the feature matrix and target vector for a list of images.

    Args:
        paths: Image paths.

    Returns:
        Tuple ``(X, y)``: X has shape (n, 36), y is the defect area fraction.
    """
    X = np.stack([extract_features(p) for p in paths])
    y = np.array([defect_area_fraction(label_path_for(p)) for p in paths])
    return X, y


def main(root: Path) -> None:
    """Extract features for train (filtered), valid and test, then save to ``results/features.npz``.

    Args:
        root: Dataset folder containing train/, valid/ and test/.
    """
    kept, _ = build_filtered_train_list(root)
    splits = {"train": kept, "valid": list_images(root, "valid"), "test": list_images(root, "test")}
    out = {}
    for name, paths in splits.items():
        t0 = time.perf_counter()
        X, y = build(paths)
        dt = time.perf_counter() - t0
        out[f"X_{name}"], out[f"y_{name}"] = X, y
        print(f"{name}: {len(paths)} images, {dt:.1f} s total, {1000 * dt / len(paths):.1f} ms/image")
    np.savez("results/features.npz", **out)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data"))
