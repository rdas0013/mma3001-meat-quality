"""Summarise the severity targets (defect area fraction, largest defect) per split."""
import sys
from pathlib import Path

import numpy as np

from meatqa.data import SPLITS, label_path_for, list_images
from meatqa.targets import defect_area_fraction, largest_defect_fraction


def summarise(values: np.ndarray) -> str:
    """Format count, mean, spread and zero-share of a target array.

    Args:
        values: 1-D array of target values.

    Returns:
        One-line summary string.
    """
    return (f"n={len(values)}  mean={values.mean():.4f}  std={values.std():.4f}  "
            f"median={np.median(values):.4f}  max={values.max():.4f}  "
            f"zero={np.mean(values == 0):.0%}")


def main(root: Path) -> None:
    """Print target statistics for every split.

    Args:
        root: Dataset folder containing train/, valid/ and test/.
    """
    for split in SPLITS:
        paths = [label_path_for(p) for p in list_images(root, split)]
        area = np.array([defect_area_fraction(p) for p in paths])
        largest = np.array([largest_defect_fraction(p) for p in paths])
        print(f"\n=== {split} ===")
        print("area fraction :", summarise(area))
        print("largest defect:", summarise(largest))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data"))
