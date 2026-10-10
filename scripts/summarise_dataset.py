"""Print dataset size, image sizes and per-class box counts for each split."""
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from meatqa.data import SPLITS, count_boxes_per_class, find_shared_sources, list_images


def main(root: Path) -> None:
    """Print a summary of the dataset to stdout.

    Args:
        root: Dataset folder containing data.yaml, train/, valid/ and test/.
    """
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    for split in SPLITS:
        images = list_images(root, split)
        sizes = Counter(Image.open(p).size for p in images)
        print(f"\n=== {split}: {len(images)} images, sizes {dict(sizes)}")
        counts = count_boxes_per_class(root, split, names)
        for n in names:
            print(f"  {n:16s} {counts[n]}")
    print("\nSource names shared between splits:", find_shared_sources(root))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data"))
