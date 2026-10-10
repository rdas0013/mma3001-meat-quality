"""Write the leakage-filtered training image list and a record of exclusions."""
import sys
from pathlib import Path

from meatqa.data import build_filtered_train_list


def main(root: Path) -> None:
    """Write ``train_filtered.txt`` (in root) and ``results/excluded_train_images.txt``.

    Args:
        root: Dataset folder containing train/, valid/ and test/.
    """
    kept, excluded = build_filtered_train_list(root)
    (root / "train_filtered.txt").write_text("\n".join(str(p.resolve()) for p in kept))
    Path("results/excluded_train_images.txt").write_text("\n".join(p.name for p in excluded))
    print(f"kept {len(kept)} train images, excluded {len(excluded)}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data"))
