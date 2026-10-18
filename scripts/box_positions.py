"""Report where the annotated boxes sit in the frame (valid split).

Prints the mean box centre, the share of boxes centred in a corner quadrant,
and the share touching an image edge. Evidence that the dataset's centre-crop
clips many boxes.
"""
from pathlib import Path

import numpy as np

from meatqa.data import label_path_for, list_images
from meatqa.targets import read_boxes


def main(root: Path = Path("data"), split: str = "valid") -> None:
    """Print box-position statistics for one split.

    Args:
        root: Dataset folder.
        split: Split to analyse.
    """
    b = np.array([x[1:] for p in list_images(root, split)
                  for x in read_boxes(label_path_for(p))])
    xc, yc, w, h = b.T
    print(f"{len(b)} boxes; mean centre x={xc.mean():.2f}, y={yc.mean():.2f}")
    corner = ((xc < .3) | (xc > .7)) & ((yc < .3) | (yc > .7))
    print(f"centre in corner quadrants (x<0.3 or >0.7 AND y<0.3 or >0.7): {np.mean(corner):.0%}")
    edge = (xc - w / 2 <= 0.005) | (xc + w / 2 >= 0.995) | (yc - h / 2 <= 0.005) | (yc + h / 2 >= 0.995)
    print(f"boxes touching an image edge: {np.mean(edge):.0%}")


if __name__ == "__main__":
    main()
