"""Where do the annotated boxes sit in the frame?"""
from pathlib import Path

import numpy as np

from meatqa.data import label_path_for, list_images
from meatqa.targets import read_boxes

b = np.array([x[1:] for p in list_images(Path("data"), "valid")
              for x in read_boxes(label_path_for(p))])
xc, yc, w, h = b.T
print(f"{len(b)} boxes; mean centre x={xc.mean():.2f}, y={yc.mean():.2f}")
print(f"centre in corner quadrants (x<0.3 or >0.7 AND y<0.3 or >0.7): "
      f"{np.mean(((xc < .3) | (xc > .7)) & ((yc < .3) | (yc > .7))):.0%}")
print(f"boxes touching an image edge: "
      f"{np.mean((xc - w/2 <= 0.005) | (xc + w/2 >= 0.995) | (yc - h/2 <= 0.005) | (yc + h/2 >= 0.995)):.0%}")
