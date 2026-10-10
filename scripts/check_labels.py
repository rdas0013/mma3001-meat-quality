"""Count images, source photos and boxes per class for each split of a YOLO dataset."""
import sys
from collections import Counter
from pathlib import Path

import yaml

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
names = yaml.safe_load((root / "data.yaml").read_text())["names"]
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]

origin_sets = {}
for split in ["train", "valid", "test"]:
    images = sorted((root / split / "images").glob("*"))
    labels = sorted((root / split / "labels").glob("*.txt"))
    # Roboflow names augmented copies <original>.rf.<hash>.jpg, so the text before ".rf." identifies the source photo
    origins = {p.name.split(".rf.")[0] for p in images}
    origin_sets[split] = origins

    boxes = Counter()
    empty = 0
    for lab in labels:
        lines = [l for l in lab.read_text().splitlines() if l.strip()]
        empty += (len(lines) == 0)
        for l in lines:
            boxes[names[int(l.split()[0])]] += 1

    print(f"\n=== {split} ===")
    print("images:", len(images), "| unique source photos:", len(origins), "| images with no boxes:", empty)
    for n in names:
        print(f"  {n:16s} {boxes[n]}")

print("\n=== leakage check (source photos shared between splits) ===")
for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
    print(f"{a} & {b}:", len(origin_sets[a] & origin_sets[b]))
