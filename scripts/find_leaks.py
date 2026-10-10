"""List source-photo names that appear in more than one split."""
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
groups = defaultdict(lambda: defaultdict(list))
for split in ["train", "valid", "test"]:
    for p in (root / split / "images").glob("*"):
        groups[p.name.split(".rf.")[0]][split].append(p.name)

for name, d in sorted(groups.items()):
    if len(d) > 1:
        print(name, {s: v for s, v in d.items()})
