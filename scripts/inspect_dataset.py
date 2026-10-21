"""Print a summary of the dataset folder: file types, structure, image sizes, label sample."""
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")

files = [p for p in root.rglob("*") if p.is_file()]
print("Total files:", len(files))
print("File types:", Counter(p.suffix.lower() for p in files))

print("\nFolder structure (2 levels):")
for d in sorted(p for p in root.rglob("*") if p.is_dir() and len(p.relative_to(root).parts) <= 2):
    n = sum(1 for f in d.iterdir() if f.is_file())
    print(f"  {d.relative_to(root)}  ({n} files)")

imgs = [p for p in files if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
print("\nImages:", len(imgs))
print("Most common sizes:", Counter(Image.open(p).size for p in imgs[:200]).most_common(5))

labels = [p for p in files if p.suffix.lower() in {".txt", ".xml", ".json", ".csv"}]
print("\nLabel-like files:", len(labels))
for p in labels[:2]:
    print(f"\n--- {p} (first 15 lines) ---")
    print("\n".join(p.read_text(errors="ignore").splitlines()[:15]))
