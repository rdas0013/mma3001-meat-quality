"""Compare timing of the original and fast feature extractors."""
import time
from pathlib import Path

from meatqa import features
from meatqa.data import list_images

paths = list_images(Path("data"), "valid")[:80]
for name, fn in (("original", features.extract_features), ("fast", features.extract_features_fast)):
    fn(paths[0])  # warm-up so one-off start-up cost is not counted
    best = min(
        (lambda t0: (sum(1 for p in paths if fn(p) is not None), time.perf_counter() - t0)[1])(time.perf_counter())
        for _ in range(3)
    )
    print(f"{name:9s}: {1000 * best / len(paths):.1f} ms/image (best of 3 passes over {len(paths)} images)")
