"""Compare timing of the original and fast (bincount) feature extractors."""
import time
from pathlib import Path
from typing import Callable

from meatqa import features
from meatqa.data import list_images


def best_pass_ms(fn: Callable, paths: list[Path], repeats: int = 3) -> float:
    """Return the best per-image time in ms over several passes.

    Args:
        fn: Feature extractor taking an image path.
        paths: Images to process.
        repeats: Number of full passes; the fastest is reported.

    Returns:
        Milliseconds per image for the fastest pass.
    """
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for p in paths:
            fn(p)
        best = min(best, time.perf_counter() - t0)
    return 1000 * best / len(paths)


def main(n_images: int = 80) -> None:
    """Print per-image timing for both extractors.

    Args:
        n_images: Number of valid images to time.
    """
    paths = list_images(Path("data"), "valid")[:n_images]
    for name, fn in (("original", features.extract_features), ("fast", features.extract_features_fast)):
        fn(paths[0])  # warm-up so one-off start-up cost is not counted
        print(f"{name:9s}: {best_pass_ms(fn, paths):.1f} ms/image "
              f"(best of 3 passes over {len(paths)} images)")


if __name__ == "__main__":
    main()
