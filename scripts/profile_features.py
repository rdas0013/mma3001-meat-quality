"""Profile feature extraction on a sample of validation images.

Writes the raw profile to ``results/profile.out`` (git-ignored) and prints
the 12 most expensive calls by cumulative time.
"""
import cProfile
import pstats
from pathlib import Path

from meatqa.data import list_images
from meatqa.features import extract_features


def main(n_images: int = 60) -> None:
    """Profile ``extract_features`` over the first ``n_images`` valid images.

    Args:
        n_images: Number of images to process.
    """
    paths = list_images(Path("data"), "valid")[:n_images]

    def run() -> None:
        """Extract features for the sample images."""
        for p in paths:
            extract_features(p)

    cProfile.runctx("run()", globals(), {"run": run}, "results/profile.out")
    pstats.Stats("results/profile.out").sort_stats("cumulative").print_stats(12)


if __name__ == "__main__":
    main()
