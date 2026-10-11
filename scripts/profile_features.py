"""Profile feature extraction on a sample of training images."""
import cProfile
import pstats
from pathlib import Path

from meatqa.data import list_images
from meatqa.features import extract_features

paths = list_images(Path("data"), "valid")[:60]


def run():
    """Extract features for the sample images."""
    for p in paths:
        extract_features(p)


cProfile.run("run()", "results/profile.out")
pstats.Stats("results/profile.out").sort_stats("cumulative").print_stats(12)
