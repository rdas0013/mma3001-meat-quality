"""Robustness test: how does the chosen SVR degrade under brightness shifts and blur?

Reads ``results/features.npz`` and the test images in ``data/``. The model is
fitted once on unmodified training features; only the test images are altered.
"""
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from meatqa.data import list_images
from meatqa.features import extract_features_fast
from meatqa.regression_metrics import r2, rmse


def perturbed_features(paths: list[Path], transform: Callable[[Image.Image], Image.Image]) -> np.ndarray:
    """Extract features from every image after applying a transform.

    Args:
        paths: Image paths.
        transform: Function mapping a PIL image to a modified PIL image.

    Returns:
        Feature matrix of shape (len(paths), N_FEATURES).
    """
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, p in enumerate(paths):
            out = Path(tmp) / f"{i}.png"  # PNG so no extra compression noise is added
            transform(Image.open(p).convert("RGB")).save(out)
            rows.append(extract_features_fast(out))
    return np.stack(rows)


def main() -> None:
    """Score the SVR on test images under brightness and blur perturbations."""
    d = np.load("results/features.npz")
    scaler = StandardScaler().fit(d["X_train"])
    model = SVR(kernel="rbf", C=1, epsilon=0.03).fit(scaler.transform(d["X_train"]), d["y_train"])
    y_test = d["y_test"]
    paths = list_images(Path("data"), "test")

    conditions = {
        "original": lambda im: im,
        "brightness x0.7": lambda im: ImageEnhance.Brightness(im).enhance(0.7),
        "brightness x0.85": lambda im: ImageEnhance.Brightness(im).enhance(0.85),
        "brightness x1.15": lambda im: ImageEnhance.Brightness(im).enhance(1.15),
        "brightness x1.3": lambda im: ImageEnhance.Brightness(im).enhance(1.3),
        "blur radius 2": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
        "blur radius 5": lambda im: im.filter(ImageFilter.GaussianBlur(5)),
    }

    print(f"{'condition':18s} test R2   RMSE     mean prediction (true mean {y_test.mean():.4f})")
    for name, fn in conditions.items():
        pred = model.predict(scaler.transform(perturbed_features(paths, fn)))
        print(f"{name:18s} {r2(y_test, pred):7.3f}  {rmse(y_test, pred):.4f}   {pred.mean():.4f}")


if __name__ == "__main__":
    main()
