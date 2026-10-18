"""Test how shrinking the image changes extraction time and model accuracy.

Resizes images (bilinear interpolation), re-extracts features, refits the SVR
on every 3rd filtered training image, and reports valid/test R2. Timings
include a PNG write and read, so they overstate the cost of a real resize.
"""
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from meatqa import features
from meatqa.data import build_filtered_train_list, label_path_for, list_images
from meatqa.regression_metrics import r2, rmse
from meatqa.targets import defect_area_fraction

TMP = Path("results/_tmp_resized.png")


def extract_resized(path: Path, scale: float) -> np.ndarray:
    """Extract features after resizing the image by ``scale``.

    Args:
        path: Image path.
        scale: Resize factor, e.g. 0.5 halves width and height.

    Returns:
        Feature vector of length ``features.N_FEATURES``.
    """
    if scale == 1.0:
        return features.extract_features(path)
    img = Image.open(path).convert("RGB")
    img.resize((int(720 * scale), int(540 * scale)), Image.BILINEAR).save(TMP)
    return features.extract_features(TMP)


def main(root: Path = Path("data")) -> None:
    """Run the resolution experiment at scales 1.0, 0.5 and 0.25.

    Args:
        root: Dataset folder.
    """
    kept, _ = build_filtered_train_list(root)
    splits = {"train": kept[::3], "valid": list_images(root, "valid"), "test": list_images(root, "test")}
    print("(train uses every 3rd filtered image to keep this quick)")

    for scale in (1.0, 0.5, 0.25):
        X, y, ms = {}, {}, {}
        for name, paths in splits.items():
            t0 = time.perf_counter()
            X[name] = np.stack([extract_resized(p, scale) for p in paths])
            ms[name] = 1000 * (time.perf_counter() - t0) / len(paths)
            y[name] = np.array([defect_area_fraction(label_path_for(p)) for p in paths])
        sc = StandardScaler().fit(X["train"])
        model = SVR(kernel="rbf", C=1, epsilon=0.03).fit(sc.transform(X["train"]), y["train"])
        out = {s: model.predict(sc.transform(X[s])) for s in ("valid", "test")}
        print(f"scale {scale}: {int(720 * scale)}x{int(540 * scale)}  extract+resize {ms['valid']:.1f} ms/image  "
              f"valid R2={r2(y['valid'], out['valid']):.3f}  test R2={r2(y['test'], out['test']):.3f}  "
              f"test RMSE={rmse(y['test'], out['test']):.4f}")


if __name__ == "__main__":
    main()
