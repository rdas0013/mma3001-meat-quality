"""Find the test images with the largest prediction errors and save them with boxes drawn on.

Reads ``results/features.npz`` and the test images in ``data/``. Writes
annotated images to ``results/failures/`` (git-ignored) and prints a summary.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from meatqa.data import label_path_for, list_images
from meatqa.targets import read_boxes

NAMES = ["loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle"]


def save_annotated(image_path: Path, out_path: Path, true: float, pred: float) -> None:
    """Save a copy of an image with its labelled boxes and a true/pred caption.

    Args:
        image_path: Source image.
        out_path: Where to save the annotated copy.
        true: True defect area fraction.
        pred: Predicted defect area fraction.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for _, xc, yc, bw, bh in read_boxes(label_path_for(image_path)):
        draw.rectangle([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h],
                       outline="red", width=3)
    draw.text((8, 8), f"true {true:.3f}  pred {pred:.3f}", fill="yellow")
    img.save(out_path)


def main(n_show: int = 8) -> None:
    """Print the largest test errors and save the worst ones as images.

    Args:
        n_show: How many of the largest errors to list and save.
    """
    d = np.load("results/features.npz")
    scaler = StandardScaler().fit(d["X_train"])
    model = SVR(kernel="rbf", C=1, epsilon=0.03).fit(scaler.transform(d["X_train"]), d["y_train"])
    pred = model.predict(scaler.transform(d["X_test"]))
    y = d["y_test"]
    paths = list_images(Path("data"), "test")
    err = pred - y  # positive = over-predicted

    out = Path("results/failures")
    out.mkdir(parents=True, exist_ok=True)

    print("Largest errors on test (pred - true):")
    for rank, i in enumerate(np.argsort(-np.abs(err))[:n_show], 1):
        classes = sorted({NAMES[c] for c, *_ in read_boxes(label_path_for(paths[i]))})
        print(f"{rank}. true={y[i]:.3f} pred={pred[i]:.3f} err={err[i]:+.3f}  classes={classes or 'none'}")
        save_annotated(paths[i], out / f"{rank}_{paths[i].name.split('.rf.')[0]}.jpg", y[i], pred[i])

    print(f"\nOver-predicted: {np.sum(err > 0)}  under-predicted: {np.sum(err < 0)}")
    print(f"Mean error (bias): {err.mean():+.4f}")
    big = y > 0.1
    print(f"Mean error for true area > 0.1: {err[big].mean():+.4f} (n={big.sum()})")
    print(f"Mean error for true area = 0: {err[y == 0].mean():+.4f} (n={np.sum(y == 0)})")


if __name__ == "__main__":
    main()
