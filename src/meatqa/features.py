"""Hand-crafted image features: HSV colour histograms and finite-difference edge strength."""
from pathlib import Path

import numpy as np
from PIL import Image

HUE_BINS = 16
SAT_BINS = 8
VAL_BINS = 8
GRID = 4  # edge strength is averaged over a GRID x GRID grid of cells
N_FEATURES = HUE_BINS + SAT_BINS + VAL_BINS + GRID * GRID  # 36


def extract_features(image_path: Path) -> np.ndarray:
    """Compute a fixed-length feature vector for one image.

    The vector joins three normalised HSV histograms (hue, saturation, value)
    with the mean gradient magnitude in each cell of a GRID x GRID grid. The
    histograms capture the colour mix (e.g. pink meat vs pale film). The grid
    captures coarse edge content, since folds and gaps create sharp brightness
    changes. Gradients use central finite differences (``np.gradient``).

    Args:
        image_path: Path to an RGB image.

    Returns:
        1-D float array of length ``N_FEATURES``.

    Raises:
        FileNotFoundError: If the image does not exist.
        OSError: If the file cannot be read as an image.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    hsv = np.asarray(img.convert("HSV"), dtype=float)

    feats = []
    for channel, bins in enumerate((HUE_BINS, SAT_BINS, VAL_BINS)):
        hist, _ = np.histogram(hsv[..., channel], bins=bins, range=(0, 256))
        # Normalise so each histogram sums to 1 and image size does not matter.
        feats.append(hist / hist.sum())

    gray = np.asarray(img.convert("L"), dtype=float)
    gy, gx = np.gradient(gray)  # central differences along rows and columns
    mag = np.hypot(gx, gy)
    h, w = mag.shape
    cells = [
        mag[i * h // GRID:(i + 1) * h // GRID, j * w // GRID:(j + 1) * w // GRID].mean()
        for i in range(GRID) for j in range(GRID)
    ]
    feats.append(np.array(cells))
    return np.concatenate(feats)
