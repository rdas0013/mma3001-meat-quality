"""Top-level inference: predict defect area and a pass/fail decision for one tray image."""
from pathlib import Path

import numpy as np
from PIL import Image

from meatqa.features import extract_features_fast

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXPECTED_SIZE = (720, 540)  # (width, height) the model was trained on
DEFAULT_THRESHOLD = 0.04    # chosen on the validation split by balanced accuracy


def load_model(path: Path) -> dict:
    """Load a trained model bundle saved by ``scripts/train_final_model.py``.

    Args:
        path: Path to a ``.joblib`` file holding ``{"scaler": ..., "model": ...}``.

    Returns:
        The bundle dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    import joblib

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing model file: {path}")
    return joblib.load(path)


def check_image(image_path: Path) -> tuple[int, int]:
    """Validate an input image and return its size.

    Args:
        image_path: Path to the image.

    Returns:
        ``(width, height)`` in pixels.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the extension is unsupported, the file cannot be read
            as an image, or the size is not 720 x 540.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type {image_path.suffix!r}; use {sorted(SUPPORTED_SUFFIXES)}")
    try:
        with Image.open(image_path) as im:
            im.load()  # forces a full decode, so truncated files fail here
            size = im.size
    except OSError as exc:  # includes PIL.UnidentifiedImageError
        raise ValueError(f"Cannot read {image_path} as an image: {exc}") from exc
    if size != EXPECTED_SIZE:
        raise ValueError(f"Image is {size[0]}x{size[1]}; expected {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}")
    return size


def assess_image(image_path: Path, bundle: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Predict defect area and a pass/fail decision for one tray image.

    The prediction is the SVR's estimate of the total annotated-box area as a
    fraction of the image, clipped at 0 because area cannot be negative. The
    tray is called ``"fail"`` if the predicted area is above ``threshold``.

    Limitations: the model correlates whole-image colour and edge summaries
    with box area. It does not locate defects, it is sensitive to darkening
    and blur, and "pass" only means no defect area was predicted.

    Args:
        image_path: Path to a 720 x 540 RGB image (.jpg, .jpeg or .png).
        bundle: Dict with a fitted ``"scaler"`` and ``"model"`` (see ``load_model``).
        threshold: Predicted-area cut-off above which the tray fails.

    Returns:
        Dict with ``image`` (str), ``predicted_area_fraction`` (float, >= 0),
        ``decision`` (``"pass"`` or ``"fail"``) and ``threshold`` (float).

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the image is unsupported, unreadable or the wrong size.
    """
    check_image(image_path)
    feats = extract_features_fast(image_path).reshape(1, -1)
    area = float(bundle["model"].predict(bundle["scaler"].transform(feats))[0])
    area = max(area, 0.0)
    return {
        "image": str(image_path),
        "predicted_area_fraction": area,
        "decision": "fail" if area > threshold else "pass",
        "threshold": threshold,
    }
