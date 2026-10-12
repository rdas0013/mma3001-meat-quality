"""Classification metrics for tray-level pass/fail (1 = fail, 0 = pass)."""
import numpy as np


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Count true/false positives and negatives, treating 1 (fail) as positive.

    Args:
        y_true: True labels, values 0 or 1.
        y_pred: Predicted labels, same length as ``y_true``.

    Returns:
        Dict with keys ``tp``, ``fp``, ``tn``, ``fn``.

    Raises:
        ValueError: If the arrays differ in length.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def _safe_div(a: float, b: float) -> float:
    """Return a / b, or 0.0 if b is zero."""
    return a / b if b else 0.0


def summarise(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy plus precision and recall for both classes.

    Pass-class metrics matter because an always-fail model has high accuracy
    but zero recall for the pass class.

    Args:
        y_true: True labels (0 or 1).
        y_pred: Predicted labels (0 or 1).

    Returns:
        Dict with ``accuracy``, ``fail_precision``, ``fail_recall``,
        ``pass_precision``, ``pass_recall`` and the four confusion counts.
    """
    c = confusion_counts(y_true, y_pred)
    n = sum(c.values())
    return {
        "accuracy": _safe_div(c["tp"] + c["tn"], n),
        "fail_precision": _safe_div(c["tp"], c["tp"] + c["fp"]),
        "fail_recall": _safe_div(c["tp"], c["tp"] + c["fn"]),
        "pass_precision": _safe_div(c["tn"], c["tn"] + c["fn"]),
        "pass_recall": _safe_div(c["tn"], c["tn"] + c["fp"]),
        **c,
    }
