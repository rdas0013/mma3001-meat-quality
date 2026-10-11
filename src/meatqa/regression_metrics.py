"""Regression metrics (unit topic 5.5): RMSE, MAE and R-squared."""
import numpy as np


def _check(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to float arrays and check they match.

    Raises:
        ValueError: If shapes differ or the arrays are empty.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same shape")
    return y_true, y_pred


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error: sqrt(mean((y_true - y_pred)^2)).

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        RMSE, in the same units as the target.
    """
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error: mean(|y_true - y_pred|).

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        MAE, in the same units as the target.
    """
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared: 1 - SS_residual / SS_total.

    0 means no better than predicting the mean of ``y_true``; negative means
    worse. Undefined if ``y_true`` is constant.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        The R-squared score.

    Raises:
        ValueError: If ``y_true`` is constant (zero total variance).
    """
    y_true, y_pred = _check(y_true, y_pred)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        raise ValueError("R2 is undefined when y_true is constant")
    return float(1 - np.sum((y_true - y_pred) ** 2) / ss_tot)
