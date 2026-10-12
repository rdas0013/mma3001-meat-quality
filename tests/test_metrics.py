"""Tests for pass/fail metrics using hand-computed answers."""
import numpy as np
import pytest

from meatqa import metrics


def test_confusion_counts_hand_counted():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0])
    assert metrics.confusion_counts(y_true, y_pred) == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}


def test_always_fail_has_high_accuracy_but_zero_pass_recall():
    y_true = np.array([1] * 85 + [0] * 15)
    s = metrics.summarise(y_true, np.ones(100, dtype=int))
    assert s["accuracy"] == pytest.approx(0.85)
    assert s["pass_recall"] == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.confusion_counts(np.array([1, 0]), np.array([1]))
