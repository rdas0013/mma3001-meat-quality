"""Tests for regression metrics using answers worked out by hand."""
import numpy as np
import pytest

from meatqa import regression_metrics as rm

Y = np.array([1.0, 2.0, 3.0, 4.0])
P = np.array([1.0, 2.0, 3.0, 6.0])  # only the last value is wrong, by 2


def test_rmse_hand_computed():
    # sqrt((0 + 0 + 0 + 4) / 4) = 1
    assert rm.rmse(Y, P) == pytest.approx(1.0)


def test_mae_hand_computed():
    # (0 + 0 + 0 + 2) / 4 = 0.5
    assert rm.mae(Y, P) == pytest.approx(0.5)


def test_r2_hand_computed():
    # mean(Y) = 2.5, SS_tot = 2.25+0.25+0.25+2.25 = 5, SS_res = 4, R2 = 1 - 4/5 = 0.2
    assert rm.r2(Y, P) == pytest.approx(0.2)


def test_r2_of_mean_predictor_is_zero():
    assert rm.r2(Y, np.full(4, Y.mean())) == pytest.approx(0.0)


def test_perfect_prediction():
    assert rm.rmse(Y, Y) == 0.0 and rm.r2(Y, Y) == 1.0


def test_constant_truth_raises_for_r2():
    with pytest.raises(ValueError):
        rm.r2(np.ones(3), np.ones(3))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        rm.rmse(np.ones(3), np.ones(2))
