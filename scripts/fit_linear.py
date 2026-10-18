"""Fit linear regression on the saved features and compare with the mean reference.

Reads ``results/features.npz``; prints RMSE, MAE, R2 and prediction time for
the train, valid and test splits.
"""
import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from meatqa.regression_metrics import mae, r2, rmse


def main() -> None:
    """Fit standardised linear regression on train and score every split.

    The scaler is fitted on train only, so no valid or test information
    leaks into the model. Linear regression has a closed-form least-squares
    solution, so there is no convergence criterion to set.
    """
    d = np.load("results/features.npz")
    scaler = StandardScaler().fit(d["X_train"])
    model = LinearRegression().fit(scaler.transform(d["X_train"]), d["y_train"])

    for split in ("train", "valid", "test"):
        X, y = scaler.transform(d[f"X_{split}"]), d[f"y_{split}"]
        t0 = time.perf_counter()
        pred = model.predict(X)
        ms = 1000 * (time.perf_counter() - t0) / len(y)
        print(f"{split}: RMSE={rmse(y, pred):.4f}  MAE={mae(y, pred):.4f}  "
              f"R2={r2(y, pred):.3f}  predict={ms:.4f} ms/image")


if __name__ == "__main__":
    main()
