"""Fit linear regression on the saved features and compare with the mean reference."""
import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from meatqa.regression_metrics import mae, r2, rmse

d = np.load("results/features.npz")
# Standardise so features with different scales are treated comparably;
# the scaler is fitted on train only, so valid/test information never leaks in.
scaler = StandardScaler().fit(d["X_train"])
model = LinearRegression().fit(scaler.transform(d["X_train"]), d["y_train"])

for split in ("train", "valid", "test"):
    X, y = scaler.transform(d[f"X_{split}"]), d[f"y_{split}"]
    t0 = time.perf_counter()
    pred = model.predict(X)
    ms = 1000 * (time.perf_counter() - t0) / len(y)
    print(f"{split}: RMSE={rmse(y, pred):.4f}  MAE={mae(y, pred):.4f}  "
          f"R2={r2(y, pred):.3f}  predict={ms:.4f} ms/image")
