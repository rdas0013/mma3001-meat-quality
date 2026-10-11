"""Score the 'predict the training mean' reference line on valid and test."""
import numpy as np

from meatqa.regression_metrics import mae, r2, rmse

d = np.load("results/features.npz")
mean_train = d["y_train"].mean()
print(f"training mean = {mean_train:.4f}")
for split in ("valid", "test"):
    y = d[f"y_{split}"]
    pred = np.full_like(y, mean_train)
    print(f"{split}: RMSE={rmse(y, pred):.4f}  MAE={mae(y, pred):.4f}  R2={r2(y, pred):.3f}")
