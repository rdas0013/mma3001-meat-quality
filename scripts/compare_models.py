"""Compare linear regression, decision tree and SVR on the saved features.

Reads ``results/features.npz``; prints fit time and RMSE, MAE, R2 and
prediction time per split for each model.
"""
import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from meatqa.regression_metrics import mae, r2, rmse


def main() -> None:
    """Fit three models on train and print scores for every split.

    Settings here are untuned first guesses (tree depth 5, SVR C=1,
    epsilon=0.01); ``tune_models.py`` chooses better ones on valid.
    """
    d = np.load("results/features.npz")
    # Scaler fitted on train only, so no valid/test information leaks in.
    scaler = StandardScaler().fit(d["X_train"])
    X = {s: scaler.transform(d[f"X_{s}"]) for s in ("train", "valid", "test")}
    y = {s: d[f"y_{s}"] for s in ("train", "valid", "test")}

    models = {
        "linear": LinearRegression(),
        # max_depth limits tree size; an unlimited tree memorises the training set.
        "tree (depth 5)": DecisionTreeRegressor(max_depth=5, random_state=0),
        # C sets the penalty on errors; epsilon is the no-penalty band.
        "SVR (rbf)": SVR(kernel="rbf", C=1.0, epsilon=0.01),
    }

    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X["train"], y["train"])
        fit_s = time.perf_counter() - t0
        print(f"\n{name}  (fit {fit_s:.2f} s)")
        for split in ("train", "valid", "test"):
            t0 = time.perf_counter()
            pred = model.predict(X[split])
            ms = 1000 * (time.perf_counter() - t0) / len(y[split])
            print(f"  {split}: RMSE={rmse(y[split], pred):.4f}  MAE={mae(y[split], pred):.4f}  "
                  f"R2={r2(y[split], pred):.3f}  predict={ms:.4f} ms/image")


if __name__ == "__main__":
    main()
