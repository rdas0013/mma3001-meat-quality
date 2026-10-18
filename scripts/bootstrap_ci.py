"""Bootstrap confidence intervals on test R2 and RMSE for linear regression and SVR.

Reads ``results/features.npz``. Resamples the 80 test images with replacement
and reports 95% percentile intervals, plus the paired SVR-minus-linear R2.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from meatqa.regression_metrics import r2, rmse


def main(n_boot: int = 2000, seed: int = 0) -> None:
    """Fit both models on train, then bootstrap their test-set scores.

    Args:
        n_boot: Number of bootstrap resamples.
        seed: Random seed, fixed so the intervals are repeatable.
    """
    d = np.load("results/features.npz")
    scaler = StandardScaler().fit(d["X_train"])  # fitted on train only
    Xtr, ytr = scaler.transform(d["X_train"]), d["y_train"]
    Xte, yte = scaler.transform(d["X_test"]), d["y_test"]

    pred = {
        "linear": LinearRegression().fit(Xtr, ytr).predict(Xte),
        "SVR": SVR(kernel="rbf", C=1, epsilon=0.03).fit(Xtr, ytr).predict(Xte),
    }

    rng = np.random.default_rng(seed)
    n = len(yte)
    idx = rng.integers(0, n, size=(n_boot, n))  # resample test images with replacement

    stats = {k: {"r2": [], "rmse": []} for k in pred}
    diff = []
    for ix in idx:
        for k, p in pred.items():
            stats[k]["r2"].append(r2(yte[ix], p[ix]))
            stats[k]["rmse"].append(rmse(yte[ix], p[ix]))
        # Paired difference: both models scored on the same resample.
        diff.append(stats["SVR"]["r2"][-1] - stats["linear"]["r2"][-1])

    for k in pred:
        for m in ("r2", "rmse"):
            lo, hi = np.percentile(stats[k][m], [2.5, 97.5])
            point = (r2 if m == "r2" else rmse)(yte, pred[k])
            print(f"{k:6s} test {m:4s}: point={point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print(f"SVR minus linear test R2: 95% CI [{lo:.4f}, {hi:.4f}]  "
          f"(share of resamples SVR > linear: {np.mean(np.array(diff) > 0):.0%})")


if __name__ == "__main__":
    main()
