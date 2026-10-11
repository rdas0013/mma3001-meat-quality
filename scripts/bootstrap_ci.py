"""Bootstrap confidence intervals on test R2 and RMSE for linear regression and SVR."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from meatqa.regression_metrics import r2, rmse

d = np.load("results/features.npz")
scaler = StandardScaler().fit(d["X_train"])  # fitted on train only
Xtr, ytr = scaler.transform(d["X_train"]), d["y_train"]
Xte, yte = scaler.transform(d["X_test"]), d["y_test"]

pred = {
    "linear": LinearRegression().fit(Xtr, ytr).predict(Xte),
    "SVR": SVR(kernel="rbf", C=1, epsilon=0.03).fit(Xtr, ytr).predict(Xte),
}

rng = np.random.default_rng(0)  # fixed seed so results are repeatable
n, B = len(yte), 2000
idx = rng.integers(0, n, size=(B, n))  # resample test images with replacement

stats = {k: {"r2": [], "rmse": []} for k in pred}
diff = []
for ix in idx:
    for k, p in pred.items():
        stats[k]["r2"].append(r2(yte[ix], p[ix]))
        stats[k]["rmse"].append(rmse(yte[ix], p[ix]))
    diff.append(stats["SVR"]["r2"][-1] - stats["linear"]["r2"][-1])

for k in pred:
    for m in ("r2", "rmse"):
        lo, hi = np.percentile(stats[k][m], [2.5, 97.5])
        print(f"{k:6s} test {m:4s}: point={(r2 if m == 'r2' else rmse)(yte, pred[k]):.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
lo, hi = np.percentile(diff, [2.5, 97.5])
print(f"SVR minus linear test R2: 95% CI [{lo:.4f}, {hi:.4f}]  (share of resamples SVR > linear: {np.mean(np.array(diff) > 0):.0%})")
