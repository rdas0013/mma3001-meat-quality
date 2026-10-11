"""Tune tree depth and SVR settings on valid only, then score test once per model."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from meatqa.regression_metrics import r2, rmse

d = np.load("results/features.npz")
# Scaler fitted on train only, so no valid/test information leaks in.
scaler = StandardScaler().fit(d["X_train"])
X = {s: scaler.transform(d[f"X_{s}"]) for s in ("train", "valid", "test")}
y = {s: d[f"y_{s}"] for s in ("train", "valid", "test")}


def score(model, split):
    """Return (RMSE, R2) of a fitted model on one split."""
    pred = model.predict(X[split])
    return rmse(y[split], pred), r2(y[split], pred)


print("=== Decision tree: max_depth ===")
best_tree = None
for depth in (2, 3, 4, 5, 6, 8, 10, 15):
    m = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X["train"], y["train"])
    tr, va = score(m, "train"), score(m, "valid")
    print(f"depth {depth:2d}: train R2={tr[1]:.3f}  valid RMSE={va[0]:.4f}  valid R2={va[1]:.3f}")
    if best_tree is None or va[1] > best_tree[0]:
        best_tree = (va[1], depth)

print("\n=== SVR (rbf): C x epsilon, valid R2 ===")
Cs, epsilons = (0.1, 1, 10, 100), (0.001, 0.01, 0.03)
print("C \\ eps  " + "  ".join(f"{e:>7}" for e in epsilons))
best_svr = None
for C in Cs:
    row = []
    for eps in epsilons:
        m = SVR(kernel="rbf", C=C, epsilon=eps).fit(X["train"], y["train"])
        va = score(m, "valid")[1]
        row.append(va)
        if best_svr is None or va > best_svr[0]:
            best_svr = (va, C, eps)
    print(f"{C:<8} " + "  ".join(f"{v:7.3f}" for v in row))

print(f"\nBest tree on valid: depth {best_tree[1]} (valid R2 {best_tree[0]:.3f})")
print(f"Best SVR on valid: C={best_svr[1]}, epsilon={best_svr[2]} (valid R2 {best_svr[0]:.3f})")

# Test is scored once, for the single chosen setting of each model.
tree = DecisionTreeRegressor(max_depth=best_tree[1], random_state=0).fit(X["train"], y["train"])
svr = SVR(kernel="rbf", C=best_svr[1], epsilon=best_svr[2]).fit(X["train"], y["train"])
print("\n=== Final test scores (chosen on valid) ===")
for name, m in (("tree", tree), ("SVR", svr)):
    rm, r = score(m, "test")
    print(f"{name}: test RMSE={rm:.4f}  test R2={r:.3f}")
