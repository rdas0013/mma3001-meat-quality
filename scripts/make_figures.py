"""Make the report figures from saved features and the chosen model settings.

Reads ``results/features.npz`` and writes PNGs to ``results/figures/``.
Models and settings match the experiment scripts (tree depth 6, SVR C=1,
epsilon=0.03). The robustness chart uses numbers copied from
``results/robustness.txt`` (re-running it needs images), so update the
dictionary below if that experiment is repeated.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from meatqa.metrics import summarise
from meatqa.regression_metrics import r2

OUT = Path("results/figures")
SPLITS = ("train", "valid", "test")
ROBUSTNESS = {  # test R2, copied from results/robustness.txt
    "original": 0.673, "bright x0.7": -0.492, "bright x0.85": 0.352,
    "bright x1.15": 0.660, "bright x1.3": 0.525, "blur r2": -0.067, "blur r5": -0.002,
}


def load() -> tuple[dict, dict]:
    """Load features, standardise with a train-only scaler, return (X, y) dicts."""
    d = np.load("results/features.npz")
    scaler = StandardScaler().fit(d["X_train"])
    X = {s: scaler.transform(d[f"X_{s}"]) for s in SPLITS}
    y = {s: d[f"y_{s}"] for s in SPLITS}
    return X, y


def test_predictions(X: dict, y: dict) -> dict[str, np.ndarray]:
    """Fit the reference and three models on train; return their test predictions."""
    models = {
        "Linear": LinearRegression(),
        "Tree (depth 6)": DecisionTreeRegressor(max_depth=6, random_state=0),
        "SVR (C=1, eps=0.03)": SVR(kernel="rbf", C=1, epsilon=0.03),
    }
    preds = {"Mean reference": np.full(len(y["test"]), y["train"].mean())}
    for name, m in models.items():
        preds[name] = m.fit(X["train"], y["train"]).predict(X["test"])
    return preds


def save(fig, name: str) -> None:
    """Save a figure as a PNG in ``OUT`` and close it."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def fig_pred_vs_true(preds: dict, y_test: np.ndarray) -> None:
    """Predicted vs true defect area on test, one panel per model."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True, sharey=True)
    for ax, (name, p) in zip(axes, preds.items()):
        ax.scatter(y_test, p, s=14, alpha=0.7)
        ax.plot([0, 0.3], [0, 0.3], "k--", lw=1)
        ax.set_title(f"{name}\nR2 = {r2(y_test, p):.3f}", fontsize=9)
        ax.set_xlabel("True area fraction")
    axes[0].set_ylabel("Predicted area fraction")
    save(fig, "pred_vs_true.png")


def fig_residuals(preds: dict, y_test: np.ndarray) -> None:
    """Residual (pred - true) against true area on test, one panel per model."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True, sharey=True)
    for ax, (name, p) in zip(axes, preds.items()):
        ax.scatter(y_test, p - y_test, s=14, alpha=0.7)
        ax.axhline(0, color="k", lw=1)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("True area fraction")
    axes[0].set_ylabel("Prediction error (pred - true)")
    save(fig, "residuals.png")


def fig_tree_depth(X: dict, y: dict) -> None:
    """Train and valid R2 against tree depth."""
    depths = (2, 3, 4, 5, 6, 8, 10, 15)
    tr, va = [], []
    for k in depths:
        m = DecisionTreeRegressor(max_depth=k, random_state=0).fit(X["train"], y["train"])
        tr.append(r2(y["train"], m.predict(X["train"])))
        va.append(r2(y["valid"], m.predict(X["valid"])))
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(depths, tr, "o-", label="train")
    ax.plot(depths, va, "s-", label="valid")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("R2")
    ax.legend()
    ax.set_title("Decision tree: overfitting with depth")
    save(fig, "tree_depth.png")


def fig_svr_heatmap(X: dict, y: dict) -> None:
    """Valid R2 for the SVR C x epsilon grid."""
    Cs, eps = (0.1, 1, 10, 100), (0.001, 0.01, 0.03)
    grid = np.array([[r2(y["valid"], SVR(C=C, epsilon=e).fit(X["train"], y["train"]).predict(X["valid"]))
                      for e in eps] for C in Cs])
    fig, ax = plt.subplots(figsize=(5, 3.8))
    im = ax.imshow(grid, cmap="viridis")
    ax.set_xticks(range(len(eps)), [str(e) for e in eps])
    ax.set_yticks(range(len(Cs)), [str(c) for c in Cs])
    ax.set_xlabel("epsilon")
    ax.set_ylabel("C")
    for i in range(len(Cs)):
        for j in range(len(eps)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, label="valid R2")
    ax.set_title("SVR sensitivity (valid R2)")
    save(fig, "svr_heatmap.png")


def fig_bootstrap(preds: dict, y_test: np.ndarray, n_boot: int = 2000, seed: int = 0) -> None:
    """Test R2 per model with bootstrap 95% intervals (same resampling as bootstrap_ci.py)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_test), size=(n_boot, len(y_test)))
    names, pts, los, his = [], [], [], []
    for name, p in preds.items():
        boot = [r2(y_test[i], p[i]) for i in idx]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        names.append(name)
        pts.append(r2(y_test, p))
        los.append(lo)
        his.append(hi)
        print(f"{name}: R2 {pts[-1]:.3f} [{lo:.3f}, {hi:.3f}]")
    pts, los, his = map(np.array, (pts, los, his))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(names, pts, yerr=[pts - los, his - pts], capsize=4)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("Test R2 (95% bootstrap CI)")
    plt.setp(ax.get_xticklabels(), rotation=15, fontsize=8)
    save(fig, "bootstrap_r2.png")


def fig_threshold(X: dict, y: dict) -> None:
    """Pass and fail recall on valid against the decision threshold."""
    model = SVR(C=1, epsilon=0.03).fit(X["train"], y["train"])
    pred, true_fail = model.predict(X["valid"]), (y["valid"] > 0).astype(int)
    ts = np.arange(0.0, 0.101, 0.01)
    rows = [summarise(true_fail, (pred > t).astype(int)) for t in ts]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(ts, [r["pass_recall"] for r in rows], "o-", label="pass recall")
    ax.plot(ts, [r["fail_recall"] for r in rows], "s-", label="fail recall")
    ax.axvline(0.04, color="k", ls="--", lw=1, label="chosen 0.04")
    ax.set_xlabel("Threshold on predicted area")
    ax.set_ylabel("Recall (valid)")
    ax.legend()
    save(fig, "threshold_sweep.png")


def fig_robustness() -> None:
    """Test R2 under brightness and blur perturbations (numbers from robustness.txt)."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(list(ROBUSTNESS), list(ROBUSTNESS.values()))
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("Test R2")
    plt.setp(ax.get_xticklabels(), rotation=25, fontsize=8)
    save(fig, "robustness.png")


def main() -> None:
    """Build every figure."""
    X, y = load()
    preds = test_predictions(X, y)
    fig_pred_vs_true(preds, y["test"])
    fig_residuals(preds, y["test"])
    fig_tree_depth(X, y)
    fig_svr_heatmap(X, y)
    fig_bootstrap(preds, y["test"])
    fig_threshold(X, y)
    fig_robustness()


if __name__ == "__main__":
    main()
