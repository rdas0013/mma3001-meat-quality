"""Pass/fail view: threshold predicted area, with the threshold chosen on valid only.

Reads ``results/features.npz``. A tray is truly "fail" if its label file has
at least one box (true area > 0). Prints a threshold sweep on valid, then
scores test once at the chosen threshold against an always-fail reference.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from meatqa.metrics import summarise
from meatqa.regression_metrics import r2


def main() -> None:
    """Sweep thresholds on valid, pick the best balanced accuracy, score test."""
    d = np.load("results/features.npz")
    scaler = StandardScaler().fit(d["X_train"])
    model = SVR(kernel="rbf", C=1, epsilon=0.03).fit(scaler.transform(d["X_train"]), d["y_train"])
    pred = {s: model.predict(scaler.transform(d[f"X_{s}"])) for s in ("valid", "test")}
    true_fail = {s: (d[f"y_{s}"] > 0).astype(int) for s in ("valid", "test")}

    for s in ("valid", "test"):
        print(f"{s}: {true_fail[s].sum()} fail, {(1 - true_fail[s]).sum()} pass")

    print("\nClipping check (R2 on test):")
    print(f"  raw {r2(d['y_test'], pred['test']):.4f}   "
          f"clipped at 0: {r2(d['y_test'], np.clip(pred['test'], 0, None)):.4f}")

    print("\nThreshold sweep on VALID (fail if predicted area > t):")
    print(f"{'t':>6} {'acc':>6} {'pass_recall':>12} {'fail_recall':>12}")
    best = None
    for t in np.arange(0.0, 0.101, 0.01):
        s = summarise(true_fail["valid"], (pred["valid"] > t).astype(int))
        print(f"{t:6.2f} {s['accuracy']:6.3f} {s['pass_recall']:12.3f} {s['fail_recall']:12.3f}")
        # Balanced accuracy, so the always-fail shortcut is not rewarded.
        score = (s["pass_recall"] + s["fail_recall"]) / 2
        if best is None or score > best[0]:
            best = (score, t)

    t = best[1]
    print(f"\nChosen threshold (best balanced accuracy on valid): {t:.2f}")
    for name, p in (("model", (pred["test"] > t).astype(int)), ("always fail", np.ones(80, dtype=int))):
        s = summarise(true_fail["test"], p)
        print(f"test {name:11s}: acc={s['accuracy']:.3f} pass_recall={s['pass_recall']:.3f} "
              f"fail_recall={s['fail_recall']:.3f}  tp={s['tp']} fp={s['fp']} tn={s['tn']} fn={s['fn']}")


if __name__ == "__main__":
    main()
