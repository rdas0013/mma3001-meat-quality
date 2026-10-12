"""Pass/fail view: threshold predicted area, with thresholds chosen on valid only."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from meatqa.metrics import summarise

d = np.load("results/features.npz")
scaler = StandardScaler().fit(d["X_train"])
model = SVR(kernel="rbf", C=1, epsilon=0.03).fit(scaler.transform(d["X_train"]), d["y_train"])
pred = {s: model.predict(scaler.transform(d[f"X_{s}"])) for s in ("valid", "test")}
true_fail = {s: (d[f"y_{s}"] > 0).astype(int) for s in ("valid", "test")}

for s in ("valid", "test"):
    print(f"{s}: {true_fail[s].sum()} fail, {(1 - true_fail[s]).sum()} pass")

print("\nClipping check (R2 on test):")
from meatqa.regression_metrics import r2
print(f"  raw {r2(d['y_test'], pred['test']):.4f}   clipped at 0: {r2(d['y_test'], np.clip(pred['test'], 0, None)):.4f}")

print("\nThreshold sweep on VALID (fail if predicted area > t):")
print(f"{'t':>6} {'acc':>6} {'pass_recall':>12} {'fail_recall':>12}")
best = None
for t in np.arange(0.0, 0.101, 0.01):
    s = summarise(true_fail["valid"], (pred["valid"] > t).astype(int))
    print(f"{t:6.2f} {s['accuracy']:6.3f} {s['pass_recall']:12.3f} {s['fail_recall']:12.3f}")
    # Balanced accuracy, so that the always-fail shortcut is not rewarded.
    score = (s["pass_recall"] + s["fail_recall"]) / 2
    if best is None or score > best[0]:
        best = (score, t)

t = best[1]
print(f"\nChosen threshold (best balanced accuracy on valid): {t:.2f}")
for name, p in (("model", (pred["test"] > t).astype(int)), ("always fail", np.ones(80, dtype=int))):
    s = summarise(true_fail["test"], p)
    print(f"test {name:11s}: acc={s['accuracy']:.3f} pass_recall={s['pass_recall']:.3f} "
          f"fail_recall={s['fail_recall']:.3f}  tp={s['tp']} fp={s['fp']} tn={s['tn']} fn={s['fn']}")
