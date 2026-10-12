"""Fit the chosen SVR on the training features and save it for assess_image."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def main() -> None:
    """Train on ``results/features.npz`` (train split only) and write ``models/svr_model.joblib``."""
    d = np.load("results/features.npz")
    scaler = StandardScaler().fit(d["X_train"])
    model = SVR(kernel="rbf", C=1, epsilon=0.03).fit(scaler.transform(d["X_train"]), d["y_train"])
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model}, "models/svr_model.joblib")
    print("saved models/svr_model.joblib")


if __name__ == "__main__":
    main()
