"""Train the learned peakfinder (per-peak true-axis classifier) and save it.

Generates a labeled feature table from simulated shots, trains a RandomForest,
reports held-out precision/recall (we care about ranking true axes high) and
feature importances, and saves the model to detector_rf.joblib.

NOTE: training and eval use the same default cell (random orientation + noise);
cross-cell generalization is future work.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, "..")
from fftindex import make_dataset
from fftindex.features import FEATURE_NAMES

if __name__ == "__main__":
    t0 = time.time()
    X, y = make_dataset(n_shots=500, rng=np.random.default_rng(0))
    print(f"dataset: {X.shape[0]} candidate peaks, "
          f"{100 * y.mean():.1f}% positive (true axis)  [{time.time() - t0:.0f}s]")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0,
                                          stratify=y)
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 n_jobs=-1, random_state=0)
    clf.fit(Xtr, ytr)

    prob = clf.predict_proba(Xte)[:, 1]
    print("\nheld-out classification (threshold 0.5):")
    print(classification_report(yte, prob > 0.5, target_names=["false", "true-axis"]))
    print(f"average precision (true-axis): {average_precision_score(yte, prob):.3f}")
    print("\nfeature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                            key=lambda t: -t[1]):
        print(f"  {name:<12} {imp:.3f}")

    joblib.dump(clf, "detector_rf.joblib")
    print(f"\nsaved detector_rf.joblib  [{time.time() - t0:.0f}s total]")
