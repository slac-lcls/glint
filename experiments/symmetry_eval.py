"""Cross-symmetry generalization test for the learned peakfinder.

Compares, per crystal symmetry (random cell each trial):
  - classical      : amplitude-ranked peaks
  - learn-1cell    : model trained on the single hexagonal 78/38 cell (memorization
                     probe -- top feature was `len`, which is cell-specific)
  - learn-mixed    : model trained on random cells across all symmetry classes

If learn-1cell collapses on other symmetries but learn-mixed holds, the gains are
real (consistency features) rather than memorized axis lengths. 'monoclinic' and
'triclinic' are the sheared cases.
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

sys.path.insert(0, "..")
from glint import LearnedPeakFinder, index_shot, make_dataset, score, simulate_shot
from glint.features import FEATURE_NAMES
from glint.lattice import SYMMETRIES, random_cell
from glint.peakfind import find_peaks_classical

N_TRIALS = 15
SETTING = dict(n_target=45, pos_sigma=0.001, frac_spurious=0.1)
MIXED_PATH = "detector_mixed.joblib"


def get_mixed():
    if not os.path.exists(MIXED_PATH):
        print("training mixed-symmetry detector (700 random-cell shots)...")
        t = time.time()
        X, y = make_dataset(n_shots=450, symmetry="random", rng=np.random.default_rng(0))
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     n_jobs=-1, random_state=0).fit(X, y)
        joblib.dump(clf, MIXED_PATH)
        print(f"  {X.shape[0]} peaks, {100 * y.mean():.0f}% positive  [{time.time()-t:.0f}s]")
        for nm, im in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1]):
            print(f"    {nm:<12} {im:.3f}")
    return LearnedPeakFinder(joblib.load(MIXED_PATH))


def solve(peakfinder, sym, seed0=5000):
    sv = 0
    for s in range(N_TRIALS):
        rng = np.random.default_rng(seed0 + s)
        shot = simulate_shot(rng=rng, cell=random_cell(rng, sym), **SETTING)
        sv += score(shot, index_shot(shot.g, shot.meta["qmax"],
                                     peakfinder=peakfinder))["solved"]
    return 100 * sv / N_TRIALS


if __name__ == "__main__":
    t0 = time.time()
    mixed = get_mixed()
    single = LearnedPeakFinder(joblib.load("detector_rf.joblib"))

    print(f"\nsolve% at {SETTING}  (random cell per trial)")
    print(f"{'symmetry':<13}{'classical':>10}{'learn-1cell':>12}{'learn-mixed':>12}")
    for sym in SYMMETRIES:
        c = solve(find_peaks_classical, sym)
        l1 = solve(single, sym)
        lm = solve(mixed, sym)
        tag = " (sheared)" if sym in ("monoclinic", "triclinic") else ""
        print(f"{sym:<13}{c:>9.0f}%{l1:>11.0f}%{lm:>11.0f}%{tag}")
    print(f"\nwall time: {time.time() - t0:.0f}s")
