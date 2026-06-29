"""Measure fftindex improvements on the SAME real lyso frames xgandalf got 100% on.

Levers (vs the 70%/78% baseline):
  #1 learned peakfinder  -- RF re-ranker (sklearn) trained on synthetic lyso shots,
                            re-orders candidates so true axes survive into the top-k
  #3 widen + diff-seed   -- topk=20, seed_diff=True (speed is free now -> search more)
  #4 localize            -- grid-free BFGS refinement of candidate vectors
     lower min_inlier_frac to accept marginal sparse frames

All scored with same_lattice vs the lysozyme cell -- identical metric to compare.py.
"""
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, "../..")
from fftindex import index_shot
from fftindex.dataset import label_true_axis
from fftindex.detector import LearnedPeakFinder
from fftindex.features import peak_features
from fftindex.lattice import cell_to_Ar
from fftindex.multishot import same_lattice
from fftindex.peakfind import find_peaks_classical
from fftindex.simulate import simulate_shot
from fftindex.transform import estimate_grid_n, fft_volume

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
LYSO_CELL = (79.02, 79.02, 37.98, 90, 90, 90)
NMAX = 176                                          # cap (lyso fits ~144) for speed


def read_frames(path):
    lines = open(path).read().split("\n")
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].startswith("FRAME"):
            i += 1; continue
        _, fid, npk = lines[i].split(); npk = int(npk)
        q = np.array([[float(x) for x in lines[i + 1 + j].split()] for j in range(npk)])
        frames.append((int(fid), q)); i += 1 + npk
    return frames


def train_rf(n_shots=500, seed=0):
    """RF re-ranker on synthetic lysozyme shots matched to the real regime
    (sparse counts, geometry-residual jitter, some spurious)."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for _ in range(n_shots):
        nt = int(rng.integers(12, 90))
        sh = simulate_shot(rng=rng, cell=LYSO_CELL, dmin=2.0, n_target=nt,
                           pos_sigma=rng.uniform(0, 0.003), frac_spurious=rng.uniform(0, 0.15))
        qmax = sh.meta["qmax"]
        vol, x = fft_volume(sh.g, qmax, n=min(NMAX, estimate_grid_n(sh.g, qmax)))
        vecs, _ = find_peaks_classical(vol, x, min_len=3.0)
        vecs = vecs[:30]
        if len(vecs) < 3:
            continue
        Xs.append(peak_features(sh.g, vecs, qmax)); ys.append(label_true_axis(sh.M, vecs))
    X, y = np.vstack(Xs), np.concatenate(ys)
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=0)
    clf.fit(X, y)
    return clf, y.mean()


frames = read_frames(sys.argv[1] if len(sys.argv) > 1 else "frames.txt")
N = len(frames)
print(f"frames: {N}  (xgandalf baseline: 100% blind & known)\n")

# default n_max=256 (the escalation the 70% baseline relies on -- DON'T cap it).
configs = {
    "baseline (defaults)":   dict(),
    "thr 0.5":               dict(min_inlier_frac=0.5),
    "localize":              dict(localize=True),
    "thr0.5 + localize":     dict(min_inlier_frac=0.5, localize=True),
}
for name, cfg in configs.items():
    ok = 0.0; t = 0.0
    for fid, q in frames:
        qmax = float(np.linalg.norm(q, axis=1).max())
        t0 = time.perf_counter(); r = index_shot(q, qmax, **cfg); t += time.perf_counter() - t0
        ok += r.M is not None and same_lattice(r.M, LYSO)
    print(f"  {name:22s}: {int(ok):2d}/{N} ({100*ok/N:3.0f}%)   {1e3*t/N:6.0f} ms/frame", flush=True)
