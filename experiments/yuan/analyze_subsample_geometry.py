"""Does the SPECIFIC geometry of which streaks got kept (not just how many) predict success,
within a fixed (NA, streak count) condition? Follow-up to subsample_streaks_na028.py: rather than
launching another multi-NA experiment blind, first squeeze more out of the data already in
results_subsample.jsonl.

Feature: min_sv, the smallest singular value of the matrix of kept streaks' UNIT G-VECTOR
directions (the actual reciprocal-lattice vectors, R @ B @ h -- not the streak's detector-position
direction, kout, which is constrained near the forward-scattering hemisphere and so is nearly
degenerate for every crystal regardless of orientation; using kout was a real bug caught while
building this, worth remembering if extending this analysis). min_sv ~ 0 means the kept G's are
nearly coplanar/degenerate; larger means they span 3-D well (independent constraints).

Finding (N=20 crystals per condition, correlational): at n_target=4, HIGHER min_sv (more
independent, well-spread constraints) correlates with LOWER success (PTS rho=-0.68 p=0.001, TAN
rho=-0.48 p=0.031) -- counterintuitive. At n_target=11 the sign FLIPS for PTS (rho=+0.58 p=0.007);
TAN shows no relationship there (p=0.77). Plausible mechanism, not proven: the search here is
brute-force random-orientation accumulation, not a closed-form solve -- at very few streaks,
well-spread G's likely carve a sharper-but-smaller true-orientation basin (better information once
found, harder for random sampling to land in by chance); at more streaks there's enough redundancy
that the search finds the right neighborhood regardless, and well-spread constraints then help
precision instead of hurting discoverability. Also: gmag_cv (spread of |G| across resolution
shells) correlates negatively with PTS success at n_target=11 (rho=-0.49, p=0.03) -- a weaker
secondary lead, not yet strong enough to build on.

  python analyze_subsample_geometry.py
"""
import json
import os
import sys

import numpy as np
import scipy.stats as st

sys.path.insert(0, "..")
from cbxd_sweep import set_NA
from generate_dataset_grid import get_orientation_pool
from subsample_streaks_na028 import NA as NA028, NOISE, N_TARGETS
from tangent_validate import _simulate_grouped

LOG = os.path.join(os.path.dirname(__file__), "results_subsample.jsonl")


def kept_G(Rt, n_target, seed):
    """Same subsampling as subsample_streaks_na028.build_subsampled, but returns the kept
    streaks' G-vectors (crystal-frame reciprocal lattice vectors, lab frame) instead of building
    a kobs/cents dataset -- this is a read-only re-derivation for analysis, not a new search."""
    rng_pick = np.random.default_rng(seed)
    streaks6 = _simulate_grouped(Rt, np.random.default_rng(seed + 1), NOISE, thin=6)
    n_total = len(streaks6)
    idx = np.arange(n_total) if n_total <= n_target else rng_pick.choice(n_total, n_target, replace=False)
    return np.array([streaks6[i]["G"] for i in idx])


def geometry_features(G):
    dirs = G / np.linalg.norm(G, axis=1, keepdims=True)
    cos = dirs @ dirs.T
    iu = np.triu_indices(len(dirs), 1)
    max_cos = float(np.max(np.abs(cos[iu])))
    mean_cos = float(np.mean(np.abs(cos[iu])))
    min_sv = float(np.linalg.svd(dirs, compute_uv=False)[-1]) if len(dirs) >= 3 else 0.0
    gmag_cv = float(np.std(np.linalg.norm(G, axis=1)) / np.mean(np.linalg.norm(G, axis=1)))
    return dict(max_cos=max_cos, mean_cos=mean_cos, min_sv=min_sv, gmag_cv=gmag_cv)


def run():
    set_NA(NA028)
    Rts = get_orientation_pool(NA028)
    by_key = {}
    for line in open(LOG):
        r = json.loads(line)
        by_key.setdefault((r["n_target"], r["i"], r["arm"]), r["frac"])

    for n_target in N_TARGETS:
        print(f"=== n_target={n_target}, subsampled NA={NA028} (G-vector geometry) ===")
        rows = []
        for i, Rt in enumerate(Rts):
            seed = 10_000 * n_target + i
            feat = geometry_features(kept_G(Rt, n_target, seed))
            feat.update(i=i, pts=by_key.get((n_target, i, "PTS"), np.nan),
                       tan=by_key.get((n_target, i, "TAN"), np.nan))
            rows.append(feat)

        print(f"{'i':>3} {'PTS':>6} {'TAN':>6} {'max_cos':>8} {'mean_cos':>9} {'min_sv':>8} {'gmag_cv':>8}")
        for r in rows:
            print(f"{r['i']:3d} {r['pts']:6.2f} {r['tan']:6.2f} {r['max_cos']:8.3f} "
                  f"{r['mean_cos']:9.3f} {r['min_sv']:8.3f} {r['gmag_cv']:8.3f}")

        pts_arr = np.array([r["pts"] for r in rows])
        tan_arr = np.array([r["tan"] for r in rows])
        for name in ("max_cos", "mean_cos", "min_sv", "gmag_cv"):
            arr = np.array([r[name] for r in rows])
            rp, rt = st.spearmanr(arr, pts_arr), st.spearmanr(arr, tan_arr)
            print(f"  Spearman({name}, PTS) = {rp.statistic:+.3f} (p={rp.pvalue:.3f})   "
                  f"Spearman({name}, TAN) = {rt.statistic:+.3f} (p={rt.pvalue:.3f})")
        print()


if __name__ == "__main__":
    run()
