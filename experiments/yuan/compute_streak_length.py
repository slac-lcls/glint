import sys
import numpy as np
sys.path.insert(0, "..")

from cbxd_sweep import set_NA
from generate_dataset_grid import NA_GRID, orientation_pool_path
from tangent_validate import _simulate_grouped

out = {}
for na in NA_GRID:
    set_NA(na)
    d = np.load(orientation_pool_path(na))
    n = len(d["n_streaks"])
    Rts = [d[f"Rt_{i:03d}"] for i in range(n)]
    lens = []
    for i, Rt in enumerate(Rts):
        streaks = _simulate_grouped(Rt, np.random.default_rng(500_000 + i), 2e-4, thin=1)
        lens += [len(s["clean"]) for s in streaks]
    lens = np.array(lens)
    out[na] = lens
    print(f"na={na}  n_streak_samples={len(lens)}  median_raw_len={np.median(lens):.1f}  "
          f"p25={np.percentile(lens,25):.1f}  p75={np.percentile(lens,75):.1f}")

np.savez("streak_length_by_na.npz", **{f"na_{na}": v for na, v in out.items()})
