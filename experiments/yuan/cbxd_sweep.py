"""Three-arm sweep over NA (github.com/slac-lcls/glint issue #9's suggested first experiment):

  arm 1 (COM):  cbxd_arm1_com.com_seed_index      -- centroid-only baseline
  arm 2 (PTS):  cbxd_hough_gpu.hough_seed_index    -- accumulator on arc points
  arm 3 (TAN):  cbxd_hough_tangent.hough_seed_index_tangent -- accumulator + tangent vote

FIRST PASS, deliberately reduced scope to fit one sitting: single noise level (2e-4, the value
used throughout this session's validation), n_coarse=1M (vs. the 5M used for the final arm-2/3
validation), ncry=6 crystals/NA. Streak length and #streaks aren't independently controllable in
the current simulate() model -- both are driven by NA (and dmin/cell), so they're reported as
OBSERVED per-NA statistics (median #streaks/crystal, median raw streak length) rather than swept
as independent axes. Extending to more crystals / a noise axis / larger n_coarse is a rerun of
this same script with bigger constants, not a redesign.

NA is a module-level constant in cbxd_joint.py (COSA/SINA derived from it at import time), and
three downstream modules (tangent_validate, cbxd_hough_gpu, and cbxd_joint itself) each hold their
own bound copy from `from cbxd_joint import COSA, SINA` -- set_NA() patches all of them.

  python cbxd_sweep.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, "..")
import cbxd_joint
from cbxd_joint import rand_rot, score, simulate

import cbxd_hough_gpu
from cbxd_hough_gpu import hough_seed_index

import tangent_validate
from tangent_validate import _simulate_grouped

from cbxd_arm1_com import com_seed_index
from cbxd_hough_tangent import hough_seed_index_tangent, streak_reprs_and_tangents

NOISE = 2e-4
SEED = 2
NCRY = 6
N_COARSE = 1_000_000
NA_GRID = (0.010, 0.016, 0.022, 0.028, 0.040)
MIN_KOBS = 5                                # below this, treat as "no usable data" (frac=0, all arms)


def set_NA(na):
    cosa, sina = float(np.cos(na)), float(np.sin(na))
    cbxd_joint.NA = na
    for mod in (cbxd_joint, tangent_validate, cbxd_hough_gpu):
        mod.COSA = cosa
        mod.SINA = sina


def frac_indexed(R, kobs, lab):
    if R is None:
        return 0.0
    return float(score(R, kobs, 0.0025, ret_mask=True)[lab].mean())


def run_cell(na, ncry=NCRY, noise=NOISE, seed=SEED, n_coarse=N_COARSE):
    set_NA(na)
    rng_master = np.random.default_rng(seed)
    fr = {"com": [], "pts": [], "tan": []}
    n_streaks_obs, raw_len_obs = [], []
    for c in range(ncry):
        Rt = rand_rot(rng_master)
        kobs, lab, cents = simulate(Rt, rng_master, noise)
        n_streaks_obs.append(len(cents))
        streaks = _simulate_grouped(Rt, np.random.default_rng(50_000 + c), noise, thin=1)
        raw_len_obs += [len(s["clean"]) for s in streaks]

        if len(kobs) < MIN_KOBS:
            fr["com"].append(0.0); fr["pts"].append(0.0); fr["tan"].append(0.0)
            continue

        R_com = com_seed_index(cents, np.random.default_rng(60_000 + c), n_coarse=n_coarse)
        fr["com"].append(frac_indexed(R_com, kobs, lab))

        R_pts = hough_seed_index(kobs, np.random.default_rng(70_000 + c), n_coarse=n_coarse)
        fr["pts"].append(frac_indexed(R_pts, kobs, lab))

        reprs, tangents = streak_reprs_and_tangents(Rt, noise, np.random.default_rng(80_000 + c))
        R_tan = hough_seed_index_tangent(kobs, reprs, tangents, np.random.default_rng(90_000 + c),
                                         n_coarse=n_coarse)
        fr["tan"].append(frac_indexed(R_tan, kobs, lab))

    def succ(v):
        return int(sum(x > 0.7 for x in v))

    return dict(
        na=na,
        n_streaks_med=float(np.median(n_streaks_obs)),
        raw_len_med=float(np.median(raw_len_obs)) if raw_len_obs else float("nan"),
        succ_com=succ(fr["com"]), succ_pts=succ(fr["pts"]), succ_tan=succ(fr["tan"]),
        med_com=100 * float(np.median(fr["com"])),
        med_pts=100 * float(np.median(fr["pts"])),
        med_tan=100 * float(np.median(fr["tan"])),
    )


if __name__ == "__main__":
    print(f"three-arm NA sweep  noise={NOISE:.0e}  ncry={NCRY}  n_coarse={N_COARSE}  seed={SEED}")
    print(f"{'NA':>6} {'#streaks':>9} {'rawlen':>7} "
          f"{'COM':>7} {'PTS':>7} {'TAN':>7} {'COM%':>6} {'PTS%':>6} {'TAN%':>6}")
    for na in NA_GRID:
        t0 = time.perf_counter()
        r = run_cell(na)
        dt = time.perf_counter() - t0
        print(f"{na:6.3f} {r['n_streaks_med']:9.1f} {r['raw_len_med']:7.1f} "
              f"{r['succ_com']:>4d}/{NCRY} {r['succ_pts']:>4d}/{NCRY} {r['succ_tan']:>4d}/{NCRY} "
              f"{r['med_com']:6.0f} {r['med_pts']:6.0f} {r['med_tan']:6.0f}   ({dt:.0f}s)", flush=True)
