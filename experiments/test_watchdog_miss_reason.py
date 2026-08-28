"""Why did the watchdog rescue find 0/42, when 10 of streaming's failures WOULD be recovered by
blind-top1 against the exact LYSO cell? Watchdog's actual rescue condition is
same_lattice(candidate, driver.Mc) AND _inliers(q, candidate) >= min_inliers -- checked against
driver's OWN (empirically imprecise, ~0.3A off) locked cell, not the exact reference. This checks,
for exactly those 10 frames, which half of that AND fails: the same_lattice comparison (imprecision
in driver.Mc itself) or the inlier-count threshold.

  python test_watchdog_miss_reason.py
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, "/sdf/home/s/smarches/glint_streamfix_wt")
sys.path.insert(0, "/sdf/home/s/smarches/glint_streamfix_wt/experiments")
import glint.glint_fast as gf
from glint.glint_fast import matched_strict, index_blind_nbest
from glint.multishot import same_lattice
import glint.replica_gpu_batch as rgb
import glint.stream_driver as sd

LYSO = gf.LYSO
GATE_FRAC, GATE_MIN = gf.GATE_FRAC, gf.GATE_MIN
FRAMES_PATH = "/sdf/home/s/smarches/glint_streamfix_wt/experiments/frames_cxidb_clean.txt"


def strict_gate(M, q, truth):
    # matched_strict, NOT the configurable matched(): under QDIST=1 matched() switches to a
    # reciprocal-distance ball and stops consulting GATE_TOL, which would silently apply the strict
    # GATE_FRAC/GATE_MIN thresholds to counts from a different matching rule (the defect Copilot
    # found inside gpass() on glint#170). Identical at the shipped QDIST=0 default. gpass() itself
    # is not usable here: it hard-codes same_lattice against LYSO, and this gate runs against an
    # arbitrary truth cell (driver.Mc).
    if M is None:
        return False
    m = matched_strict(M, q)
    return (m / len(q) >= GATE_FRAC) and (m >= GATE_MIN) and same_lattice(M, truth)


def push_blind_q(driver, qq):
    driver.n_pushed += 1
    if len(qq) >= driver.min_peaks:
        nb = driver._blind_index(qq, driver.warmup_nbest)
        driver._rc.add_frame([c for c, _ in nb])
        driver.n_warmup += 1
        if driver._warmup_buf is not None:
            driver._warmup_buf.append(qq)
    Mc, sup, _ = driver._rc.verdict()
    if Mc is not None:
        driver._lock(Mc, sup, standardize=True)


def main():
    t0 = time.time()
    frames = list(gf.load(FRAMES_PATH))
    n = len(frames)

    N = 400; pix_mm = 0.1; dist_mm = 100.0; wave_A = 1.0
    panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
                   res=1.0 / (pix_mm / 1000.0), cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5),
                   coffset=0.0, min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
    clen = dist_mm / 1000.0
    drv = sd.StreamDriver(None, panels, clen, wave_A, (N, N), dtype=np.uint16,
                          B=20, dmin=2.0, tol=0.002, warmup_nbest=3)
    warmup_idx = []
    for i, qq in enumerate(frames):
        qq = np.asarray(qq, float)
        if drv._blind:
            warmup_idx.append(i)
            push_blind_q(drv, qq)
        else:
            slot = drv._n
            drv._q[slot] = qq
            drv._n += 1; drv.n_pushed += 1
            if drv._n == drv.B:
                drv.flush()
    drv.flush()

    warmup_set = set(warmup_idx)
    post_lock_idx = [i for i in range(n) if i not in warmup_set]
    qs = [np.asarray(frames[i], float) for i in post_lock_idx]
    Ms = rgb.index_fused(qs, drv.Mc, B=len(qs))
    streaming_ok = {i: strict_gate(np.asarray(M, float) if M is not None else None, frames[i], drv.Mc)
                    for i, M in zip(post_lock_idx, Ms)}
    streaming_fail = [i for i in post_lock_idx if not streaming_ok[i]]

    nbest = 3
    n_same_lattice_fails = n_inlier_fails = n_both_pass = n_no_top1_match_lyso = 0
    for i in streaming_fail:
        q = frames[i]
        nb = index_blind_nbest(q, nbest)
        c, _ = nb[0]                                    # top-1 only, matching the "blind_top1" category
        c = np.asarray(c, float)
        matches_lyso = same_lattice(c, LYSO)
        if not matches_lyso:
            n_no_top1_match_lyso += 1
            continue
        matches_drv = same_lattice(c, drv.Mc)
        ninl = int((np.abs((q @ c) - np.round(q @ c)).max(1) < gf.GATE_TOL).sum())
        inl_ok = ninl >= GATE_MIN
        if matches_drv and inl_ok:
            n_both_pass += 1
        elif not matches_drv:
            n_same_lattice_fails += 1
            print(f"  frame {i}: top-1 blind candidate matches EXACT LYSO but NOT driver.Mc "
                  f"(same_lattice fails against the empirically-derived cell)  inliers={ninl}")
        elif not inl_ok:
            n_inlier_fails += 1
            print(f"  frame {i}: top-1 blind candidate matches BOTH cells but inlier count {ninl} < {GATE_MIN}")

    print(f"\nof the {n_no_top1_match_lyso + n_same_lattice_fails + n_inlier_fails + n_both_pass} frames "
          f"whose top-1 blind candidate matches exact LYSO:")
    print(f"  top-1 does NOT even match exact LYSO (different failure mode entirely): {n_no_top1_match_lyso}")
    print(f"  matches LYSO, but same_lattice(candidate, driver.Mc) FAILS (cell-precision sensitive): "
          f"{n_same_lattice_fails}")
    print(f"  matches LYSO AND driver.Mc, but inlier count < {GATE_MIN}: {n_inlier_fails}")
    print(f"  matches driver.Mc AND clears inlier count (watchdog SHOULD have rescued this): {n_both_pass}")

    print(f"\nDONE elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
