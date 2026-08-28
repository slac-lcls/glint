"""Measure the fractional miss-gate (min_inlier_frac) two ways, to pick a defensible default.

PART 1 -- COST, on the real 120-frame cxidb set: how many genuinely-good frames does a fractional
bar reject? Isolates the gate itself (index against the locked cell, then evaluate _fits at each
threshold) rather than the integrate path, which this q-vector dataset cannot exercise. Cross-checks
each rejection against the strict research gate (same_lattice AND matched_frac>=25% AND >=10 refl),
so a rejection is scored as GOOD (the frame was failing anyway) or COSTLY (a frame that would have
passed the research bar).

PART 2 -- BENEFIT, on a synthetic sample change at the DENSE peak count that defeated the count gate:
cell-B frames were collecting a median 84 chance inliers against cell A, clearing every count gate,
so nothing missed and no relock could ever fire. Does the fraction restore detection?

  python test_inlier_frac_gate.py
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import glint.glint_fast as gf
from glint.glint_fast import matched
from glint.lattice import cell_to_Ar
from glint.multishot import same_lattice
import glint.replica_gpu_batch as rgb
import glint.stream_driver as sd

LYSO = gf.LYSO
FRAMES = os.path.join(os.path.dirname(HERE), "experiments", "frames_cxidb_clean.txt")
if not os.path.exists(FRAMES):
    FRAMES = "/sdf/home/s/smarches/glint_streamfix_wt/experiments/frames_cxidb_clean.txt"
FRACS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35]
fails = []


def check(name, cond, msg=""):
    print(f"  {name:40s}: {'PASS' if cond else 'FAIL'}   {msg}")
    if not cond:
        fails.append(name)


def strict(M, q):
    if M is None or not same_lattice(M, LYSO):
        return False
    m = matched(M, q)
    return m / len(q) >= 0.25 and m >= 10


def push_blind_q(drv, qq):
    drv.n_pushed += 1
    if len(qq) >= drv.min_peaks:
        nb = drv._blind_index(qq, drv.warmup_nbest)
        drv._rc.add_frame([c for c, _ in nb])
        drv.n_warmup += 1
    Mc, sup, _ = drv._rc.verdict()
    if Mc is not None:
        drv._lock(Mc, sup, standardize=True)


def part1():
    print("=== PART 1: cost on real cxidb (gate isolated) ===")
    frames = list(gf.load(FRAMES))
    N = 400
    panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
                   res=1.0 / (0.1 / 1000.0), cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5),
                   coffset=0.0, min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
    # DEFAULT config: adaptive_relock=False, so min_inliers falls back to min_peaks. This is the
    # single-cell path, which until now applied no ingest test at all.
    drv = sd.StreamDriver(None, panels, 0.1, 1.0, (N, N), dtype=np.uint16, B=20, dmin=2.0,
                          tol=0.002, warmup_nbest=3)
    post = []
    for i, qq in enumerate(frames):
        qq = np.asarray(qq, float)
        if drv._blind:
            push_blind_q(drv, qq)
        else:
            post.append(i)
    qs = [np.asarray(frames[i], float) for i in post]
    Ms = rgb.index_fused(qs, drv.Mc, B=len(qs))
    Ms = [np.asarray(M, float) if M is not None else None for M in Ms]
    good = [strict(M, q) for M, q in zip(Ms, qs)]            # would clear the RESEARCH bar
    print(f"  locked_after={drv.locked_after}  post-lock frames={len(qs)}  "
          f"clear the strict research gate: {sum(good)}")

    print(f"\n  {'frac':>6s} {'accepted':>9s} {'rejected':>9s} {'of which COSTLY':>16s}  "
          f"{'(a good frame lost)':>20s}")
    costly_at = {}
    for fr in FRACS:
        drv.min_inlier_frac = fr
        acc = [M is not None and abs(np.linalg.det(M)) >= 1.0 and drv._fits(q, M)
               for M, q in zip(Ms, qs)]
        rej = len(qs) - sum(acc)
        costly = sum(1 for a, g in zip(acc, good) if g and not a)
        costly_at[fr] = costly
        print(f"  {fr:6.2f} {sum(acc):9d} {rej:9d} {costly:16d}")
    drv.min_inlier_frac = 0.0

    check("frac<=0.10 costs no research-gate frame", all(costly_at[f] == 0 for f in (0.02, 0.05, 0.10)),
          f"costly at 0.02/0.05/0.10 = {[costly_at[f] for f in (0.02,0.05,0.10)]}")
    check("frac=0.0 reproduces the count gate", costly_at[0.0] == 0)
    return costly_at


def part2():
    print("\n=== PART 2: benefit -- does relock fire at the density that defeated the count gate? ===")
    from test_stream_subruns import frames_for, PAN, CLEN, LAM, NPX, CELL_A, CELL_B
    import test_stream_subruns as tss
    tss.NPEAKS = 10 ** 6                      # DISABLE subsampling -> the dense regime that failed
    imgs = frames_for(CELL_A, 40, 11) + frames_for(CELL_B, 40, 22)

    out = {}
    for fr in (0.0, 0.10, 0.20):
        drv = sd.StreamDriver(None, PAN, CLEN, LAM, (NPX, NPX), dtype=np.uint16, B=8, dmin=2.5,
                              tol=0.004, adaptive_relock=True, min_inliers=30, min_inlier_frac=fr,
                              warmup_nbest=3)
        for f in imgs:
            drv.push(f)
        drv.flush()
        s = drv.stats()
        out[fr] = (s.get("n_relock", 0), s.get("n_cells", 1), s.get("integrated", 0))
        print(f"  min_inlier_frac={fr:.2f}: n_relock={out[fr][0]}  n_cells={out[fr][1]}  "
              f"integrated={out[fr][2]}")

    check("count gate alone MISSES the sample change", out[0.0][1] == 1,
          f"n_cells={out[0.0][1]} -- the documented failure, reproduced")
    check("fractional gate DETECTS the sample change", out[0.10][1] >= 2 and out[0.20][1] >= 2,
          f"n_cells at 0.10/0.20 = {out[0.10][1]}/{out[0.20][1]}")


def part3():
    """The single-cell path (adaptive_relock=False) must now REFUSE a frame that does not fit the
    locked cell. Previously it ran with gate=False and integrated anything with |det| >= 1, so a
    wholly wrong registration still contributed reflections to the merge. Locking the driver onto the
    WRONG cell makes that unambiguous: every frame is cell A, the driver is told cell B."""
    print("\n=== PART 3: does the single-cell path gate at all? (driver locked to the WRONG cell) ===")
    from test_stream_subruns import frames_for, PAN, CLEN, LAM, NPX, CELL_A, CELL_B
    imgs = frames_for(CELL_A, 16, 7)
    MB = cell_to_Ar(*CELL_B)

    out = {}
    for fr in (None, 0.0, 0.10, 0.15, 0.20, 0.25):
        kw = dict(min_inliers=1, min_inlier_frac=0.0) if fr is None else dict(min_inlier_frac=fr)
        drv = sd.StreamDriver(MB, PAN, CLEN, LAM, (NPX, NPX), dtype=np.uint16, B=8, dmin=2.5,
                              tol=0.004, **kw)
        for f in imgs:
            drv.push(f)
        drv.flush()
        s = drv.stats()
        out[fr] = (s["integrated"], s["gate_rejected"])
        lab = "ungated (old behaviour)" if fr is None else f"min_inlier_frac={fr:.2f}"
        print(f"  {lab:24s}: integrated={s['integrated']:3d}  gate_rejected={s['gate_rejected']:3d}")

    check("ungated arm integrates every misfit", out[None][0] == len(imgs) and out[None][1] == 0,
          f"integrated={out[None][0]}/{len(imgs)} -- the old silent-acceptance behaviour")
    check("the DEFAULT refuses every misfit", out[0.15][0] == 0 and out[0.15][1] == len(imgs),
          f"{out[0.15][1]}/{len(imgs)} refused at the 0.15 default vs 0/{len(imgs)} ungated")
    # 0.10 is only PARTIAL, which is why it is not the default: index_fused OPTIMISES the fit rather
    # than returning a chance-level one, so a wrong-cell frame lands above the 2.7% random rate.
    check("0.10 is only partial (why 0.15 is the default)", 0 < out[0.10][1] < len(imgs),
          f"{out[0.10][1]}/{len(imgs)} refused at 0.10 -- a chance floor alone is not enough")
    check("rejections are reported in stats()", "gate_rejected" in
          sd.StreamDriver(MB, PAN, CLEN, LAM, (NPX, NPX), dtype=np.uint16, B=8).stats(),
          "dropping data must never be silent")


def main():
    part1()
    part2()
    part3()
    print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
