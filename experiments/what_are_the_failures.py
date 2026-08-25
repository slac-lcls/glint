"""Interrogate the "28/42 genuinely hard" claim from test_perframe_diff.py.

That claim called a frame "genuinely hard, not an architecture gap" when NONE of blind-top1,
blind-Nbest, or known-cell-rescue cleared the strict gate. But all three of those are GLINT --
they share objective_t, the Fibonacci grid DIRS, and anneal_batch_t. A systematic blind spot common
to all three would look EXACTLY like "genuinely hard". So the label was never actually tested against
anything independent.

This does that test. Cached per-frame results from genuinely independent implementations exist on
the SAME 120 frames (verified byte-identical frames file, md5 f89fa57d...): xgandalf blind + known-cell
(xg_*120.txt) and ffbidx (ffbidx_120.txt). Scored at the IDENTICAL strict gate. Also reports peak
counts per group, to separate "hard because the data is sparse" from "hard because our search misses".

  python what_are_the_failures.py
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

# GLINT_WT overrides, so a different checkout can import strict_gate/LYSO from here (the retry
# arsenal does) without silently pulling glint out of one hard-coded worktree.
WT = os.environ.get("GLINT_WT", "/sdf/home/s/smarches/glint_streamfix_wt")
XG = os.environ.get("GLINT_XG", "/sdf/home/s/smarches/git/glint/experiments/xgandalf")
sys.path.insert(0, WT)
sys.path.insert(0, WT + "/experiments")
import glint.glint_fast as gf
from glint.glint_fast import matched, index_blind_nbest
from glint.multishot import same_lattice
from glint.replica_gpu import index_known_gpu_cell
import glint.replica_gpu_batch as rgb
import glint.stream_driver as sd

LYSO = gf.LYSO
GATE_FRAC, GATE_MIN, TOL = 0.25, 10, 0.15
FRAMES_PATH = WT + "/experiments/frames_cxidb_clean.txt"


def strict_gate(M, q, truth=None):
    if M is None:
        return False
    truth = LYSO if truth is None else truth
    if not same_lattice(M, truth):
        return False
    m = matched(M, q)
    return (m / len(q) >= GATE_FRAC) and (m >= GATE_MIN)


def parse_xg(path):
    """verbatim from xgandalf/score_xg_gate.py"""
    res = {}
    for line in open(path):
        p = line.split()
        if len(p) < 2:
            continue
        if p[1] == "NONE":
            res[int(p[0])] = None
        else:
            B = np.array(list(map(float, p[1:10]))).reshape(3, 3)
            res[int(p[0])] = np.linalg.inv(B).T if np.median(np.linalg.norm(B, axis=0)) < 1 else B
    return res


def parse_ffbidx(path):
    """verbatim from compare_ffbidx.py"""
    out = {}
    for ln in open(path):
        t = ln.split()
        if len(t) < 2:
            continue
        fid = int(t[0])
        out[fid] = None if t[1] == "NONE" else np.array(list(map(float, t[1:10]))).reshape(3, 3).T
    return out


def push_blind_q(driver, qq):
    driver.n_pushed += 1
    if len(qq) >= driver.min_peaks:
        nb = driver._blind_index(qq, driver.warmup_nbest)
        driver._rc.add_frame([c for c, _ in nb])
        driver.n_warmup += 1
        if getattr(driver, "_warmup_buf", None) is not None:
            driver._warmup_buf.append(qq)
    Mc, sup, _ = driver._rc.verdict()
    if Mc is not None:
        driver._lock(Mc, sup, standardize=True)


def main():
    t0 = time.time()
    frames = list(gf.load(FRAMES_PATH))
    n = len(frames)

    # ---- reproduce the exact streaming run from test_perframe_diff.py ----
    N = 400
    panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
                   res=1.0 / (0.1 / 1000.0), cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5),
                   coffset=0.0, min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
    drv = sd.StreamDriver(None, panels, 100.0 / 1000.0, 1.0, (N, N), dtype=np.uint16,
                          B=20, dmin=2.0, tol=0.002, warmup_nbest=3)
    warmup_idx = []
    for i, qq in enumerate(frames):
        qq = np.asarray(qq, float)
        if drv._blind:
            warmup_idx.append(i)
            push_blind_q(drv, qq)
        else:
            drv._q[drv._n] = qq
            drv._n += 1; drv.n_pushed += 1
            if drv._n == drv.B:
                drv.flush()
    drv.flush()

    warmup_set = set(warmup_idx)
    post_lock_idx = [i for i in range(n) if i not in warmup_set]
    Ms = rgb.index_fused([np.asarray(frames[i], float) for i in post_lock_idx], drv.Mc,
                         B=len(post_lock_idx))
    streaming_fail = [i for i, M in zip(post_lock_idx, Ms)
                      if not strict_gate(np.asarray(M, float) if M is not None else None,
                                         frames[i], drv.Mc)]
    print(f"locked_after={drv.locked_after}   streaming fails "
          f"{len(streaming_fail)}/{len(post_lock_idx)} post-lock  [{time.time()-t0:.0f}s]")

    # ---- classify by offline GLINT's full arsenal, exactly as test_perframe_diff.py did ----
    hard, recoverable = [], []
    for i in streaming_fail:
        q = frames[i]
        hit = None
        for k, (c, _) in enumerate(index_blind_nbest(q, 3)):
            if same_lattice(c, LYSO):
                hit = c; break
        if hit is None:
            Mr = index_known_gpu_cell(q, LYSO)
            if Mr is not None and same_lattice(Mr, LYSO):
                hit = Mr
        (recoverable if (hit is not None and strict_gate(hit, q, LYSO)) else hard).append(i)
    print(f"offline-GLINT recoverable: {len(recoverable)}   'genuinely hard': {len(hard)}")

    # ---- the actual test: do INDEPENDENT indexers clear the same gate on those frames? ----
    xgb, xgk = parse_xg(XG + "/xg_blind120.txt"), parse_xg(XG + "/xg_known120.txt")
    ffb = parse_ffbidx(XG + "/ffbidx_120.txt")
    IND = [("xgandalf-blind", xgb), ("xgandalf-known", xgk), ("ffbidx-known", ffb)]

    succeeded = [i for i in range(n) if i not in set(streaming_fail) and i not in warmup_set]
    groups = [("streaming SUCCESS (control)", succeeded),
              ("offline-GLINT recoverable", recoverable),
              ("'genuinely hard' (the 28)", hard)]

    print(f"\n{'group':32s} {'n':>4s} {'npeaks med':>11s}  " +
          "  ".join(f"{nm:>14s}" for nm, _ in IND) + f"  {'ANY indep':>10s}")
    per_frame_any = {}
    for label, idxs in groups:
        if not idxs:
            print(f"{label:32s} {0:4d}   (empty)"); continue
        pk = [len(frames[i]) for i in idxs]
        cols, anyhit = [], []
        for i in idxs:
            hits = [strict_gate(res.get(i), frames[i], LYSO) for _, res in IND]
            per_frame_any[i] = any(hits)
            anyhit.append(any(hits))
        for k, (_, res) in enumerate(IND):
            c = sum(1 for i in idxs if strict_gate(res.get(i), frames[i], LYSO))
            cols.append(f"{c:5d}/{len(idxs):<3d} {100*c/len(idxs):3.0f}%")
        a = sum(anyhit)
        print(f"{label:32s} {len(idxs):4d} {int(np.median(pk)):11d}  " + "  ".join(cols) +
              f"  {a:4d} {100*a/len(idxs):3.0f}%")

    solved_hard = [i for i in hard if per_frame_any[i]]
    print(f"\nof the {len(hard)} frames labelled 'genuinely hard', an INDEPENDENT indexer clears the "
          f"identical strict gate on {len(solved_hard)}: {sorted(solved_hard)}")
    if hard:
        print(f"  npeaks of the hard set: {sorted(len(frames[i]) for i in hard)}")
        print(f"  npeaks of streaming successes: median {int(np.median([len(frames[i]) for i in succeeded]))}, "
              f"min {min(len(frames[i]) for i in succeeded)}")
    print(f"\nDONE elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
