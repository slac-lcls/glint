"""The gap_at_scale.py arms, run on REAL frame sets.

glint#73 showed on 2000 synthetic stills that the streaming-vs-timeless gap is one missing per-frame
blind retry, closing 98-100% of it in the regime the real data occupies. This runs the IDENTICAL five
arms on real frames so the claim is not synthetic-only. The retry arm is also the offline post-rescue
pass the flagged-peak channel feeds, so its column is the measured post-rescue yield.

  python gap_on_real.py frames_a.txt [frames_b.txt ...]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT); sys.path.insert(0, WT + "/experiments")
import glint.glint_fast as gf
from glint.glint_fast import index_blind_nbest
from what_are_the_failures import strict_gate, push_blind_q, LYSO
import glint.replica_gpu_batch as rgb
import glint.stream_driver as sd

N_PANEL, GATE_MIN = 400, gf.GATE_MIN


def run_driver(frames, dmin, warmup_rescue, adaptive_relock):
    panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
                   res=1.0 / (0.1 / 1000.0), cx=-(N_PANEL/2.0-0.5), cy=-(N_PANEL/2.0-0.5),
                   coffset=0.0, min_fs=0, max_fs=N_PANEL-1, min_ss=0, max_ss=N_PANEL-1)]
    kw = dict(B=20, dmin=dmin, tol=0.002, warmup_nbest=3, warmup_rescue=warmup_rescue)
    if adaptive_relock:
        kw.update(adaptive_relock=True, min_inliers=GATE_MIN)
    drv = sd.StreamDriver(None, panels, 0.1, 1.0, (N_PANEL, N_PANEL), dtype=np.uint16, **kw)
    warm = []
    for i, qq in enumerate(frames):
        if drv._blind:
            warm.append(i); push_blind_q(drv, qq)
        else:
            drv._q[drv._n] = qq; drv._n += 1; drv.n_pushed += 1
            if drv._n == drv.B:
                drv.flush()
    drv.flush()
    post = [i for i in range(len(frames)) if i not in set(warm)]
    if drv.Mc is None or not post:
        return None, [], drv
    Ms = rgb.index_fused([frames[i] for i in post], drv.Mc, B=len(post))
    ok, failed = [], []
    for i, M in zip(post, Ms):
        MM = None if M is None else np.asarray(M, float)
        (ok if strict_gate(MM, frames[i], drv.Mc) else failed).append(i)
    n_warm_ok = 0
    if warmup_rescue and warm:
        wm = rgb.index_fused([frames[i] for i in warm], drv.Mc, B=len(warm))
        n_warm_ok = sum(1 for i, M in zip(warm, wm)
                        if strict_gate(None if M is None else np.asarray(M, float),
                                       frames[i], drv.Mc))
    return len(ok) + n_warm_ok, failed, drv


def blind_retry(frames, idxs, Mc):
    ok = []
    for i in idxs:
        q = frames[i]
        try:
            for c, _s in index_blind_nbest(q, 3):
                if c is not None and strict_gate(np.asarray(c, float), q, Mc):
                    ok.append(i)
                    break
        except Exception:
            pass
    return ok


def main():
    paths = sys.argv[1:] or ["frames_cxidb_clean.txt", "frames_dials60.txt"]
    print(f"{'dataset':22s} {'N':>4s} {'dmin':>5s} | {'offline':>7s} {'stream':>6s} {'+warm':>6s} "
          f"{'+relock':>7s} {'+retry':>6s} | {'gap':>4s} {'closed':>7s}")
    print("-" * 92)
    for p in paths:
        frames = [np.asarray(f, float) for f in gf.load(p) if len(f) >= 6]
        n = len(frames)
        dmin = 1.0 / max(np.linalg.norm(f, axis=1).max() for f in frames)
        from glint.hybrid_stream import hybrid_index
        res, _st = hybrid_index(frames, Mc_known=LYSO, warmup=True)
        off = sum(int(strict_gate(r["M"], q, LYSO)) for r, q in zip(res, frames))
        base, _fb, _db = run_driver(frames, dmin, False, False)
        wu, failed_w, drvw = run_driver(frames, dmin, True, False)
        rl, _fr, _dr = run_driver(frames, dmin, True, True)
        rec = blind_retry(frames, failed_w, drvw.Mc) if drvw.Mc is not None else []
        gap = (off or 0) - (wu or 0)
        closed = f"{100.0*len(rec)/gap:.0f}%" if gap > 0 else "n/a"
        print(f"{os.path.basename(p):22s} {n:4d} {dmin:5.2f} | {off:7d} "
              f"{-1 if base is None else base:6d} {-1 if wu is None else wu:6d} "
              f"{-1 if rl is None else rl:7d} {(wu or 0)+len(rec):6d} | {gap:4d} {closed:>7s}")
        print(f"{'':22s}      retry recovered {len(rec)}/{len(failed_w)} failures: {rec[:20]}")


if __name__ == "__main__":
    main()
