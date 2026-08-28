"""Does committing to a cell early cost yield FOREVER, or does it amortize over a long run?

The abstract says the live path indexes 65% where offline reaches 76%. Two components are being
summed and they scale differently:
  - the WARM-UP cost is fixed: frames spent on discovery are ~5/N, so it vanishes as N grows.
  - the RETRY gap is PER-FRAME: post-lock the driver runs known-cell only, so every frame that a
    blind retry would have caught is lost again. That does not amortize.
If the second dominates, the penalty at 1e6 frames is the same as at 120.

Part 2 asks the question that only matters at long run length: the cell is committed once, but a real
run DRIFTS. This plants a slow scale drift and asks whether the committed cell decays -- and whether
adaptive re-lock recovers it. (Refining the locked cell continuously was measured separately and
rejected: the strict gate cannot resolve cell precision, +-0.6% swings the rate non-monotonically.)

  python gap_vs_runlength.py
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT); sys.path.insert(0, WT + "/experiments")
import glint.glint_fast as gf
from glint.glint_fast import index_blind_nbest
from glint.predict import recip_from_M
from what_are_the_failures import strict_gate, push_blind_q, LYSO
import glint.replica_gpu_batch as rgb
import glint.stream_driver as sd

QMAX, WAVE_GEN, TOL_EXC = 0.606, 1.3, 0.004
N_PANEL, GATE_MIN = 400, 10
N_LAT, NPK, SIG = 35, 100, 0.015          # the real data's regime (lattice fraction 0.35)


def rand_rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def make_frame(rng, R0, astar):
    R = R0 @ rand_rot(rng)
    nb = [int(np.ceil(QMAX / np.linalg.norm(R[j]))) + 1 for j in range(3)]
    g = np.stack(np.meshgrid(*[np.arange(-n, n+1) for n in nb], indexing="ij"), -1).reshape(-1, 3)
    g = g[np.any(g != 0, 1)]
    q = g @ R
    q = q[np.linalg.norm(q, axis=1) <= QMAX]
    exc = np.abs(q[:, 2] + 0.5 * WAVE_GEN * np.einsum("ij,ij->i", q, q))
    pool = q[exc < TOL_EXC]
    if len(pool) < N_LAT:
        return None
    w = 1.0 / (np.linalg.norm(pool, axis=1) ** 2 + 1e-3); w /= w.sum()
    lat = pool[rng.choice(len(pool), size=N_LAT, replace=False, p=w)]
    ns = NPK - N_LAT
    d = rng.normal(size=(ns, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    out = np.vstack([lat, d * rng.uniform(0.03, QMAX, size=ns)[:, None]])
    out = out + rng.normal(scale=SIG * astar, size=out.shape)
    rng.shuffle(out)
    return out


def gen(n, seed, drift=0.0):
    """drift = fractional cell-scale change accumulated linearly across the run."""
    rng = np.random.default_rng(seed)
    R0 = recip_from_M(LYSO)
    astar = float(min(np.linalg.norm(R0[j]) for j in range(3)))
    out = []
    while len(out) < n:
        scale = 1.0 + drift * (len(out) / max(n - 1, 1))
        f = make_frame(rng, R0 / scale, astar)      # cell grows -> reciprocal shrinks
        if f is not None:
            out.append(f)
    return out


def run_stream(frames, relock):
    panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
                   res=1.0/(0.1/1000.0), cx=-(N_PANEL/2.0-0.5), cy=-(N_PANEL/2.0-0.5),
                   coffset=0.0, min_fs=0, max_fs=N_PANEL-1, min_ss=0, max_ss=N_PANEL-1)]
    kw = dict(B=20, dmin=1.0/QMAX, tol=0.002, warmup_nbest=3, warmup_rescue=True)
    if relock:
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
    if drv.Mc is None:
        return 0, 0
    Ms = rgb.index_fused(frames, drv.Mc, B=len(frames))
    ok = sum(1 for i, M in enumerate(Ms)
             if strict_gate(None if M is None else np.asarray(M, float), frames[i], drv.Mc))
    return ok, len(warm)


def offline(frames):
    from glint.hybrid_stream import hybrid_index
    res, _ = hybrid_index(frames, Mc_known=LYSO, warmup=True)
    return sum(int(strict_gate(r["M"], q, LYSO)) for r, q in zip(res, frames))


def main():
    t0 = time.time()
    print("PART 1 -- static cell: does the penalty amortize as the run grows?\n")
    print(f"{'N':>6s} {'offline':>9s} {'streaming':>10s} {'gap (pp)':>9s} {'warm-up as % of run':>21s}")
    print("-" * 62)
    for N in (120, 500, 1500, 3000):
        fr = gen(N, 7)
        yo = offline(fr); ys, nw = run_stream(fr, False)
        print(f"{N:6d} {100.0*yo/N:8.1f}% {100.0*ys/N:9.1f}% {100.0*(yo-ys)/N:9.1f} "
              f"{100.0*nw/N:20.2f}%")

    print(f"\nPART 2 -- N=1500, a cell that drifts across the run\n")
    print(f"{'drift':>7s} {'offline':>9s} {'stream (no relock)':>19s} {'stream (relock)':>16s}")
    print("-" * 56)
    for d in (0.0, 0.003, 0.010):
        fr = gen(1500, 11, drift=d)
        yo = offline(fr)
        y1, _ = run_stream(fr, False)
        y2, _ = run_stream(fr, True)
        print(f"{100*d:6.1f}% {100.0*yo/1500:8.1f}% {100.0*y1/1500:18.1f}% "
              f"{100.0*y2/1500:15.1f}%")
    print(f"\nDONE elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
