"""Streaming vs timeless at scale, on synthetic stills calibrated to the real cxidb set.

The 120-frame real set gives 73 / 78 / 91 -- a 13-frame residual gap after warmup_rescue, too few
frames to dissect. This generates thousands of stills whose difficulty is a KNOB, so the gap can be
measured as a function of difficulty and attributed to a mechanism instead of a frame list.

THE KNOB IS LATTICE FRACTION, not peak count. The peel study over all 42 real failures showed the
discriminator is dominance -- best single lattice covers 0.435 of peaks on frames streaming solves
vs 0.275 on frames it fails -- and NOT the number of lattices (1.55 vs 1.62, no separation). So the
ladder holds npk at the real median of 100 and varies only how many of those peaks are the crystal.
The rungs bracket the real regime on both sides.

THE HYPOTHESIS UNDER TEST. Offline runs blind + N-best on EVERY frame and only then falls back to
known-cell rescue. Streaming, once locked, runs known-cell ONLY -- no per-frame blind retry. If that
is the whole residual gap, a blind retry on exactly the frames that fail the gate should close it.
Arm E does that, and it is simultaneously the offline post-rescue pass the flagged-frame channel was
built to feed, so one run answers both questions.

Generator, calibrated against frames_cxidb_clean.txt (median 100 peaks/frame, |q| <= 0.606 1/A,
dmin 1.65 A, chance matched_frac 0.019): random orientation -> reciprocal lattice -> Ewald shell ->
sample n_lat reflections weighted toward low |q| (what a peakfinder actually reports) -> pad to npk
with spurious peaks at random directions -> jitter every q by sigma * |a*|.

  python gap_at_scale.py [n_frames_per_condition]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT); sys.path.insert(0, WT + "/experiments")
import glint.glint_fast as gf
from glint.glint_fast import matched, index_blind_nbest
from glint.multishot import same_lattice
from glint.predict import recip_from_M
from what_are_the_failures import strict_gate, push_blind_q, LYSO
import glint.replica_gpu_batch as rgb
import glint.stream_driver as sd

QMAX = 0.606                      # 1/A, matches the real set (dmin 1.65 A)
WAVE_GEN = 1.3                    # A, cxidb lysozyme
TOL_EXC = 0.004                   # generous Ewald shell -> a POOL; the sample below sets the count
N_PANEL = 400
GATE_FRAC, GATE_MIN = 0.25, 10

# (label, n_lattice, npk, jitter sigma in |a*|).  Real: solved 0.435, failed 0.275.
CONDITIONS = [
    ("frac0.45", 45, 100, 0.010),
    ("frac0.35", 35, 100, 0.015),
    ("frac0.27", 27, 100, 0.020),
    ("frac0.22", 22, 100, 0.025),
    ("frac0.18", 18, 100, 0.030),
]


def rand_rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def make_frame(rng, R0, n_lat, npk, sigma, astar):
    Rot = rand_rot(rng)
    R = R0 @ Rot
    nb = [int(np.ceil(QMAX / np.linalg.norm(R[j]))) + 1 for j in range(3)]
    g = np.stack(np.meshgrid(*[np.arange(-n, n + 1) for n in nb], indexing="ij"), -1).reshape(-1, 3)
    g = g[np.any(g != 0, 1)]
    q = g @ R
    q = q[np.linalg.norm(q, axis=1) <= QMAX]
    exc = np.abs(q[:, 2] + 0.5 * WAVE_GEN * np.einsum("ij,ij->i", q, q))
    pool = q[exc < TOL_EXC]
    if len(pool) < n_lat:
        return None
    w = 1.0 / (np.linalg.norm(pool, axis=1) ** 2 + 1e-3)
    w /= w.sum()
    lat = pool[rng.choice(len(pool), size=n_lat, replace=False, p=w)]
    ns = npk - n_lat
    d = rng.normal(size=(ns, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    rr = rng.uniform(0.03, QMAX, size=ns)
    out = np.vstack([lat, d * rr[:, None]])
    out = out + rng.normal(scale=sigma * astar, size=out.shape)
    rng.shuffle(out)
    return out


def run_driver(frames, warmup_rescue, adaptive_relock):
    panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
                   res=1.0 / (0.1 / 1000.0), cx=-(N_PANEL/2.0-0.5), cy=-(N_PANEL/2.0-0.5),
                   coffset=0.0, min_fs=0, max_fs=N_PANEL-1, min_ss=0, max_ss=N_PANEL-1)]
    kw = dict(B=20, dmin=1.0/QMAX, tol=0.002, warmup_nbest=3, warmup_rescue=warmup_rescue)
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
    """Arm E: what offline does that streaming does not -- blind + N-best on the failures."""
    ok = 0
    for i in idxs:
        q = frames[i]
        try:
            for k, (c, s) in enumerate(index_blind_nbest(q, 3)):
                if c is not None and strict_gate(np.asarray(c, float), q, Mc):
                    ok += 1
                    break
        except Exception:
            pass
    return ok


def main():
    NF = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    t0 = time.time()
    R0 = recip_from_M(LYSO)
    astar = float(min(np.linalg.norm(R0[j]) for j in range(3)))
    print(f"synthetic stills: {NF} frames/condition, LYSO, qmax={QMAX} 1/A, lambda={WAVE_GEN} A")
    print(f"real-set anchors: streaming solves at lattice frac 0.435, fails at 0.275\n")
    print(f"{'cond':9s} {'npk':>4s} {'true':>5s} | {'offline':>7s} {'stream':>7s} {'+warmup':>7s} "
          f"{'+relock':>7s} {'+retry':>7s} | {'gap':>5s} {'closed':>7s}")
    print("-" * 88)
    for label, n_lat, npk_t, sig in CONDITIONS:
        rng = np.random.default_rng(1234 + n_lat)
        frames = []
        while len(frames) < NF:
            f = make_frame(rng, R0, n_lat, npk_t, sig, astar)
            if f is not None:
                frames.append(f)
        npk = int(np.median([len(f) for f in frames]))

        from glint.hybrid_stream import hybrid_index
        res, st = hybrid_index(frames, Mc_known=LYSO, warmup=True)
        off = sum(int(strict_gate(r["M"], q, LYSO)) for r, q in zip(res, frames))

        base, _fb, _db = run_driver(frames, False, False)
        wu, failed_w, drvw = run_driver(frames, True, False)
        rl, _fr, _dr = run_driver(frames, True, True)
        retry = blind_retry(frames, failed_w, drvw.Mc) if drvw.Mc is not None else 0
        gap = (off or 0) - (wu or 0)
        closed = f"{100.0*retry/gap:.0f}%" if gap > 0 else "n/a"
        print(f"{label:9s} {npk:4d} {n_lat/npk_t:5.2f} | {off:7d} "
              f"{-1 if base is None else base:7d} {-1 if wu is None else wu:7d} "
              f"{-1 if rl is None else rl:7d} {(wu or 0)+retry:7d} | {gap:5d} {closed:>7s}")
    print(f"\nall rates out of {NF} frames; DONE elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
