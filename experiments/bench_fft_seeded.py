"""FFT-seeded GLINT front-end (user idea) vs blind Fibonacci, across a sampling-density sweep.

Same lyso lattice; regimes = realistic sparse single-shot -> still slice -> oscillation wedges -> full
rotation. Compare index_blind_fast (70k blind Fibonacci starts) vs index_blind_fft_seeded (FFT-volume
peaks as data-driven seeds, Fibonacci fallback when starved). Metric: correct-cell rate + wall-time.
Hypothesis: on dense data FFT seeds are few + correct -> same rate, much faster (fewer refine starts);
on sparse the FFT is starved -> falls back to Fibonacci (no worse).

  ~/... pytorch env on a GPU node:  srun ... --gpus 1 python bench_fft_seeded.py
"""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("STEPS", "8")
import numpy as np
import torch
from glint.glint_fast import index_blind_fast, index_blind_cluster_seeded


def timed(fn, g):
    """Run fn(g); return (ok_M, seconds) catching OOM/errors as a failure."""
    t0 = time.perf_counter()
    try:
        M = fn(g)
    except Exception:
        torch.cuda.empty_cache(); return None, time.perf_counter() - t0
    return M, time.perf_counter() - t0

CELL = np.array([79.0, 79.0, 38.0]); LAM = 1.0; DMIN = 3.0
QMAX = 1.0 / DMIN
B = np.diag(1.0 / CELL)
REPS = 8
WEDGES = [("sparse ~100pk", -1.0), ("still slice", 0.0), ("wedge 5deg", 5.0),
          ("wedge 20deg", 20.0), ("wedge 60deg", 60.0), ("full rotation", 180.0)]


def rot(ax, th):
    ax = ax / np.linalg.norm(ax); c, s = np.cos(th), np.sin(th); x, y, z = ax
    return np.array([[c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
                     [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
                     [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]])


H = (np.ceil(QMAX * CELL) + 1).astype(int)
HKL = np.mgrid[-H[0]:H[0]+1, -H[1]:H[1]+1, -H[2]:H[2]+1].reshape(3, -1).T
HKL = HKL[np.any(HKL != 0, axis=1)]


def observed_rlps(A, wedge_deg, osc_axis, rng):
    q0 = HKL @ A.T
    m = np.linalg.norm(q0, axis=1) <= QMAX
    q0, hk = q0[m], HKL[m]
    tol = 0.004
    if wedge_deg <= 0.0:
        exc = q0[:, 2] + 0.5 * LAM * np.einsum("ij,ij->i", q0, q0)
        obs = np.abs(exc) < tol
        g = hk[obs] @ A.T
        if wedge_deg == -1.0 and len(g) > 100:               # realistic sparse: ~100 strongest-ish
            g = g[rng.choice(len(g), 100, replace=False)]
        return g
    phis = np.deg2rad(np.linspace(0, wedge_deg, max(2, int(wedge_deg * 2))))
    obs = np.zeros(len(q0), bool)
    for ph in phis:
        qp = q0 @ rot(osc_axis, ph).T
        obs |= np.abs(qp[:, 2] + 0.5 * LAM * np.einsum("ij,ij->i", qp, qp)) < tol
    return hk[obs] @ A.T


def cell_ok(M):
    if M is None:
        return False
    L = np.sort(np.linalg.norm(np.asarray(M, float), axis=0))
    return bool(np.all(np.abs(L - np.sort(CELL)) <= 0.05 * np.sort(CELL)))


# warmup
_w = np.random.default_rng(1)
_g = observed_rlps(rot([1, 1, 1], 0.5) @ B, 0.0, np.array([0, 1, 0.]), _w)
index_blind_fast(_g); index_blind_cluster_seeded(_g)

rng = np.random.default_rng(0)
print(f"cell {CELL.tolist()}  dmin {DMIN}  reps {REPS}  STEPS={os.environ['STEPS']}")
print(f"{'regime':16}{'n_rlps':>8}{'Fib rate':>9}{'Fib ms':>9}{'Clus rate':>10}{'Clus ms':>9}{'speedup':>9}")
for label, W in WEDGES:
    ns, okF, okS, tF, tS = [], 0, 0, [], []
    for r in range(REPS):
        A = rot(rng.normal(size=3), rng.uniform(0, np.pi)) @ B
        g = observed_rlps(A, W, rng.normal(size=3), rng)
        ns.append(len(g))
        mF, dF = timed(index_blind_fast, g); tF.append(dF); okF += cell_ok(mF)
        mS, dS = timed(index_blind_cluster_seeded, g); tS.append(dS); okS += cell_ok(mS)
    mtF, mtS = 1e3*np.median(tF), 1e3*np.median(tS)
    print(f"  {label:14}{int(np.median(ns)):>8}{100*okF//REPS:>8}%{mtF:>9.1f}{100*okS//REPS:>8}%{mtS:>9.1f}{mtF/mtS:>8.1f}x", flush=True)
