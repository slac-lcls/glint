"""Does the rescue's HALF-turn azimuth sweep cost indexing rate on OBLIQUE cells?

replica_gpu builds axis1 as  a1(th) = L1*(c01*cn + s01*(cos th * u + sin th * v))  over
TH = linspace(0, pi, NANG) -- half the cone. th -> th+pi sends a1 -> 2*c01*L1*cn - a1, which is
-a1 (same lattice vector, so half is complete) ONLY when c01 == 0. For c01 != 0 the -a1 partner
sits on the SUPPLEMENTARY cone and is never generated, so half the cone is unreachable.

Arms (all at NANG samples unless noted):
  half   [0,pi)   NANG      -- current behaviour
  full   [0,2pi)  NANG      -- matched COST (1 deg steps instead of 0.5)
  full2  [0,2pi)  2*NANG    -- matched RESOLUTION (2x the axis1 objective work)

Regimes: dense (3D-complete cloud) and still (Ewald-shell slice, the rescue's real regime).

  python azimuth_coverage.py [ntrial] [regime: both|dense|still]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glint.replica_gpu as rg
from glint.replica_gpu import index_known_gpu_cell
from glint.multishot import same_lattice

NTRIAL = int(sys.argv[1]) if len(sys.argv) > 1 else 40
REGIME = sys.argv[2] if len(sys.argv) > 2 else "both"
DMIN, LAM = 3.0, 1.3               # 3 A resolution, ~9.5 keV
QMAX = 1.0 / DMIN
DENSE_CAP = 3000                   # subsample dense clouds (memory: objective is NC*NANG x P)

CELLS = [                          # = experiments/gen_cells.py
    ("lyso_tet",  (79.0, 79.0, 38.0, 90, 90, 90)),
    ("prok_tet",  (68.5, 68.5, 109.0, 90, 90, 90)),
    ("thaum_tet", (58.0, 58.0, 130.0, 90, 90, 90)),
    ("hex_P6",    (105.0, 105.0, 75.0, 90, 90, 120)),
    ("cubic_ins", (78.0, 78.0, 78.0, 90, 90, 90)),
    ("ortho_lg",  (60.0, 110.0, 135.0, 90, 90, 90)),
    ("tricl",     (45.0, 55.0, 65.0, 80, 85, 95)),
    ("mono",      (60.0, 70.0, 90.0, 90, 105, 90)),
    ("large_tet", (140.0, 140.0, 150.0, 90, 90, 90)),
    ("trig_ob",   (70.0, 70.0, 70.0, 100, 100, 100)),
]


def cell_to_A(a, b, c, al, be, ga):
    al, be, ga = np.radians([al, be, ga])
    va = [a, 0, 0]
    vb = [b * np.cos(ga), b * np.sin(ga), 0]
    cx = c * np.cos(be); cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    vc = [cx, cy, np.sqrt(max(c * c - cx * cx - cy * cy, 1e-9))]
    return np.array([va, vb, vc]).T            # real-space cell, cols a,b,c


def rand_rot(r):
    u1, u2, u3 = r.random(3)
    q = np.array([np.sqrt(1-u1)*np.sin(2*np.pi*u2), np.sqrt(1-u1)*np.cos(2*np.pi*u2),
                  np.sqrt(u1)*np.sin(2*np.pi*u3), np.sqrt(u1)*np.cos(2*np.pi*u3)])
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rlps(Ar, rng):
    """All reciprocal-lattice points within QMAX for real-space basis Ar (cols a,b,c)."""
    B = np.linalg.inv(Ar).T                                  # reciprocal, cols a*,b*,c*
    H = (np.ceil(QMAX / np.linalg.norm(B, axis=0)) + 1).astype(int)
    g = np.mgrid[-H[0]:H[0]+1, -H[1]:H[1]+1, -H[2]:H[2]+1].reshape(3, -1).T
    g = g[np.any(g != 0, axis=1)]
    q = g @ B.T
    return q[np.linalg.norm(q, axis=1) <= QMAX]


def still(q, rng, target=45):
    """Ewald-shell slice: a still records rlps with |q + k0| = |k0|, i.e. qz = -lam*|q|^2/2.
    Widen the mosaic tolerance until ~target spots are on the shell."""
    resid = np.abs(q[:, 2] + LAM * (q ** 2).sum(1) / 2.0)
    tol = 2e-4
    for _ in range(40):
        sel = resid < tol
        if sel.sum() >= target:
            break
        tol *= 1.4
    idx = np.flatnonzero(sel)
    if len(idx) > target:
        idx = rng.choice(idx, target, replace=False)
    return q[idx]


def set_grid(full, mult):
    """Force the azimuth grid for one arm. _azimuth_grid() dispatches on |c01| to (CA,SA) or
    (CA2,SA2), so BOTH pairs are overwritten -- otherwise the arm would only bind on the branch
    the cell happens to select."""
    n = int(rg.NANG_BASE * mult)
    th = np.linspace(0, (2 * np.pi if full else np.pi), n, endpoint=False)
    rg.NANG = n
    ca = torch.as_tensor(np.cos(th), dtype=torch.float64, device=rg.DEV)
    sa = torch.as_tensor(np.sin(th), dtype=torch.float64, device=rg.DEV)
    rg.CA = rg.CA2 = ca
    rg.SA = rg.SA2 = sa


rg.NANG_BASE = rg.NANG
ARMS = [("half", False, 1), ("full", True, 1), ("full2", True, 2)]


def gate(M, q, Mc):
    """repo Table-1 gate: correct lattice AND >=25% of spots indexed AND >=10 reflections."""
    if M is None or not same_lattice(np.asarray(M, float), Mc):
        return 0
    r = np.asarray(q) @ M
    m = int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())
    return int(m / len(q) >= 0.25 and m >= 10)


def run(regime):
    print(f"\n=== {regime.upper()} ===  ntrial={NTRIAL}/cell  dev={rg.DEV}  NANG_BASE={rg.NANG_BASE}")
    print(f"{'cell':<11}{'c01':>8}{'n':>7} |{'half':>7}{'full':>7}{'full2':>7} |{'ms/frame (half/full/full2)':>30}")
    tot = {a[0]: 0 for a in ARMS}; ntot = 0
    for name, cp in CELLS:
        A0 = cell_to_A(*cp)
        _, c01, _, _, _ = rg._axes_from_cell(A0)
        rng = np.random.default_rng(abs(hash(name)) % 2**31)
        frames, Mcs = [], []
        for t in range(NTRIAL):
            Ar = rand_rot(rng) @ A0
            q = rlps(Ar, rng)
            if regime == "dense":
                if len(q) > DENSE_CAP:
                    q = q[rng.choice(len(q), DENSE_CAP, replace=False)]
            else:
                q = still(q, rng)
            if len(q) >= 10:
                frames.append(q); Mcs.append(Ar)
        okc, msc = {}, {}
        for aname, full, mult in ARMS:
            set_grid(full, mult)
            index_known_gpu_cell(frames[0], Mcs[0])                      # warmup for this grid
            if rg.DEV == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = [index_known_gpu_cell(q, Mc) for q, Mc in zip(frames, Mcs)]
            if rg.DEV == "cuda":
                torch.cuda.synchronize()
            msc[aname] = 1e3 * (time.perf_counter() - t0) / len(frames)
            okc[aname] = sum(gate(M, q, Mc) for M, q, Mc in zip(res, frames, Mcs))
            tot[aname] += okc[aname]
        ntot += len(frames)
        n = len(frames); med = int(np.median([len(f) for f in frames]))
        print(f"{name:<11}{c01:+8.4f}{med:7d} |" +
              "".join(f"{100*okc[a[0]]//n:6d}%" for a in ARMS) +
              f" |{msc['half']:11.1f}{msc['full']:9.1f}{msc['full2']:9.1f}", flush=True)
    print(f"{'TOTAL':<11}{'':8}{'':7} |" + "".join(f"{100*tot[a[0]]//ntot:6d}%" for a in ARMS) +
          f"   ({ntot} frames)")


if __name__ == "__main__":                    # importable: azimuth_oblique.py reuses the helpers
    for r in (["dense", "still"] if REGIME == "both" else [REGIME]):
        run(r)
