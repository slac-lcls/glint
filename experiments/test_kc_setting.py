"""The known-cell engines hand back a basis of the crystal lattice, in a setting equivalent to the reference.

WHY THIS EXISTS (review r2, finding s4-01). The anchor pool is a half-sphere, so on a frame whose true shortest
axis points to z<0 the anchor is -v0 and the axis-1 sweep finds -v1. _third_axis then placed a2 at the
reference's handedness: the mirror image of -v2 through the (v0, v1) plane. Unless a2 is perpendicular to v0 and
v1, that is not a lattice vector, and the anneal returned a basis with the reference metric that indexes ~40 % of
the spots. same_lattice compares metrics only, and the live gate (0.15) passes 40 %, so StreamDriver integrated
those frames with hkl that are not Laue-equivalent to the truth: 15/24 triclinic, 13/24 monoclinic (b in the
middle) and 15/24 monoclinic (b shortest) on b91ce63. Orthogonal-a2 cells (orthorhombic, tetragonal, hexagonal)
were never affected and must run exactly the code they ran before. A second, rarer failure on the same cells
(present before the mirror too): the anneal converges to another basis of the right lattice, (-a, -c, -b) on the
triclinic HEWL cell or (a, -b, -a-c) on the b-mid monoclinic one, which _relabel_like cannot undo either;
replica_gpu._ref_setting moves those to the setting whose metric matches the reference.

For each frame, T = inv(M_true) @ M must be an integer unimodular matrix whose transpose is in the Laue group
(EQUIV). INEQUIV = a lattice basis in another setting; MISOR = not a basis of the lattice at all.

  PYTHONPATH=. python experiments/test_kc_setting.py      # exit 0 = all pass  (CPU torch, ~20 s; skips without)
"""
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:
    print(f"SKIP {os.path.basename(__file__)} -- no torch: glint.replica_gpu cannot import here")
    sys.exit(0)
if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (1, 12):
    print(f"SKIP {os.path.basename(__file__)} -- torch {torch.__version__} is below pyproject's floor (>= 1.12)")
    sys.exit(0)

import glint.replica_gpu as rg                                          # noqa: E402
import glint.replica_gpu_batch as rgb                                   # noqa: E402
from glint.lattice import cell_to_Ar                                    # noqa: E402
from glint.simulate import simulate_shot                                # noqa: E402
from glint.stream_driver import StreamDriver, _relabel_like, laue_ops   # noqa: E402

N = 24
NPAN = 64
PANELS = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=1e4,
               cx=-(NPAN / 2.0 - 0.5), cy=-(NPAN / 2.0 - 0.5), coffset=0.0,
               min_fs=0, max_fs=NPAN - 1, min_ss=0, max_ss=NPAN - 1)]
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + str(detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def classify(Mt, M, ops):
    T = np.linalg.solve(Mt, M); Ti = np.rint(T)
    if np.abs(T - Ti).max() > 0.05 or abs(round(np.linalg.det(Ti))) != 1:
        return "MISOR"
    return "EQUIV" if tuple(Ti.T.astype(int).ravel()) in ops else "INEQUIV"


def shots_for(cell, seed=12345):
    rng = np.random.default_rng(seed)
    return [simulate_shot(cell=cell, n_target=100, dmin=2.0, pos_sigma=2e-4, rng=rng) for _ in range(N)]


def driver_bases(Mc, shots, laue):
    """Mcan for every frame the real StreamDriver integrates (default gates), keyed by event."""
    d = StreamDriver(Mc, PANELS, 0.1, 1.3, (NPAN, NPAN), dtype=np.uint16, B=N, dmin=2.0, use_gpu=False, laue=laue)
    cap, cur = {}, {}
    real_int, real_std = d._integrate_one, d._standardize

    def spy_int(i, M, grid, acc, *a, **kw):
        cur["ev"] = int(d._idx[i])
        return real_int(i, M, grid, acc, *a, **kw)

    def spy_std(M, ref=None):
        out = real_std(M, ref=ref)
        if "ev" in cur:
            cap[cur.pop("ev")] = np.array(out, float)
        return out

    d._integrate_one = spy_int; d._standardize = spy_std
    for s in shots:
        d.push_q(s.g)
    d.flush()
    return cap


AFFECTED = [("triclinic 50/60/70/80/85/95", (50., 60., 70., 80., 85., 95.), "-1"),
            ("near-orthogonal triclinic 50/60/70/89.5/90/90.2",
             (50., 60., 70., 89.5, 90., 90.2), "-1"),
            ("monoclinic b-mid 40/60/70 beta=105", (40., 60., 70., 90., 105., 90.), "2/m_uab"),
            ("monoclinic b-shortest 60/40/70 beta=105", (60., 40., 70., 90., 105., 90.), "2/m_uab"),
            ("triclinic HEWL 27.24/31.87/34.23/88.52/108.53/111.89", (27.24, 31.87, 34.23, 88.52, 108.53, 111.89), "-1")]
# Pseudo-merohedral: |a+c| = 64.16 against a = 64.2, so (a, b, c) and (-a, -b, a+c) have the same metric to 0.2 %
# and geometry cannot tell them apart (an intensity-based question). Only "a lattice basis" is asserted here.
PSEUDO = [("monoclinic myoglobin 64.2/30.9/34.8 beta=105.8", (64.2, 30.9, 34.8, 90., 105.8, 90.), "2/m_uab")]
CONTROLS = [("orthorhombic 40/60/70", (40., 60., 70., 90., 90., 90.), "mmm"),
            ("tetragonal 79/79/38", (79., 79., 38., 90., 90., 90.), "4/mmm"),
            ("hexagonal 60/60/100", (60., 60., 100., 90., 90., 120.), "6/mmm")]

for name, cell, laue in AFFECTED + CONTROLS + PSEUDO:
    print(f"\n{name}  (Laue {laue})")
    ops = {tuple(np.asarray(o, int).ravel()) for o in laue_ops(laue)}
    Mc = cell_to_Ar(*cell)
    shots = shots_for(cell)
    floor = 0 if (name, cell, laue) in PSEUDO else N
    per = collections.Counter(classify(s.M, _relabel_like(np.asarray(M, float), Mc), ops) if M is not None else "None"
                              for s, M in zip(shots, (rg.index_known_gpu_cell(s.g, Mc) for s in shots)))
    check(f"per-frame index_known_gpu_cell: no non-lattice basis, >= {floor}/{N} equivalent",
          per["MISOR"] == 0 and per["EQUIV"] >= floor, dict(per))
    bat = collections.Counter(classify(s.M, _relabel_like(np.asarray(M, float), Mc), ops) if M is not None else "None"
                              for s, M in zip(shots, rgb.index_fused([s.g for s in shots], Mc, B=N)))
    check(f"batched index_fused: no non-lattice basis, >= {floor}/{N} equivalent",
          bat["MISOR"] == 0 and bat["EQUIV"] >= floor, dict(bat))
    cap = driver_bases(Mc, shots, laue)
    drv = collections.Counter(classify(shots[ev].M, Mcan, ops) for ev, Mcan in cap.items())
    check(f"StreamDriver integrates {len(cap)}/{N}: no non-lattice basis, >= {floor} equivalent",
          drv["MISOR"] == 0 and drv["EQUIV"] >= floor, dict(drv))

print("\northogonal-a2 cells do not take the two-handed path (they run the code they ran before)")
for name, cell, _ in CONTROLS:
    L, c01, c02, c12, _ = rg._axes_from_cell(cell_to_Ar(*cell))
    check(f"{name}: _both_hands is False", not rg._both_hands(float(L[2]), c01, c02, c12))
# Consensus or lock cells can be slightly skewed even when the underlying crystal is orthogonal. Without
# explicit symmetry metadata, that skew cannot safely be used to rule out the opposite hand.
for label, cp in (("consensus-like 79.1/78.95/38.02/90.03/89.98/90.04", (79.1, 78.95, 38.02, 90.03, 89.98, 90.04)),
                  ("GPU stream lock 78.706/78.792/37.813/89.81/90.08/90.19",
                   (78.706, 78.792, 37.813, 89.8107, 90.0832, 90.1885))):
    L, c01, c02, c12, _ = rg._axes_from_cell(cell_to_Ar(*cp))
    check(f"skewed {label}: _both_hands is True", rg._both_hands(float(L[2]), c01, c02, c12))
for name, cell, _ in AFFECTED:
    L, c01, c02, c12, _ = rg._axes_from_cell(cell_to_Ar(*cell))
    check(f"{name}: _both_hands is True", rg._both_hands(float(L[2]), c01, c02, c12))

print(f"\n{'FAILURES: ' + ', '.join(FAILS) if FAILS else 'ALL PASS'}")
sys.exit(1 if FAILS else 0)
