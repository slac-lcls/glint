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
Near-orthogonal low-symmetry cells (tilt under the 1 deg gate) take that path only when their Laue class is given
(laue=; review of #225): the metric alone cannot tell them from the skewed lock cell of an orthogonal lattice.

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
            ("monoclinic b-mid 40/60/70 beta=105", (40., 60., 70., 90., 105., 90.), "2/m_uab"),
            ("monoclinic b-shortest 60/40/70 beta=105", (60., 40., 70., 90., 105., 90.), "2/m_uab"),
            ("triclinic HEWL 27.24/31.87/34.23/88.52/108.53/111.89", (27.24, 31.87, 34.23, 88.52, 108.53, 111.89), "-1")]
# Pseudo-merohedral: |a+c| = 64.16 against a = 64.2, so (a, b, c) and (-a, -b, a+c) have the same metric to 0.2 %
# and geometry cannot tell them apart (an intensity-based question). Only "a lattice basis" is asserted here.
PSEUDO = [("monoclinic myoglobin 64.2/30.9/34.8 beta=105.8", (64.2, 30.9, 34.8, 90., 105.8, 90.), "2/m_uab")]
CONTROLS = [("orthorhombic 40/60/70", (40., 60., 70., 90., 90., 90.), "mmm"),
            ("tetragonal 79/79/38", (79., 79., 38., 90., 90., 90.), "4/mmm"),
            ("hexagonal 60/60/100", (60., 60., 100., 90., 90., 120.), "6/mmm")]

# Near-orthogonal LOW-symmetry cells (review of #225): the tilt of a2 is under the 1 deg gate, so without a class
# they look like the skewed lock cell of an orthogonal lattice. With their Laue class given (laue=, as StreamDriver
# and hybrid_index pass it) they take the two-handed path and its setting step.
NEAR = [("near-orthogonal triclinic 50/60/70/89.5/90/90.2", (50., 60., 70., 89.5, 90., 90.2), "-1"),
        ("near-orthogonal triclinic 50/60/70/90.3/89.6/90.4", (50., 60., 70., 90.3, 89.6, 90.4), "-1"),
        ("near-orthogonal monoclinic b-mid 40/60/70 beta=90.6", (40., 60., 70., 90., 90.6, 90.), "2/m_uab"),
        ("near-orthogonal monoclinic b-shortest 60/40/70 beta=90.6", (60., 40., 70., 90., 90.6, 90.), "2/m_uab"),
        ("pseudo-cubic rhombohedral 50/50/50 alpha=89.5", (50., 50., 50., 89.5, 89.5, 89.5), "-3m_R"),
        ("pseudo-cubic rhombohedral 50/50/50 alpha=89.9 (the 0.1 deg boundary)", (50., 50., 50., 89.9, 89.9, 89.9), "-3m_R")]
NEAR_FLOOR = {}                    # every NEAR cell: 24/24

for name, cell, laue in AFFECTED + CONTROLS + PSEUDO + NEAR:
    print(f"\n{name}  (Laue {laue})")
    ops = {tuple(np.asarray(o, int).ravel()) for o in laue_ops(laue)}
    Mc = cell_to_Ar(*cell)
    shots = shots_for(cell)
    floor = 0 if (name, cell, laue) in PSEUDO else NEAR_FLOOR.get(name, N)
    kw = {"laue": laue} if (name, cell, laue) in NEAR else {}          # the original cells: no class, as before
    per = collections.Counter(classify(s.M, _relabel_like(np.asarray(M, float), Mc), ops) if M is not None else "None"
                              for s, M in zip(shots, (rg.index_known_gpu_cell(s.g, Mc, **kw) for s in shots)))
    check(f"per-frame index_known_gpu_cell: no non-lattice basis, >= {floor}/{N} equivalent",
          per["MISOR"] == 0 and per["EQUIV"] >= floor, dict(per))
    bat = collections.Counter(classify(s.M, _relabel_like(np.asarray(M, float), Mc), ops) if M is not None else "None"
                              for s, M in zip(shots, rgb.index_fused([s.g for s in shots], Mc, B=N, **kw)))
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
# A consensus or lock cell of an orthogonal lattice is skewed; the paper's pipelines and StreamDriver run on such
# cells, so they must not take the new path. The second is StreamDriver's actual GPU lock on the cxidb-17 480
# (0.27 deg tilt), which the first version of this gate (0.2 A) let through.
for label, cp in (("consensus-like 79.1/78.95/38.02/90.03/89.98/90.04", (79.1, 78.95, 38.02, 90.03, 89.98, 90.04)),
                  ("GPU stream lock 78.706/78.792/37.813/89.81/90.08/90.19",
                   (78.706, 78.792, 37.813, 89.8107, 90.0832, 90.1885))):
    L, c01, c02, c12, _ = rg._axes_from_cell(cell_to_Ar(*cp))
    check(f"lysozyme {label}: _both_hands is False", not rg._both_hands(float(L[2]), c01, c02, c12))
for name, cell, _ in AFFECTED:
    L, c01, c02, c12, _ = rg._axes_from_cell(cell_to_Ar(*cell))
    check(f"{name}: _both_hands is True", rg._both_hands(float(L[2]), c01, c02, c12))
for label, cp in (("consensus-like", (79.1, 78.95, 38.02, 90.03, 89.98, 90.04)),
                  ("GPU stream lock", (78.706, 78.792, 37.813, 89.8107, 90.0832, 90.1885))):
    L, c01, c02, c12, _ = rg._axes_from_cell(cell_to_Ar(*cp))
    check(f"lysozyme {label} with its class 4/mmm: _both_hands is False",
          not rg._both_hands(float(L[2]), c01, c02, c12, "4/mmm"))

print("\nthe declared class decides near 90 deg; without one the 1 deg tilt gate does (unchanged)")
for name, cell, laue in NEAR:
    L, c01, c02, c12, _ = rg._axes_from_cell(cell_to_Ar(*cell))
    check(f"{name}: _both_hands True with laue {laue!r}", rg._both_hands(float(L[2]), c01, c02, c12, laue))
    check(f"{name}: _both_hands False with no class", not rg._both_hands(float(L[2]), c01, c02, c12))

print("\n_ref_setting leaves a non-finite or singular basis alone (it used to; a raise would kill the batch)")
Mt = cell_to_Ar(*NEAR[0][1])
for label, Mb in (("NaN", np.full((3, 3), np.nan)), ("zero", np.zeros((3, 3)))):
    try:
        out = rg._ref_setting(Mb, Mt); ok = out is Mb
    except Exception as e:                                              # noqa: BLE001
        ok, out = False, repr(e)
    check(f"{label} basis returned unchanged", ok, "" if ok else out)

print("\n_ref_setting's sign-flip step: only angles the reference resolves (>= KC_FLIP_MIN_DEG from 90) count")


def _sf(cell):                                   # the reference basis, columns shortest-first (as the engines return)
    A = cell_to_Ar(*cell)
    return A[:, np.argsort(np.linalg.norm(A, axis=0))]


R = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))[0]
R = R * np.sign(np.linalg.det(R))                # a proper rotation: the same frame seen at an arbitrary orientation
for label, cell, D, want_flip in (
        ("triclinic 89.9/90/90.2 flipped (-a,-b,c)", (50., 60., 70., 89.9, 90., 90.2), (-1, -1, 1), True),
        ("triclinic 89.5/90/90.2 flipped (-a,b,-c)", (50., 60., 70., 89.5, 90., 90.2), (-1, 1, -1), True),
        ("monoclinic beta=105 (exact 90s) given the 2-fold (-a,b,-c)", (40., 60., 70., 90., 105., 90.), (-1, 1, -1), False),
        ("hexagonal 60/60/40 (columns c,a,b) given the flip of its two 90 deg cosines", (60., 60., 40., 90., 90., 120.),
         (1, -1, -1), False),
        ("hexagonal 60/60/40 (columns c,a,b) given a flip of its 120 deg cosine", (60., 60., 40., 90., 90., 120.),
         (-1, 1, -1), True),
        ("r199-like 27.92/62.95/60.11 (columns a,c,b) flipped in beta=90.07", (27.92, 62.95, 60.11, 90., 90.07, 90.),
         (-1, 1, -1), False)):
    S = _sf(cell); Mi = R @ S @ np.diag(D)
    out = rg._ref_setting(Mi, cell_to_Ar(*cell))
    got = not np.array_equal(out, Mi)
    back = np.allclose(out.T @ out, S.T @ S, atol=1e-6)        # back in a setting with the reference's metric
    check(f"{label}: {'back in the reference metric' if want_flip else 'left exactly as it is'}",
          (got and back) if want_flip else (not got), f"changed={got}")

print("\n_closest_setting (shortlist + exact einsum) picks what the full 3480-way search picks, ties included")
rs = np.random.default_rng(5); nbad = 0; ntot = 0
for cell in ((50., 60., 70., 80., 85., 95.), (79.1, 79.1, 38., 90., 90., 90.), (60., 60., 40., 90., 90., 120.),
             (27.24, 31.87, 34.23, 88.52, 108.53, 111.89), (50., 50., 50., 90., 90., 90.)):
    S = _sf(cell); G0 = S.T @ S; nrm = np.abs(G0).max()
    for t in range(40):
        Mb = S @ rg._UNIMOD[rs.integers(len(rg._UNIMOD))] @ (np.eye(3) + (t % 2) * rs.normal(0, 1e-3, (3, 3)))
        G = Mb.T @ Mb
        full = np.abs(np.einsum('kji,jl,klm->kim', rg._UNIMOD, G, rg._UNIMOD) - G0).max((1, 2)) / nrm
        k, dk = rg._closest_setting(G, G0, nrm); kf = int(np.argmin(full))
        nbad += (k != kf) or (dk != full[kf]); ntot += 1
check("same index and deviation on every metric (exact-tie metrics included)", nbad == 0, f"{nbad}/{ntot} differ")

print("\nStreamDriver passes its class for the primary cell only, not for a relocked extra cell")
dP = StreamDriver(Mt, PANELS, 0.1, 1.3, (NPAN, NPAN), dtype=np.uint16, B=4, dmin=2.0, use_gpu=False, laue="-1")
lyso = cell_to_Ar(78.706, 78.792, 37.813, 89.8107, 90.0832, 90.1885)
check("primary cell gets laue='-1'", dP._kc_kw(dP.Mc) == {"laue": "-1"} and dP._kc_kw() == {"laue": "-1"})
check("another cell gets no class", dP._kc_kw(lyso) == {})

print("\nthe CUDA-graph cache is keyed on the handedness branch, not on the cell alone")
P0 = rgb._cell_params(cell_to_Ar(*NEAR[0][1])); P1 = rgb._cell_params(cell_to_Ar(*NEAR[0][1]), laue="-1")
check("_cell_params carries both (P[-2]) and keeps nc last", P0[-2] is False and P1[-2] is True and P0[-1] == P1[-1])

print(f"\n{'FAILURES: ' + ', '.join(FAILS) if FAILS else 'ALL PASS'}")
sys.exit(1 if FAILS else 0)
