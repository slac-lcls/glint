"""Does the gated double-hit rule count crystals rather than peaks?

The bare deflate-and-reindex rule was measured to be a peak-count artifact on real data (mfxl1038923
r0278, job 34409542: 95% real vs 93% azimuth-scrambled at a median 62-peak residual). The fix is
`multilattice.second_lattice_verdict`: same-cell gate (a same-protein double SHARES lattice 1's
cell) plus an ORIENTATION clone gate (a mosaic tail of lattice 1 also shares the cell -- same_lattice
cannot police it -- but its "second lattice" re-indexes peaks that are lattice-1 points just outside
the deflation tolerance, which orientation_clone_fraction detects).

CPU-only: the indexer is faked per case, so what is under test is exactly the gating logic.
Convention throughout: M columns are real-space cell vectors, hkl = q @ M, q = hkl @ inv(M).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint.multilattice import (deflate_peaks, orientation_clone_fraction, scramble_azimuth,
                                second_lattice_verdict)

FAILS = []
rng = np.random.default_rng(7)


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


def rot(axis, deg):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    a = np.radians(deg); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                                       [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


M1 = np.diag([40.0, 50.0, 60.0])                     # real-space columns; hkl = q @ M1


def peaks_of(M, n=25, noise=1e-4):
    hkl = rng.integers(-6, 7, (n, 3)).astype(float)
    hkl = hkl[np.abs(hkl).sum(1) > 0][:n]
    return hkl @ np.linalg.inv(M) + rng.normal(0, noise, (len(hkl), 3))


print("scramble_azimuth preserves |q| and q_z exactly")
q = peaks_of(M1, 40)
qs = scramble_azimuth(q, np.random.default_rng(1))
check("|q| preserved", np.allclose(np.linalg.norm(q, axis=1), np.linalg.norm(qs, axis=1)))
check("q_z preserved", np.allclose(q[:, 2], qs[:, 2]))
check("coherence destroyed (transverse moved)", not np.allclose(q[:, 0], qs[:, 0]))

print("\nTRUE DOUBLE: same cell, rotated 25 deg -> accepted")
M2 = rot([1, 2, 3], 25.0) @ M1                       # same cell, genuinely new orientation
resid = peaks_of(M2, 20)
v = second_lattice_verdict(resid, M1, lambda r, n: [(M2, 1.0)])
check("raw fires", v["raw"])
check("cell matches", v["cell_match"])
check("not a clone", v["clone_fraction"] < 0.5, v["clone_fraction"])
check("ACCEPTED", v["accepted"])

print("\nOPPORTUNISTIC DIFFERENT CELL: indexes the residual but is not the protein -> rejected")
M_arb = rot([3, 1, 0], 40.0) @ np.diag([33.0, 47.0, 71.0])
resid = peaks_of(M_arb, 20)                          # peaks genuinely on the wrong lattice
v = second_lattice_verdict(resid, M1, lambda r, n: [(M_arb, 1.0)])
check("raw fires (old rule would count this)", v["raw"])
check("cell gate rejects it", not v["cell_match"] and not v["accepted"])

print("\nMOSAIC-TAIL CLONE: a small-angle rotation of lattice 1 -> raw fires, clone gate rejects")
# The real failure mode: a mosaic block a degree or two off lattice 1. Its peaks index cleanly
# under the rotated basis M2c, while under M1 they sit just OUTSIDE the deflation tolerance
# (dev ~ |hkl| * theta) -- so they survive deflation and the bare rule counts a second crystal.
M2c = rot([0, 1, 1], 1.8) @ M1
hkl = rng.integers(-9, 10, (60, 3)).astype(float)
hkl = hkl[np.abs(hkl).max(1) >= 5]                    # high orders: dev under M1 past the tol
q_all = hkl @ np.linalg.inv(M2c)
dev1 = np.abs(q_all @ M1 - np.round(q_all @ M1)).max(1)
resid = q_all[(dev1 > 0.15) & (dev1 < 0.35)]          # exactly the peaks deflation leaves behind
assert len(resid) >= 8, len(resid)
v = second_lattice_verdict(resid, M1, lambda r, n: [(M2c, 1.0)])
check("raw fires (old rule counts the mosaic tail as a double)", v["raw"])
check("cell matches (same_lattice cannot police this)", v["cell_match"])
check("clone gate rejects it", v["clone_fraction"] >= 0.5 and not v["accepted"],
      f"clone_fraction {v['clone_fraction']:.2f}")

print("\nclone gate does NOT fire on the true double")
resid = peaks_of(M2, 20)
cf = orientation_clone_fraction(resid, M2, M1)
check("true double clone_fraction low", cf < 0.5, cf)

print("\nDEFLATION + verdict end to end: two crystals in one shot")
qA, qB = peaks_of(M1, 40), peaks_of(M2, 25)
resid = deflate_peaks(np.vstack([qA, qB]), M1)
check("deflation leaves ~B", abs(len(resid) - len(qB)) <= 3, (len(resid), len(qB)))
v = second_lattice_verdict(resid, M1, lambda r, n: [(M2, 1.0)])
check("second crystal accepted on the residual", v["accepted"])

print("\nsmall residual refuses quietly")
v = second_lattice_verdict(peaks_of(M2, 3), M1, lambda r, n: [(M2, 1.0)])
check("below min_peaks -> nothing fires", not v["raw"] and not v["accepted"])

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
