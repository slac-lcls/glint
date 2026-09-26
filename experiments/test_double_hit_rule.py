"""Does the gated double-hit rule count crystals rather than peaks?

The bare deflate-and-reindex rule was measured to be a peak-count artifact on real data (mfxl1038923
r0278, job 34409542: 95% real vs 93% azimuth-scrambled at a median 62-peak residual). The fix is
`multilattice.second_lattice_verdict`: a same-cell gate (a same-protein double SHARES lattice 1's
cell) plus an ORIENTATION gate (a mosaic tail of lattice 1 shares the cell too, so same_lattice
cannot police it -- but it sits a FEW DEGREES from lattice 1, while a second crystal is at a generic
angle).

The orientation gate is `misorientation_deg`, NOT the clone fraction that first filled that slot.
Both pass the synthetic tests below; on real data (job 34468402) the clone fraction was scored
against the angle and let 55 of 72 mosaic clones through while killing 19 of 135 genuine doubles, so
it was demoted to a diagnostic. The lesson is in this file deliberately: a gate that separates
cleanly on planted clones can still fail on real ones, and only an INDEPENDENT discriminator shows
it. These tests therefore check the angle gate and keep the clone fraction as a recorded value.

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
check("misorientation recovered", abs(v["misorientation"] - 25.0) < 0.5, v["misorientation"])
check("ACCEPTED", v["accepted"])

print("\nOPPORTUNISTIC DIFFERENT CELL: indexes the residual but is not the protein -> rejected")
M_arb = rot([3, 1, 0], 40.0) @ np.diag([33.0, 47.0, 71.0])
resid = peaks_of(M_arb, 20)                          # peaks genuinely on the wrong lattice
v = second_lattice_verdict(resid, M1, lambda r, n: [(M_arb, 1.0)])
check("raw fires (old rule would count this)", v["raw"])
check("cell gate rejects it", not v["cell_match"] and not v["accepted"])

print("\nMOSAIC-TAIL CLONE: a small-angle rotation of lattice 1 -> raw + cell fire, ANGLE rejects")
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
check("angle gate rejects it", not v["accepted"], f"misorientation {v['misorientation']:.2f} deg")
check("the angle IS the planted 1.8 deg", abs(v["misorientation"] - 1.8) < 0.3, v["misorientation"])

print("\nmisorientation is measured modulo the lattice's OWN symmetry")
# Negating two axes describes the SAME lattice; the angle must not change. Getting this wrong
# inflates small angles into large ones and would let every clone through the gate.
v2 = second_lattice_verdict(peaks_of(M2, 20), M1,
                            lambda r, n: [(M2 @ np.diag([1.0, -1.0, -1.0]), 1.0)])
check("sign-flipped axes give the same angle", abs(v2["misorientation"] - 25.0) < 0.5,
      v2["misorientation"])
check("and the same verdict", v2["accepted"])

print("\nthe angle survives an independently refined cell (1% off)")
v3 = second_lattice_verdict(peaks_of(M2, 20), M1, lambda r, n: [(M2 * 1.01, 1.0)])
check("1% cell mismatch does not corrupt the angle", abs(v3["misorientation"] - 25.0) < 1.0,
      v3["misorientation"])

print("\nclone_fraction is still RECORDED (demoted to a diagnostic, see the module docstring)")
resid = peaks_of(M2, 20)
cf = orientation_clone_fraction(resid, M2, M1)
check("reported for the true double", 0.0 <= cf <= 1.0, cf)

print("\nDEFLATION + verdict end to end: two crystals in one shot")
qA, qB = peaks_of(M1, 40), peaks_of(M2, 25)
resid = deflate_peaks(np.vstack([qA, qB]), M1)
check("deflation leaves ~B", abs(len(resid) - len(qB)) <= 3, (len(resid), len(qB)))
v = second_lattice_verdict(resid, M1, lambda r, n: [(M2, 1.0)])
check("second crystal accepted on the residual", v["accepted"])

print("\nsmall residual refuses quietly")
v = second_lattice_verdict(peaks_of(M2, 3), M1, lambda r, n: [(M2, 1.0)])
check("below min_peaks -> nothing fires", not v["raw"] and not v["accepted"])

print("\n50-seed sweep: doubles at random orientations accepted, mosaic tails rejected")
# Random orientations, not hand-picked axes: with 4 lattice symmetry ops the chance that a genuinely
# random second crystal falls under the 15 deg cut is 6.5e-5 (measured by Monte Carlo), so a sweep
# this size should lose none of them.
acc_true = clone_rej = 0
for s in range(50):
    r = np.random.default_rng(1000 + s)
    A = r.normal(size=(3, 3)); Q, R = np.linalg.qr(A); Q = Q * np.sign(np.diag(R))
    Mt = Q @ M1                                       # random orientation, same cell
    if second_lattice_verdict(peaks_of(Mt, 20), M1, lambda q, n: [(Mt, 1.0)])["accepted"]:
        acc_true += 1
    Mc = rot(r.normal(size=3), float(r.uniform(0.5, 4.0))) @ M1        # a mosaic tail, 0.5-4 deg
    if not second_lattice_verdict(peaks_of(Mc, 20), M1, lambda q, n: [(Mc, 1.0)])["accepted"]:
        clone_rej += 1
check("50/50 random-orientation doubles accepted", acc_true == 50, acc_true)
check("50/50 mosaic tails rejected", clone_rej == 50, clone_rej)

# ---------------------------------------------------------------------------------------------------
# misorientation_deg compares LATTICES, not bases. Until this was fixed it compared bases: it tried
# only signed axis permutations of the two incoming matrices, so the same lattice written in another
# basis -- or in the same basis with the opposite handedness, which buerger_reduce returns about half
# the time -- read tens of degrees away from itself. Every test above hands it matching bases, which
# is why none of them could see that. The two real pairs below come from the 23 Sep cxidb-17 replay
# (per-lattice + double-hit arm), where 22 of 178 recorded "second crystals" at >= 15 deg were
# within 15 deg of lattice 1.
from glint.multilattice import misorientation_deg  # noqa: E402


def cell(a, b, c, al, be, ga):
    al, be, ga = np.radians([al, be, ga])
    cx = c * np.cos(be); cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    return np.column_stack([[a, 0, 0], [b * np.cos(ga), b * np.sin(ga), 0],
                            [cx, cy, np.sqrt(c * c - cx * cx - cy * cy)]])


def random_basis(r):
    while True:                                       # integer, |det| = 1, entries up to 2: skewed
        T = r.integers(-2, 3, (3, 3)).astype(float)
        if abs(abs(np.linalg.det(T)) - 1) < 0.5:
            return T


def ang(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


F_ = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]).T
I_ = np.array([[-.5, .5, .5], [.5, -.5, .5], [.5, .5, -.5]]).T
C_ = np.array([[.5, .5, 0], [-.5, .5, 0], [0, 0, 1]]).T
LYS = cell(79.02, 79.02, 37.98, 90, 90, 90)
AP = cell(40, 50, 60, 80, 95, 105)
MP = cell(40, 50, 60, 90, 105, 90)
OP = cell(40, 50, 60, 90, 90, 90)
TI = cell(50, 50, 80, 90, 90, 90)
HP = cell(60, 60, 90, 90, 90, 120)
HR = cell(50, 50, 50, 70, 70, 70)
CP = cell(50, 50, 50, 90, 90, 90)
BRAVAIS = {                                           # name: (basis, proper holohedry order)
    "aP": (AP, 1), "mP": (MP, 2), "mC": (MP @ C_, 2), "oP": (OP, 4), "oC": (OP @ C_, 4),
    "oF": (OP @ F_, 4), "oI": (OP @ I_, 4), "tP": (LYS, 8), "tI": (TI @ I_, 8), "hP": (HP, 12),
    "hR": (HR, 6), "cP": (CP, 24), "cF": (CP @ F_, 24), "cI": (CP @ I_, 24)}
LAUE = {"aP": "-1", "mP": "2/m", "mC": "2/m", "oP": "mmm", "oC": "mmm", "oF": "mmm", "oI": "mmm",
        "tP": "4/mmm", "tI": "4/mmm", "hP": "6/mmm", "hR": "-3m_R", "cP": "m-3m", "cF": "m-3m",
        "cI": "m-3m"}
CONV = {"aP": AP, "mP": MP, "mC": MP, "oP": OP, "oC": OP, "oF": OP, "oI": OP, "tP": LYS, "tI": TI,
        "hP": HP, "hR": HR, "cP": CP, "cF": CP, "cI": CP}


def close_group(gens):
    G = [np.eye(3)]
    ch = True
    while ch:
        ch = False
        for g in list(G):
            for s in gens:
                h = s @ g
                if not any(np.allclose(h, x, atol=1e-10) for x in G):
                    G.append(h); ch = True
    return G


def independent_sym_ops(name):
    C = CONV[name]
    a, b, c = (C[:, i].copy() for i in range(3))
    laue = LAUE[name]
    if laue == "-1":
        gens = []
    elif laue == "2/m":
        gens = [rot(b, 180)]
    elif laue == "mmm":
        gens = [rot(a, 180), rot(b, 180)]
    elif laue == "4/mmm":
        gens = [rot(c, 90), rot(a, 180)]
    elif laue == "6/mmm":
        gens = [rot(c, 60), rot(a, 180)]
    elif laue == "-3m_R":
        gens = [rot((a + b + c).copy(), 120), rot((a - b).copy(), 180)]
    elif laue == "m-3m":
        gens = [rot((a + b + c).copy(), 120), rot(c, 90)]
    else:
        raise ValueError(laue)
    return close_group(gens)

print("\nmisorientation_deg: the lattice's own proper symmetry is found for all 14 Bravais classes")
bad = {k: len(independent_sym_ops(k)) for k, (_, n) in BRAVAIS.items() if len(independent_sym_ops(k)) != n}
check("independent proper holohedry order (1, 2, 4, 6, 8, 12, 24)", not bad, bad)
H = BRAVAIS["hP"][0]
g = misorientation_deg(H, rot(H[:, 2].copy(), 60) @ H)    # copy: rot() normalises its axis in place
check("hexagonal: 60 deg about c is the lattice itself (a signed permutation cannot express it)",
      g < 1e-4, g)

print("\nmisorientation_deg: known Laue classes supply the fixed proper group")
check("monoclinic laue=2/m keeps 180 deg about b at 0",
      misorientation_deg(MP, rot(MP[:, 1].copy(), 180) @ MP, laue="2/m") < 1e-4)
MC = cell(40, 55, 30, 90, 90, 105)
check("monoclinic laue=2/m_uac keeps 180 deg about c at 0 after reduction reorders axes",
      misorientation_deg(MC, rot(MC[:, 2].copy(), 180) @ MC, laue="2/m_uac") < 1e-4)
check("hexagonal laue=6/mmm keeps 60 deg about c at 0",
      misorientation_deg(H, rot(H[:, 2].copy(), 60) @ H, laue="6/mmm") < 1e-4)
check("rhombohedral laue=-3m_R keeps 120 deg about the 3-fold axis at 0",
      misorientation_deg(HR, rot((HR[:, 0] + HR[:, 1] + HR[:, 2]).copy(), 120) @ HR, laue="-3m_R") < 1e-4)
known_worst = max(misorientation_deg(M, S @ M, laue=LAUE[k])
                  for k, (M, _) in BRAVAIS.items() for S in independent_sym_ops(k))
check(f"known-laue path keeps every independent proper operator at 0 (max {known_worst:.1e} deg)",
      known_worst < 1e-4, known_worst)

print("\nmisorientation_deg: the same lysozyme lattice in another basis is 0 deg")
for name, T in [("a <-> b swap (left-handed)", np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]])),
                ("one flipped axis (left-handed)", np.diag([-1.0, 1, 1])),
                ("a+b, b, c", np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1.0]])),
                ("c first (the order buerger_reduce emits)", np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.0]]))]:
    a = misorientation_deg(LYS, LYS @ T)
    check(f"{name}: {a:.4f} deg", a < 1e-4, a)       # arccos near 1: ~1e-6 deg of float noise
r = np.random.default_rng(11)
worst = max(misorientation_deg(LYS, LYS @ random_basis(r)) for _ in range(50))
check("50 random skewed bases: all 0 deg", worst < 1e-4, worst)

print("\nmisorientation_deg: planted rotations recovered exactly in any basis, handedness, 1% cell drift")
for deg in (3.0, 20.0, 35.0):              # below the smallest symmetry-equivalent for this axis
    R = rot([1.0, 2.0, 3.0], deg)
    got = [misorientation_deg(LYS, R @ LYS @ T) for T in
           (np.eye(3), np.diag([-1.0, 1, 1]), np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]]))]
    check(f"{deg:g} deg in 3 bases -> {', '.join(f'{g:.3f}' for g in got)}",
          max(abs(g - deg) for g in got) < 1e-4, got)
g = misorientation_deg(LYS, rot([3.0, -1.0, 2.0], 25.0) @ (LYS * 1.01) @ np.diag([1.0, -1, 1]))
check(f"25 deg, 1% larger cell, left-handed basis -> {g:.3f}", abs(g - 25.0) < 0.5, g)

print("\nmisorientation_deg = brute-force minimum over the lattice's symmetry group (14 classes x 30)")
r, worst = np.random.default_rng(5), 0.0
for k, (M, _) in BRAVAIS.items():
    S = independent_sym_ops(k)
    for _ in range(30):
        R = rot(r.normal(size=3), r.uniform(0, 180))
        worst = max(worst, abs(misorientation_deg(M, R @ M @ random_basis(r))
                               - min(ang(R @ s) for s in S)))
check(f"max deviation {worst:.1e} deg", worst < 1e-4, worst)

print("\nclass operators are Miller-index operators: transposed, each one is a symmetry of the conventional cell")
# Composition with the metric-inferred operators masks a wrong operator table in misorientation_deg itself,
# so check the table directly: untransposed, the hexagonal 6-fold sends a to a - b (103.9 A on this cell).
from glint.lattice import standardize_axes  # noqa: E402
from glint.multilattice import _PROPER_LAUE_OPS, _canonical_basis, _is_metric_symmetry  # noqa: E402
bad = {}
for key, M in (("6/mmm", HP), ("6/m", HP), ("-3m1", HP), ("-31m", HP), ("-3", HP), ("4/mmm", LYS), ("mmm", OP)):
    Bn = standardize_axes(_canonical_basis(M)[0], laue=key)
    P = [np.asarray(S, int) for S in _PROPER_LAUE_OPS[key] if round(np.linalg.det(S)) == 1]
    kept = sum(_is_metric_symmetry(Bn, S.T) for S in P)
    if kept != len(P):
        bad[key] = (kept, len(P))
check("every transposed proper operator preserves the conventional metric (7 classes)", not bad, bad)

print("\nmisorientation_deg: a known class applies in its conventional setting, whatever the caller's axis order")
# The stream driver's lattice 1 can arrive with c first. Conjugating the class operators from the
# caller's order put the 4-fold about an a axis, and two real cxidb-17 pairs read 45 and 60 degrees
# instead of 97 and 91 (glint#207). Expected values: the brute-force minimum over the fixed group.
Lc = LYS[:, [2, 0, 1]]
for deg, axis in ((62.3, [-0.54, -0.32, 0.41]), (84.7, [-0.13, 1.37, -0.67]), (118.1, [0.9, 0.09, -0.74])):
    R = rot(axis, deg)
    want = min(ang(R @ s) for s in independent_sym_ops("tP"))
    g4, g0 = misorientation_deg(Lc, R @ LYS, laue="4/mmm"), misorientation_deg(Lc, R @ LYS)
    check(f"c-first lattice 1, planted {deg:g} deg: 4/mmm {g4:.2f}, no class {g0:.2f}, expected {want:.2f}",
          abs(g4 - want) < 1e-4 and abs(g0 - want) < 1e-4, (g4, g0, want))
REAL2 = {   # cxidb-17 replay (per-lattice + double-hit arm): frame -> (M1, M2, independent near-isometry search)
    169: ([[3.1648, 30.5757, 71.8533], [29.727, -45.5595, 13.5818], [22.8133, 57.9762, -27.6267]],
          [[22.0375, -61.0171, -20.375], [20.5633, 47.3694, -46.2648], [-23.1346, -15.6583, -60.8078]], 91.03),
    368: ([[-34.905, -18.8366, 23.8616], [2.5453, -69.2417, -36.8297], [13.0434, -33.6004, 63.9082]],
          [[14.3348, -1.1264, -72.9918], [10.8812, 74.8481, 7.8677], [33.2345, -23.9301, 28.7678]], 97.10)}
for fr, (A, Bm, want) in REAL2.items():
    g4, g0 = misorientation_deg(A, Bm, laue="4/mmm"), misorientation_deg(A, Bm)
    check(f"real frame {fr} (a genuine double hit): 4/mmm {g4:.2f}, no class {g0:.2f}, expected {want:.2f}",
          abs(g4 - want) < 0.05 and abs(g0 - want) < 0.05, (g4, g0))

print("\nmisorientation_deg: a near-orthogonal triclinic cell does not gain a false 2-fold when its class is given")
# Without a class the near-90-degree sign tolerance (which real orthorhombic data need, below) would grant
# this cell a 2-fold it does not have (glint#207 review). The documented remedy is laue="-1".
T = cell(40, 50, 60, 88, 92, 97)
g = misorientation_deg(T, rot(T[:, 2].copy(), 178.0) @ T, laue="-1")
check(f"triclinic, laue='-1': 178 deg about c stays 178, not a fake 2-fold -> {g:.2f}", abs(g - 178.0) < 1e-4, g)
# Two noisy refinements of it (angles 88/92 vs 88.6/91.4): a signed metric match finds a pseudo 2-fold between
# them. The class path must use ONE basis correspondence expanded by the class's group, not every
# tolerance-inferred operation (glint#207 review), so laue="-1" still reads ~178.
T2n = cell(40.2, 49.8, 60.1, 88.6, 91.4, 97.3)
g = misorientation_deg(T, rot(T[:, 2].copy(), 178.0) @ T2n, laue="-1")
check(f"noisy triclinic pair, laue='-1': ~178 deg, no pseudo 2-fold -> {g:.2f}", abs(g - 178.0) < 1.0, g)

print("\nmisorientation_deg: a lattice refined to 93.6 deg still finds its symmetry (relaxed tolerance)")
# same_lattice compares |cos|, so two refinements on opposite sides of 90 deg pass the cell gate while
# failing a signed-cosine test at the first tolerance; one such pair occurred on the replay (frame 46).
M1d, M2d = cell(76.55, 81.29, 37.0, 90, 90, 93.6), rot([1.0, 1.0, 0.0], 40.0) @ cell(78.6, 78.8, 37.7, 90, 90, 90)
g = misorientation_deg(M1d, M2d, laue="4/mmm")
check(f"40 deg between distorted cells -> {g:.2f}", abs(g - 40.0) < 4.0, g)

print("\nmisorientation_deg: 90-degree angles refined to opposite sides of 90 keep the full symmetry")
# A refined 90-degree angle comes back at 92 in one cell and 88 in the other (mfxl1038923: up to ~7 degrees).
# A signed-cosine match then dropped the 2-folds that flip that sign, and a 2-degree mosaic pair read
# as the symmetry-equivalent 178 degrees -- 42 pairs of the mfxl census did this with the first version.
O1, O2 = cell(43.6, 67.8, 89.1, 91.0, 89.5, 92.0), cell(43.9, 67.5, 89.4, 89.2, 90.6, 88.0)
for name, T in [("same handedness", np.eye(3)), ("opposite handedness", np.diag([1.0, 1, -1]))]:
    for laue in ("mmm", None):
        g = misorientation_deg(O1, rot([2.0, -1.0, 0.5], 2.0) @ O2 @ T, laue=laue)
        check(f"2 deg mosaic pair, {name}, laue={laue!r} -> {g:.2f}", g < 5.0, g)

print("\nmisorientation_deg on two REAL replay pairs the old version read as ~90 deg")
REAL = {   # frame: (M1, M2, what the old version recorded)
    155: ([[19.941, 74.9458, -6.3703], [6.8435, 11.2947, 37.1143], [74.6988, -21.8871, -2.6321]],
          [[-6.3436, 21.0358, -74.9919], [37.1065, 9.6343, -11.2021], [-2.8829, 75.48, 22.204]], 89.97),
    320: ([[-25.3096, -73.3133, 6.8433], [56.6332, -9.3564, 25.7882], [-48.299, 27.5557, 26.8039]],
          [[8.5174, 24.364, 72.9145], [25.0442, -58.4023, 7.0691], [27.1343, 46.5569, -29.3891]], 87.91)}
for fr, (A, Bm, was) in REAL.items():
    g = misorientation_deg(A, Bm, laue="4/mmm")
    hand = "opposite" if np.linalg.det(A) * np.linalg.det(Bm) < 0 else "same"
    check(f"frame {fr} ({hand} handedness): recorded {was} deg, now {g:.2f} deg -> not a second crystal",
          g < 5.0, g)
    g0 = misorientation_deg(A, Bm)
    check(f"frame {fr}, no class given (the default path): {g0:.2f} deg", abs(g0 - g) < 1e-6, (g0, g))
    check(f"frame {fr}: second_lattice_verdict rejects it",
          not second_lattice_verdict(peaks_of(np.asarray(Bm), 20), A,
                                     lambda q, n, Bm=Bm: [(np.asarray(Bm), 1.0)],
                                     laue="4/mmm")["accepted"])

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
