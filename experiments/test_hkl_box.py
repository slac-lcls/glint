"""Every |q| <= qmax Miller-index enumeration must cover the whole sphere for OBLIQUE cells.

REGRESSION. With R = inv(M) (rows a*, b*, c*) and q = hkl @ R, each index is h_i = q . a_i, where a_i
is the i-th real-space axis (column i of M). So the per-axis bound that holds for every cell is
|h_i| <= qmax * |a_i|. GLINT bounded the box by qmax / |a*_i| instead. That is the same number for an
orthogonal axis, and smaller by cos(angle(a_i, a*_i)) otherwise, so on oblique cells the box cut off
part of the resolution sphere. The +1 slack hid small obliquity: monoclinic beta <= ~110 deg lost
nothing. Hexagonal/trigonal cells, monoclinic beta >~ 115 deg, general triclinic cells and the
blind-mode primitive basis of a centred lattice lost high-resolution reflections. The same bound was
in four functions, all checked here:
  - predict._hkl_grid: predict_spots (the offline --integrate predictor), HKLGrid (StreamDriver's
    per-frame predictor) and synth_sfx;
  - stream_driver.theoretical_unique: the completeness denominator;
  - multishot.reference_lattice;
  - alias_gate.tightness: the node count behind `occupancy`. There the undercount made the score
    depend on the BASIS. On lysozyme-like cells it made AliasGate refuse the true cell (or adopt a
    half-volume one), because the skewed index-2 derivatives had their node count cut short.

The reference is independent of all of those boxes. It enumerates a cube of half-width
ceil(qmax / sigma_min(R)) + 1: since |q| = |hkl @ R| >= sigma_min(R) |hkl|, no hkl outside that cube
can be in the sphere. Nothing it computes uses a per-axis bound. The guards that already existed
could not catch this. test_streamdriver_laue's reference sphere is built with _hkl_grid itself, and
test_axis_standardizer compares HKLGrid with predict_spots, which share the box.

Exact |q| == qmax (and |exc| == tol) ties are excluded from the set comparisons: which side of the cut
a tie lands on is decided by rounding, not by the box.

Cells, all at dmin 2.0 A (the default everywhere):
  oblique -- triclinic 40/50/60 (105, 95, 120); the reduced primitive basis of body-centred cubic
             insulin (68.3 A, all angles 109.47), a basis blind mode can hand back for a cI crystal;
             monoclinic P21 60/40/70 beta 120; hexagonal 60/60/40 gamma 120;
  controls -- lysozyme 79.1/79.1/38.0 and P212121 40/60/80. The old bound already got these right
             and they must stay exact.
Then the alias gate: tightness is basis-invariant and matches a brute-force node count on a
tetragonal 79/79/38 leader and its whole index-2 family; and confirm_frames on 24 sparse stills of
that lattice CONFIRMS the true leader (the old count refused it), while super-cell leaders are still
refused and adopt=True still recovers the true cell.

Run: `PYTHONPATH=. python experiments/test_hkl_box.py` (exit 1 on any failure). numpy + scipy only.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import glint.alias_gate as ag
from glint.lattice import buerger_reduce, cell_to_Ar, random_rotation
from glint.multishot import reference_lattice, same_lattice
from glint.predict import _hkl_grid, predict_spots, project_q, recip_from_M
from glint.stream_driver import HKLGrid, _asu_key, laue_ops, theoretical_unique

T0 = time.time()
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


DMIN = 2.0
QMAX = 1.0 / DMIN
TIE = 1e-9                     # relative width of the "exactly on the cut" band excluded from set checks
# One flat 2100 px x 75 um panel at 0.10 m, 1.305 A: edge d 1.99 A, corner d 1.60 A, so the whole
# dmin-2.0 sphere's Ewald slice is on the panel and the outer shell is where the old box lost rows.
PANELS = [dict(name="p0", fs=np.array([1.0, 0.0, 0.0]), ss=np.array([0.0, 1.0, 0.0]), res=1.0 / 75e-6,
               cx=-1050.0, cy=-1050.0, coffset=0.0, min_fs=0, max_fs=2099, min_ss=0, max_ss=2099)]
CLEN, WAVE = 0.10, 1.305

# name -> (cell, Laue classes to count theoretical_unique in, oblique?)
CELLS = {
    "triclinic 40/50/60 (105,95,120)":          ((40, 50, 60, 105, 95, 120), ("-1",), True),
    "cI insulin blind primitive 68.3 (109.47)": ((68.3, 68.3, 68.3, 109.47, 109.47, 109.47), ("-1",), True),
    "monoclinic P21 60/40/70 beta=120":         ((60, 40, 70, 90, 120, 90), ("-1", "2/m"), True),
    "hexagonal 60/60/40 gamma=120":             ((60, 60, 40, 90, 90, 120), ("-1", "6/mmm"), True),
    "CONTROL lysozyme 79.1/79.1/38.0":          ((79.1, 79.1, 38.0, 90, 90, 90), ("-1", "4/mmm"), False),
    "CONTROL P212121 40/60/80":                 ((40, 60, 80, 90, 90, 90), ("-1", "mmm"), False),
}


def key3(g):
    g = np.asarray(g, np.int64)
    return (g[:, 0] + 4096) * (1 << 26) + (g[:, 1] + 4096) * (1 << 13) + (g[:, 2] + 4096)


def brute_sphere(R, qmax):
    """(hkl, q, |q|) for every nonzero hkl with |hkl @ R| <= qmax*(1+TIE), from the singular-value cube."""
    B = int(np.ceil(qmax / np.linalg.svd(R, compute_uv=False).min())) + 1
    r = np.arange(-B, B + 1)
    g = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    g = g[np.any(g != 0, axis=1)]
    q = g @ R
    qn = np.sqrt(np.einsum("ij,ij->i", q, q))
    k = qn <= qmax * (1 + TIE)
    return g[k], q[k], qn[k]


def diff_counts(got_hkl, ref_hkl, ref_tie):
    """(missing, extra): reference rows absent from `got` (ties excluded), and `got` rows absent from
    the reference (which includes the tie band, so a tie row is never an 'extra')."""
    kg, kr = key3(got_hkl), key3(ref_hkl)
    missing = int((~np.isin(kr[~ref_tie], kg)).sum())
    extra = int((~np.isin(kg, kr)).sum())
    return missing, extra


def brute_predict(gb, R, qmax, tol):
    """Brute-force predict_spots: Ewald gate + projection over the brute-force sphere. Returns (hkl, tie)
    where tie marks rows within the |q| == qmax or |exc| == tol band."""
    q = gb @ R
    qn2 = np.einsum("ij,ij->i", q, q)
    exc = q[:, 2] + 0.5 * WAVE * qn2
    sel = (np.abs(exc) < tol * (1 + TIE)) & (qn2 <= (qmax * (1 + TIE)) ** 2)
    _, _, pan = project_q(q[sel], PANELS, CLEN, WAVE)
    on = pan >= 0
    hkl = gb[sel][on]
    qs, es = qn2[sel][on], np.abs(exc[sel][on])
    tie = (np.abs(np.sqrt(qs) - qmax) <= qmax * TIE) | (np.abs(es - tol) <= tol * TIE)
    return hkl, tie


# ------------------------------------------------------------------------- 1. the enumerations
print(f"== 1. hkl box vs brute-force sphere, dmin {DMIN} A ==")
for name, (cell, laues, oblique) in CELLS.items():
    M = cell_to_Ar(*cell)
    R = recip_from_M(M)
    gb, qb, qnb = brute_sphere(R, QMAX)
    tie = np.abs(qnb - QMAX) <= QMAX * TIE
    inside = ~tie & (qnb <= QMAX)
    need = np.abs(gb).max(0).tolist()
    print(f" {name}: {int(inside.sum())} hkl strictly inside the sphere, max |h,k,l| needed {need}")

    g, _ = _hkl_grid(R, QMAX)
    m, x = diff_counts(g, gb, tie)
    check(f"{name}: _hkl_grid == sphere", m == 0 and x == 0, f"missing {m}, extra {x}")

    G = HKLGrid(M, DMIN, gpu=False)
    m, _ = diff_counts(G.g, gb, tie)
    check(f"{name}: HKLGrid(gpu=False).g covers the sphere", m == 0, f"missing {m}")

    V = reference_lattice(M, QMAX)
    m, x = diff_counts(np.rint(V @ M).astype(np.int64), gb, tie)
    check(f"{name}: multishot.reference_lattice == sphere", m == 0 and x == 0, f"missing {m}, extra {x}")

    for L in laues:
        ops = laue_ops(L)
        lo = int(np.unique(_asu_key(gb[inside], ops)).size)
        hi = int(np.unique(_asu_key(gb, ops)).size)
        n = theoretical_unique(M, DMIN, ops)
        check(f"{name}: theoretical_unique[{L}] == brute force", lo <= n <= hi,
              f"{n} vs brute {lo}" + (f"..{hi}" if hi != lo else "") + f" (undercount {lo - n})")

    rng = np.random.default_rng(20261001)
    mp = mg = xp = xg = tot = 0
    for _ in range(4):
        Rf = recip_from_M(random_rotation(rng) @ M)
        for tol, getter, acc in ((0.006, lambda: predict_spots(Rf, PANELS, CLEN, WAVE, dmin=DMIN, tol=0.006,
                                                              is_recip=True), "p"),
                                 (0.002, lambda: G.predict(Rf, PANELS, CLEN, WAVE, tol=0.002, is_recip=True), "g")):
            ref, rtie = brute_predict(gb, Rf, QMAX, tol)
            out = getter()
            m, x = diff_counts(np.c_[out["h"], out["k"], out["l"]], ref, rtie)
            if acc == "p":
                mp += m; xp += x; tot += int((~rtie).sum())
            else:
                mg += m; xg += x
    check(f"{name}: predict_spots == brute-force prediction (4 orientations, tol 0.006)",
          mp == 0 and xp == 0, f"missing {mp} of {tot}, extra {xp}")
    check(f"{name}: HKLGrid.predict == brute-force prediction (4 orientations, tol 0.002)",
          mg == 0 and xg == 0, f"missing {mg}, extra {xg}")
print(f" [{time.time() - T0:.1f} s]")


# ------------------------------------------------------------------------- 2. alias_gate.tightness
def brute_npred(M, qmax):
    _, _, qn = brute_sphere(recip_from_M(M), qmax)
    return int(((qn > 1e-9) & (qn <= qmax)).sum())


def n_occupied(M, Q, hkl_tol=0.15):
    hf = Q @ M
    h = np.round(hf)
    inl = np.abs(hf - h).max(1) < hkl_tol
    return len({tuple(x) for x in h[inl].astype(int)})


print("\n== 2. alias_gate.tightness: node count == brute force, and invariant under a change of basis ==")
TET = cell_to_Ar(79.0, 79.0, 38.0, 90, 90, 90)
rng = np.random.default_rng(7)
Mt = random_rotation(rng) @ TET
_, qnodes, qn_all = brute_sphere(recip_from_M(Mt), 0.30)
qnodes = qnodes[(qn_all > 1e-9) & (qn_all <= 0.30)]
# noise-free nodes plus 1e-6/A of jitter: far below hkl_tol, but it keeps the largest |q| (which sets
# tightness's qmax) off any exact node radius, so the count has no ties to round either way
Q = qnodes[rng.choice(len(qnodes), 150, replace=False)] + rng.normal(0.0, 1e-6, (150, 3))
qmax_Q = float(np.linalg.norm(Q, axis=1).max())
U_LIST = [np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]], float),
          np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], float),
          np.array([[1, 0, 1], [0, 1, -1], [0, 0, 1]], float)]
fam = ag.derivative_lattices(Mt, 2)
bad_count, bad_basis, worst = [], [], 1.0
for k, D in enumerate(fam):
    occ_ref = n_occupied(D, Q) / max(brute_npred(D, qmax_Q), 1)
    s0, c0, o0 = ag.tightness(D, Q)
    if abs(o0 - occ_ref) > 1e-12 * occ_ref:
        bad_count.append(k)
        worst = max(worst, o0 / occ_ref)
    for U in U_LIST:
        s1, c1, o1 = ag.tightness(D @ U, Q)
        if not (c1 == c0 and o1 == o0):
            bad_basis.append(k)
            break
check(f"tightness occupancy == occupied / brute-force node count on all {len(fam)} members of the "
      f"79/79/38 index-2 family", not bad_count,
      f"{len(bad_count)} members off, worst occupancy {worst:.3f}x the true value")
check(f"tightness identical for D and D @ U (3 unimodular U) on all {len(fam)} members",
      not bad_basis, f"{len(bad_basis)} members change with the basis")
print(f" [{time.time() - T0:.1f} s]")


# ------------------------------------------------------------------------- 3. AliasGate.confirm_frames
print("\n== 3. AliasGate.confirm_frames on 24 sparse stills of 79/79/38 (45 peaks, |q| <= 0.33) ==")
NF, NPK, QS = 24, 45, 0.33


def stills(Mtrue, seed):
    r = np.random.default_rng(seed)
    out = []
    for _ in range(NF):
        Rot = random_rotation(r)
        _, qq, qn = brute_sphere(recip_from_M(Rot @ Mtrue), QS)
        qq = qq[(qn > 1e-9) & (qn <= QS)]
        out.append((qq[r.choice(len(qq), NPK, replace=False)] + r.normal(0.0, 0.0008, (NPK, 3)), Rot))
    return out


FR = stills(TET, 20261001)
SUPER_AX = TET @ np.diag([2.0, 1, 1])
SUPER_FD = TET @ np.array([[2.0, 1, 0], [0, 1, 0], [0, 0, 1]])
# adopt=True reaches the same verdict as adopt=False (it only changes what is returned), so each
# super-cell needs one run per question: is it refused, and does adopt recover the true cell.
for lname, L, adopt, want in (("TRUE", TET, False, "confirm"),
                              ("super-cell M @ diag(2,1,1)", SUPER_AX, False, "refuse"),
                              ("super-cell M @ diag(2,1,1)", SUPER_AX, True, "adopt TRUE"),
                              ("super-cell M @ [[2,1,0],[0,1,0],[0,0,1]]", SUPER_FD, True, "adopt TRUE")):
    Lr = buerger_reduce(L)                       # the basis a blind indexer would hand back
    gate = ag.AliasGate(adopt=adopt)
    out = gate.confirm_frames(Lr, [(q, Rot @ Lr) for q, Rot in FR])
    i = gate.info
    got = (None if out is None else "TRUE" if same_lattice(out, TET) else
           f"V/V_true {abs(np.linalg.det(out) / np.linalg.det(TET)):.2f}")
    tag = f"leader {lname}, adopt={adopt}"
    print(f"   {tag}: {i['verdict']} {i['frames_beaten']}/{i['frames_tested']} beaten, "
          f"median best/leader {i['median_best_over_leader']:.3f} -> {got}")
    if want == "confirm":
        check(f"{tag}: confirms the true cell", i["verdict"] == "confirm" and got == "TRUE",
              f"{i['verdict']} ({i['frames_beaten']}/{i['frames_tested']} beaten) -> {got}")
    elif want == "refuse":
        check(f"{tag}: refused", i["verdict"] == "refuse" and out is None, f"{i['verdict']} -> {got}")
    else:
        check(f"{tag}: adopts the true cell", i["verdict"] == "adopt" and got == "TRUE",
              f"{i['verdict']} -> {got}")

print(f"\n[{time.time() - T0:.1f} s]")
if FAILS:
    print(f"{len(FAILS)} FAILED:\n  " + "\n  ".join(FAILS))
    sys.exit(1)
print("ALL PASS")
