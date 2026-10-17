"""CPU tests for glint.predict.write_solution_file: the CrystFEL --indexing=file solution file puts the axes in
the setting its lattice code names.

WHY THIS EXISTS. For monoclinic codes the writer used to pass the known-cell engine's column order straight
through. On LUTE's cxil1015922 r136 test (C2, 111.94/172.23/41.23, beta 106.2 on b) every line came out as
41/111/172 with the 106-degree angle between a and b, labelled ``mCb`` (unique axis b), and CrystFEL's cell
check rejected all 27 frames. With a reference cell the writer now picks the proper signed permutation of the
columns closest to it; tetragonal / orthorhombic / hexagonal codes keep their standard setting, byte for byte.

Without a reference cell, triclinic, primitive monoclinic and rhombohedral codes used to get the length-only rule
frame by frame, which keeps each frame's column signs. On mfx101343025 r194 (28.0/62.5/60.9, beta 90.8, written
as ``mPb`` with no --cell) 101 of 229 frames came out at beta 89.2, 180 degrees about a from the rest: a setting
2/m does not relate to theirs. Every frame of such a run now gets the run's own setting.

  PYTHONPATH=. python experiments/test_solution_file.py      # exit 0 = all pass
"""
import io
import itertools
import os
import sys
import tempfile
from contextlib import redirect_stderr

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint.lattice import cell_to_Ar                                    # noqa: E402
from glint.predict import _canonical_axes, write_fromfile, write_solution_file   # noqa: E402
from glint.stream import has_cell, write_stream, written_note           # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


def rot(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def params(A):
    L = np.linalg.norm(A, axis=0)
    ang = lambda i, j: float(np.degrees(np.arccos(A[:, i] @ A[:, j] / L[i] / L[j])))   # noqa: E731
    return np.r_[L, ang(1, 2), ang(0, 2), ang(0, 1)]


def written(path):
    """Real-space bases (columns a, b, c in A) and codes from a solution file."""
    out = []
    for line in open(path).read().splitlines():
        t = line.split()
        B = np.array([float(x) for x in t[2:11]]).reshape(3, 3).T / 10.0      # columns a*, b*, c* in 1/A
        out.append((np.linalg.inv(B).T, t[-1]))
    return out


def run(results, code, ref=None):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.sol")
        err = io.StringIO()
        with redirect_stderr(err):
            n = write_solution_file(results, p, code, ref_cell=ref)
        return n, written(p), open(p).read(), err.getvalue()


C2 = cell_to_Ar(111.94, 172.23, 41.23, 90, 106.2, 90)       # cxil1015922 'B2', unique axis b
P21 = cell_to_Ar(28.00, 62.50, 60.90, 90, 90.8, 90)         # mfx101343025 r194
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)

# The engine's column order on r136: (c, a, b) -- a cyclic permutation, so still right-handed -- and an odd
# permutation with one axis negated, which is right-handed too.
CASES = [("cyclic (c, a, b)", lambda A: A[:, [2, 0, 1]]),
         ("odd (b, a, c) with -c", lambda A: A[:, [1, 0, 2]] * np.array([1, 1, -1]))]

print("monoclinic codes with the reference cell: the reference's setting, the same orientation")
for cell, code, name in ((C2, "mCb", "C2 111.94/172.23/41.23 beta 106.2"), (P21, "mPb", "P21 28.0/62.5/60.9 beta 90.8")):
    for label, perm in CASES:
        R = [rot(s) for s in range(4)]
        res = [{"image": "x.cxi", "event": i, "M": perm(Ri @ cell)} for i, Ri in enumerate(R)]
        n, W, _, err = run(res, code, ref=cell)
        good = all(np.allclose(params(A), params(cell), atol=1e-4) for A, _ in W)
        check(f"{name}, engine order {label}: lengths and beta as the reference", good and n == 4,
              [np.round(params(A), 2) for A, _ in W[:1]])
        # R @ reference, up to the 2-fold about b (a -> -a, c -> -c): a proper operator of 2/m that keeps the metric,
        # so either is the same crystal in the same setting.
        check(f"{name}, {label}: the written basis is R @ reference up to the 2-fold about b (right-handed)",
              all(any(np.allclose(A, Ri @ cell @ np.diag(s), atol=1e-6) for s in ((1, 1, 1), (-1, 1, -1)))
                  and np.linalg.det(A) > 0 for (A, _), Ri in zip(W, R)))
        check(f"{name}, {label}: code written as given, nothing on stderr", all(c == code for _, c in W) and not err, err)

print("\nrhombohedral hR with the reference cell (standardize_axes leaves -3m_R as handed in)")
HR = cell_to_Ar(50.0, 50.0, 50.0, 75.0, 75.0, 75.0)
# (b, a, -c): right-handed, but the negated axis turns two of the 75-degree angles into 105 degrees
res = [{"image": "x.cxi", "event": i, "M": rot(30 + i) @ HR[:, [1, 0, 2]] * np.array([1, 1, -1])} for i in range(3)]
n, W, _, err = run(res, "hR", ref=HR)
check("hR: the reference's angles (75, 75, 75), right-handed",
      all(np.allclose(params(A), params(HR), atol=1e-4) and np.linalg.det(A) > 0 for A, _ in W) and not err,
      [np.round(params(A), 2) for A, _ in W[:1]])

print("\nbase-centered code without the reference cell: the old behavior, and a warning")
res = [{"image": "x.cxi", "event": 0, "M": rot(9) @ C2[:, [2, 0, 1]]}]
n, W, _, err = run(res, "mCb")
check("the axes stay in the engine's order (the r136 failure, reproduced)",
      np.allclose(params(W[0][0])[:3], [41.23, 111.94, 172.23], atol=1e-3), np.round(params(W[0][0]), 2))
check("a monoclinic code without a reference says so on stderr", "without a reference cell" in err, err)
n, W, _, err = run(res, "aP")
check("a triclinic code without a reference writes silently, as before", not err, err)

print("\ntetragonal / orthorhombic / hexagonal codes are unchanged, reference or not")
res = [{"image": "x.cxi", "event": i, "M": rot(20 + i) @ LYSO[:, [2, 0, 1]]} for i in range(3)]
_, _, txt_ref, _ = run(res, "tPc", ref=LYSO)
_, W, txt_none, _ = run(res, "tPc")
check("tPc with and without ref_cell: byte-identical", txt_ref == txt_none)
check("tPc still puts the 4-fold axis (37.98) in c", all(abs(params(A)[2] - 37.98) < 1e-3 for A, _ in W))

print("\na frame that is no permutation of the reference is left alone, with a warning")
other = rot(3) @ cell_to_Ar(50.0, 70.0, 90.0, 90, 100.0, 90)
n, W, txt, err = run([{"image": "x.cxi", "event": 0, "M": other}], "mPb", ref=P21)
check("written in the setting _canonical_axes gives it (the old behavior)",
      np.allclose(W[0][0], _canonical_axes(other, centering="P"), atol=1e-3))
check("and counted on stderr", "no signed permutation of the reference" in err, err)

print("\nno reference cell: every frame of a run in ONE setting")
# Each frame: the cell with noise (0.3% on edges, 0.15 degrees on angles), at its own orientation, handed in with
# random column order and signs, left-handed included, as the blind search's reduced cells come.
SIGNED = [np.eye(3)[:, list(p)] * np.array(s) for p in itertools.permutations(range(3))
          for s in itertools.product((1.0, -1.0), repeat=3)]


def noisy_run(cell, n, seed):
    g = np.random.default_rng(seed)
    truth, res = [], []
    for i in range(n):
        C = cell_to_Ar(*(np.array(cell[:3]) * (1 + 0.003 * g.normal(size=3))),
                       *(np.array(cell[3:]) + 0.15 * g.normal(size=3)))
        T = rot(1000 * seed + i) @ C
        truth.append(T)
        res.append({"image": "x.cxi", "event": i, "M": T @ SIGNED[g.integers(len(SIGNED))]})
    return truth, res


def relabelings(truth, W):
    """The signed permutation taking each frame's true basis to the written one."""
    return [np.rint(np.linalg.solve(T, A)) for T, (A, _) in zip(truth, W)]


R194 = (28.00, 62.50, 60.90, 90.0, 90.8, 90.0)       # beta 0.8 from 90, the two long edges 2.6% apart
truth, res = noisy_run(R194, 60, seed=5)
n, W, _, err = run(res, "mPb")
X = relabelings(truth, W)
TWO_B = np.diag([-1.0, 1.0, -1.0])                   # the 2-fold about b: the one proper operator of 2/m
check("mPb, r194-like cell: every frame is the true basis up to the 2-fold about b (none at 180 degrees about a)",
      n == 60 and all(np.allclose(x, np.eye(3)) or np.allclose(x, TWO_B) for x in X),
      sum(not (np.allclose(x, np.eye(3)) or np.allclose(x, TWO_B)) for x in X))
P = np.array([params(A) for A, _ in W])
check("mPb: the code's setting, a 28 / b 62.5 (unique) / c 60.9, beta obtuse on every frame, right-handed",
      np.allclose(np.median(P[:, :3], 0), R194[:3], atol=0.3) and (P[:, 4] > 90).all()
      and all(np.linalg.det(A) > 0 for A, _ in W), np.round(np.median(P, 0), 2))
check("mPb: stderr names the run's cell", "the run's setting, cell 28" in err, err)
n, W, _, err = run(res, "mPc")
P = np.array([params(A) for A, _ in W])
check("mPc: the unique axis 62.5 in c and the obtuse angle between a and b, on every frame",
      np.allclose(np.median(P[:, :3], 0), [28.0, 60.9, 62.5], atol=0.3) and (P[:, 5] > 90).all(),
      np.round(np.median(P, 0), 2))
# Before: the length-only rule frame by frame. Same frames, how many came out in the other setting.
# Its setting is (60.9, 62.5, 28): the angle between a and c is beta on some frames and 180 - beta on others.
obtuse = sum(params(_canonical_axes(r["M"]))[4] > 90 for r in res)
check("(the length-only rule on the same frames splits them between two settings)", 0 < obtuse < len(res),
      obtuse)

TRI = (30.0, 40.0, 50.0, 80.0, 95.0, 105.0)
truth, res = noisy_run(TRI, 30, seed=6)
n, W, _, err = run(res, "aP")
X = relabelings(truth, W)
check("aP, triclinic cell: one relabeling for every frame (-1 has no proper operator but the identity)",
      n == 30 and all(np.allclose(x, X[0]) for x in X) and not err, err)

HRC = (50.0, 50.0, 50.0, 75.0, 75.0, 75.0)
bad_runs = []
for sd in range(7, 13):                              # several runs: some seeds come with mixed column signs
    truth, res = noisy_run(HRC, 40, seed=sd)
    n, W, _, err = run(res, "hR")
    ang = np.array([params(A)[3:] for A, _ in W])
    if not (n == 40 and (np.ptp(ang, axis=1) < 1.5).all() and ((ang < 90).all() or (ang > 90).all())):
        bad_runs.append(sd)
check("hR without a reference: every frame's three angles equal, one sign convention for the run "
      "(a mixed-sign seed must not set 75/105/105 for all)", not bad_runs, bad_runs)

print("\na frame with no usable cell is skipped; one singular basis used to raise and lose the whole file")
flat = LYSO.copy(); flat[:, 2] = 0.0                                    # a zero-length axis: det M = 0
BAD = [{"image": "x.cxi", "event": 10, "M": flat},
       {"image": "x.cxi", "event": 11, "M": np.full((3, 3), np.nan)},
       {"image": "x.cxi", "event": 12, "M": 0.5 * np.eye(3)}]           # finite, but 0.125 A^3
for code, ref, cell in (("tPc", None, LYSO), ("mPb", P21, P21), ("aP", None, LYSO)):
    good = [{"image": "x.cxi", "event": i, "M": rot(40 + i) @ cell} for i in range(3)]
    try:
        n, W, txt, err = run(good[:1] + BAD + good[1:], code, ref=ref)
    except np.linalg.LinAlgError as e:
        check(f"{code}: no exception", False, e)
        continue
    _, _, txt_good, _ = run(good, code, ref=ref)
    check(f"{code}: the 3 good frames are written, byte-identical to a file without the bad ones",
          n == 3 and txt == txt_good, n)
    check(f"{code}: the 3 skipped frames are counted on stderr", "3 frame(s) have no usable cell" in err, err)
with tempfile.TemporaryDirectory() as d:
    n_stream = write_stream(good[:1] + BAD + good[1:], os.path.join(d, "o.stream"))
check("the stream writer holds the same 3 frames", n_stream == 3, n_stream)
check("has_cell: None, singular, non-finite and sub-1 A^3 bases are not cells; a real one is",
      [has_cell(M) for M in (None, flat, np.full((3, 3), np.nan), 0.5 * np.eye(3), LYSO)]
      == [False, False, False, False, True])

note = written_note({"n_idx": 300, "n_written": 200})
check("the run summary names a stream holding 200 of 300 indexed",
      note is not None and "written to stream  : 200 crystals, not 300" in note, note)
check("and adds nothing when the counts agree, or with no count from the writer",
      written_note({"n_idx": 300, "n_written": 300}) is None and written_note({"n_idx": 300}) is None)

print("\nthe old name")
check("write_fromfile is write_solution_file", write_fromfile is write_solution_file)

print(f"\n{'FAILURES: ' + ', '.join(FAILS) if FAILS else 'ALL PASS'}")
sys.exit(1 if FAILS else 0)
