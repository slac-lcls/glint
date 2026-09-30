"""CPU tests for glint.predict.write_solution_file: the CrystFEL --indexing=file solution file puts the axes in
the setting its lattice code names.

WHY THIS EXISTS. For monoclinic codes the writer used to pass the known-cell engine's column order straight
through. On LUTE's cxil1015922 r136 test (C2, 111.94/172.23/41.23, beta 106.2 on b) every line came out as
41/111/172 with the 106-degree angle between a and b, labelled ``mCb`` (unique axis b), and CrystFEL's cell
check rejected all 27 frames. With a reference cell the writer now picks the proper signed permutation of the
columns closest to it; tetragonal / orthorhombic / hexagonal codes keep their standard setting, byte for byte.

  PYTHONPATH=. python experiments/test_solution_file.py      # exit 0 = all pass
"""
import io
import os
import sys
import tempfile
from contextlib import redirect_stderr

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint.lattice import cell_to_Ar                                    # noqa: E402
from glint.predict import write_fromfile, write_solution_file           # noqa: E402

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

print("\nwithout the reference cell: the old behaviour, and a warning for monoclinic codes")
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
_, _, txt_old, _ = run([{"image": "x.cxi", "event": 0, "M": other}], "mPb")
check("written exactly as without a reference (the old behaviour)", txt == txt_old)
check("and counted on stderr", "no signed permutation of the reference" in err, err)

print("\nthe old name")
check("write_fromfile is write_solution_file", write_fromfile is write_solution_file)

print(f"\n{'FAILURES: ' + ', '.join(FAILS) if FAILS else 'ALL PASS'}")
sys.exit(1 if FAILS else 0)
