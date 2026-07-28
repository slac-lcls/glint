"""CPU tests for the lock-time alias gate (glint.alias_gate) -- pure numpy, no GPU/torch.

Dual mode: `pytest experiments/test_alias_gate.py`, or `python experiments/test_alias_gate.py` for a
PASS/FAIL loop. Cell convention (== stream_driver._inliers): M columns are the real cell vectors, so
hkl = q @ M and q = hkl @ inv(M).
"""
import numpy as np

from glint.alias_gate import hnf_matrices, derivative_lattices, tightness, AliasGate
from glint.multishot import same_lattice

SEED = 20260727
M_TRUE = np.diag([40.0, 55.0, 70.0])          # orthorhombic P, distinct axes (no accidental symmetry)


def _observed_q(M, rng, frac=0.7, hmax=6):
    """Reciprocal peaks on the lattice of M: a random `frac` of the integer nodes inside |q| <= 1/dmin,
    with a little positional noise. q = hkl @ inv(M)."""
    Minv = np.linalg.inv(M)
    H = np.array([(h, k, l)
                  for h in range(-hmax, hmax + 1)
                  for k in range(-hmax, hmax + 1)
                  for l in range(-hmax, hmax + 1)
                  if (h or k or l)], float)
    q = H @ Minv
    qn = np.sqrt((q * q).sum(1))
    qmax = np.quantile(qn, 0.35)                # a modest resolution shell so the box isn't empty
    q = q[qn <= qmax]
    keep = rng.random(len(q)) < frac
    q = q[keep]
    q = q + rng.normal(0.0, 0.0008, q.shape)    # sub-tolerance jitter
    return q


# --------------------------------------------------------------------------------- HNF / enumeration
def test_hnf_counts_and_dets():
    for idx, n in [(1, 1), (2, 7), (3, 13)]:
        H = hnf_matrices(idx)
        assert len(H) == n, (idx, len(H))
        for h in H:
            assert abs(round(np.linalg.det(h)) - idx) == 0, (idx, np.linalg.det(h))


def test_derivatives_include_true():
    """The index-2 super-cell S = M@H must have the true cell M back among ITS derivatives (S@inv(H))."""
    H = hnf_matrices(2)[0]
    S = M_TRUE @ H
    cands = derivative_lattices(S, max_index=2)
    assert any(same_lattice(M_TRUE, c) for c in cands), "true cell not recovered as a derivative of its super-cell"


# ------------------------------------------------------------------------------------ discrimination
def test_true_cell_is_strictly_tightest():
    rng = np.random.default_rng(SEED)
    Q = _observed_q(M_TRUE, rng)
    cands = derivative_lattices(M_TRUE, max_index=2)
    scores = np.array([tightness(M, Q)[0] for M in cands])
    assert int(np.argmax(scores)) == 0, ("true cell not the tightest", scores[:5])
    # and it wins by a real margin, not a numerical tie
    runner = np.sort(scores)[-2]
    assert scores[0] > 1.15 * runner, (scores[0], runner)


def test_supercell_loses_on_occupancy_not_coverage():
    """The diagnostic property: a super-cell keeps coverage ~1 but loses occupancy (systematic absences)."""
    rng = np.random.default_rng(SEED + 1)
    Q = _observed_q(M_TRUE, rng)
    H = hnf_matrices(2)[-1]                      # a det-2 form
    S = M_TRUE @ H
    _, cov_t, occ_t = tightness(M_TRUE, Q)
    _, cov_s, occ_s = tightness(S, Q)
    assert cov_s > 0.9 and cov_t > 0.9, (cov_t, cov_s)         # both index the peaks
    assert occ_s < 0.75 * occ_t, (occ_t, occ_s)               # super-cell is half-empty


# ------------------------------------------------------------------------------------ the gate calls
def test_confirm_true_leader():
    rng = np.random.default_rng(SEED + 2)
    Q = _observed_q(M_TRUE, rng)
    out = AliasGate().confirm(M_TRUE, Q)
    assert out is not None and same_lattice(out, M_TRUE), "gate rejected the true leader"


def test_reject_supercell_leader():
    """If consensus locked a super-cell alias, the default gate REFUSES (None); adopt-mode swaps in truth."""
    rng = np.random.default_rng(SEED + 3)
    Q = _observed_q(M_TRUE, rng)
    S = M_TRUE @ hnf_matrices(2)[-1]
    assert AliasGate(adopt=False).confirm(S, Q) is None, "gate confirmed a super-cell alias"
    adopted = AliasGate(adopt=True).confirm(S, Q)
    assert adopted is not None and same_lattice(adopted, M_TRUE), "adopt-mode did not recover the true cell"


def test_empty_Q_does_not_block():
    assert same_lattice(AliasGate().confirm(M_TRUE, np.zeros((0, 3))), M_TRUE)


def test_singular_cell_does_not_crash():
    """The blind indexer can hand the gate a (near-)singular candidate on off-regime frames -- it must be
    refused, never crash on inv() (regression for the GPU-run LinAlgError)."""
    rng = np.random.default_rng(SEED + 9)
    Q = _observed_q(M_TRUE, rng)
    Msing = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0]])   # rows 0,1 collinear -> det 0
    assert tightness(Msing, Q)[0] == 0.0                        # scored zero, no exception
    derivs = derivative_lattices(Msing, 2)                      # degenerate leader -> just [M], no inv() blowup
    assert len(derivs) == 1
    out = AliasGate().confirm(Msing, Q)                         # must return without raising
    assert out is not None


def test_pseudocentered_data_prefers_smaller_cell():
    """Honest limit: if the data genuinely populates ONLY a sublattice (pseudo-centering), the smaller
    cell is the correct, tighter explanation and adopt-mode returns it -- the gate follows the data, and a
    single frame cannot distinguish true absence from missing measurement. Documented, not a failure."""
    rng = np.random.default_rng(SEED + 4)
    Msub = M_TRUE @ np.linalg.inv(hnf_matrices(2)[-1])         # a smaller (sub) cell of M_TRUE
    Q = _observed_q(Msub, rng)                                # peaks live only on the sub-lattice
    out = AliasGate(adopt=True).confirm(M_TRUE, Q)             # leader is the (too-large) M_TRUE
    assert out is not None and not same_lattice(out, M_TRUE), "should have moved off the over-large cell"
    assert same_lattice(out, Msub), "should have adopted the data's actual (smaller) lattice"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
    raise SystemExit(1 if fails else 0)
