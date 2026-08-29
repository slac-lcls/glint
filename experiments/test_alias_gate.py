"""CPU tests for the lock-time alias gate (glint.alias_gate) -- pure numpy, no GPU/torch.

Dual mode: `pytest experiments/test_alias_gate.py`, or `python experiments/test_alias_gate.py` for a
PASS/FAIL loop. Cell convention (== stream_driver._inliers): M columns are the real cell vectors, so
hkl = q @ M and q = hkl @ inv(M).
"""
import numpy as np

from glint.alias_gate import (hnf_matrices, derivative_lattices, tightness, AliasGate,
                              null_coverage, coverage_floor)
from glint.lattice import random_rotation
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


def _same_sublattice(A, B):
    """A and B span the same sublattice iff A^-1 B is integral with |det| == 1 (a change of basis)."""
    T = np.linalg.solve(A, B)
    return np.allclose(T, np.round(T), atol=1e-9) and abs(abs(np.linalg.det(T)) - 1.0) < 1e-9


def test_hnf_matrices_are_distinct_sublattices():
    """The enumeration must be CANONICAL, not merely the right size.

    test_hnf_counts_and_dets above passes for a broken enumeration, and did: reducing each
    off-diagonal modulo its COLUMN's diagonal instead of its ROW's gives 7/13/35 matrices spanning
    only 3/3/9 distinct sublattices. The counts agree because sum(d*f^2) == sum(a^2*d) over a triple
    set closed under permutation, so the cardinality is invariant under exactly the error it is
    supposed to catch. Test the equivalence the canonical form is defined by, not its cardinality.

    Counts are the number of index-n sublattices of Z^3 (OEIS A001001): 1, 7, 13, 35, 31, 91.
    """
    for idx, n in [(1, 1), (2, 7), (3, 13), (4, 35), (5, 31), (6, 91)]:
        H = hnf_matrices(idx)
        assert len(H) == n, (idx, len(H))
        for i, A in enumerate(H):
            for B in H[i + 1:]:
                assert not _same_sublattice(A, B), (
                    f"index {idx}: two enumerated forms span the SAME sublattice:\n{A}\n{B}")


def test_hnf_family_contains_off_diagonal_forms():
    """Regression guard on the specific loss: the broken enumeration kept only axial doublings.

    [[2,1,0],[0,1,0],[0,0,1]] is an ordinary index-2 sublattice with a live off-diagonal. It was
    absent, so every 'the gate refused the alias' verdict was on an axial doubling only -- the
    face-diagonal cells the module docstring names as the target were never in the family.
    """
    target = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert any(_same_sublattice(H, target) for H in hnf_matrices(2))


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


# ------------------------------------------------------- orientation coherence (the pooled-voter bug)
def _still_q(M, rng, n=120, qmax=0.30):
    """A sparse still: n peaks scattered over the MANY nodes inside |q| <= qmax, the way a real frame
    samples a thin slice. Not interchangeable with `_observed_q`, which puts a fixed fraction of EVERY
    node on every frame: pool a few of those and the peaks outnumber the nodes, every occupancy pins at
    1, and the pooled-input failure this file tests for cannot be reproduced (nodes here ~1.7e4 vs 120
    peaks a frame, which is the real ratio's order)."""
    Minv = np.linalg.inv(M)
    rlen = np.sqrt((Minv ** 2).sum(1))
    hb = np.ceil(qmax / np.maximum(rlen, 1e-12)).astype(int)
    out = []
    while sum(len(o) for o in out) < n:
        H = np.column_stack([rng.integers(-hb[i], hb[i] + 1, 4 * n) for i in range(3)]).astype(float)
        q = H[np.abs(H).sum(1) > 0] @ Minv
        out.append(q[np.sqrt((q * q).sum(1)) <= qmax])
    q = np.vstack(out)[:n]
    return q + rng.normal(0.0, 0.0008, q.shape)


def _frames(M, rng, k=24, n=120):
    """k frames of the SAME lattice at k different orientations, as (q, M_i) pairs -- the input a
    cross-frame consensus lock actually has. M_i = R_i @ M is the lattice as oriented on frame i."""
    out = []
    for _ in range(k):
        Mi = random_rotation(rng) @ M
        out.append((_still_q(Mi, rng, n=n), Mi))
    return out


def test_null_coverage_is_the_measured_chance_floor():
    """The analytic (2*tol)^3: a cell scores this on peaks it bears no relation to, whatever its shape.
    This is the number the pooled input collapsed onto (0.027 predicted, 0.0276-0.0302 measured across
    the whole index-2 family on 37,531 cxidb-62 peaks)."""
    rng = np.random.default_rng(SEED + 11)
    Q = rng.normal(0.0, 0.02, (20000, 3))                      # peaks with no lattice at all
    for M in (M_TRUE, M_TRUE @ hnf_matrices(2)[-1], M_TRUE @ np.linalg.inv(hnf_matrices(2)[-1])):
        cov = tightness(M, Q)[1]
        assert abs(cov - null_coverage(0.15)) < 0.006, (cov, null_coverage(0.15))


def test_coverage_floor_is_where_a_halfvolume_alias_breaks_even():
    """The floor is derived, not tuned: at the break-even coherent fraction (safety=1) an ideal index-2
    alias reaches exactly the margin, so anything below is decided by volume alone. The shipped default
    carries a safety factor over that, for the upward bias of maximising over the derivative family."""
    c0, m, k = null_coverage(0.15), 1.10, 2.0
    cov = coverage_floor(0.15, m, int(k), safety=1.0)
    f = (cov - c0) / (1.0 - c0)                                # invert cov = f + (1-f)c0
    ratio = k * ((f / k + (1.0 - f) * c0) / (f + (1.0 - f) * c0)) ** 2      # score ~ N*cov^2/V
    assert abs(ratio - m) < 1e-6, (ratio, m)
    assert coverage_floor(0.15, 2.5, 2) == 0.0                 # margin >= index: nothing to defend against
    assert abs(coverage_floor() - 2.0 * cov) < 1e-12           # default safety=2


def test_pooled_multiorientation_does_not_refuse():
    """THE REGRESSION. Pool many orientations into one cloud and every candidate sits at chance coverage,
    so the score is a 1/V preference for the smallest cell and the leader loses however right it is. The
    gate must recognise the input as untestable and abstain -- never refuse a lock on this evidence."""
    rng = np.random.default_rng(SEED + 5)
    fr = _frames(M_TRUE, rng, k=24)
    leader = fr[0][1]                                          # the lock is expressed in ONE orientation
    Q = np.vstack([q for q, _ in fr])
    gate = AliasGate()
    out = gate.confirm(leader, Q)
    assert out is not None and same_lattice(out, M_TRUE), "gate refused a TRUE cell on pooled voters"
    assert gate.info["verdict"] == "abstain", gate.info
    # and the reason is the documented one: everything is at chance, smallest cell wins
    scores = [tightness(M, Q)[0] for M in derivative_lattices(leader, 2)]
    assert int(np.argmax(scores)) != 0, "pooled cloud unexpectedly favoured the leader"


def test_confirm_frames_confirms_true_cell():
    """The same voters, scored per frame in each frame's own orientation, confirm the true lock."""
    rng = np.random.default_rng(SEED + 6)
    fr = _frames(M_TRUE, rng, k=12)
    gate = AliasGate()
    out = gate.confirm_frames(M_TRUE, fr)
    assert out is not None and same_lattice(out, M_TRUE), gate.info
    assert gate.info["verdict"] == "confirm", gate.info
    assert gate.info["frames_tested"] == 12, gate.info


def test_confirm_frames_still_refuses_a_supercell():
    """Protection retained: a super-cell leader is out-tightened on frame after frame (it is a property of
    the lattice, not of the shot), so the majority rule refuses it and adopt-mode recovers the true cell."""
    rng = np.random.default_rng(SEED + 7)
    H = hnf_matrices(2)[-1]
    fr = _frames(M_TRUE, rng, k=12)
    sup = [(q, M @ H) for q, M in fr]                          # same frames, super-cell leader
    assert AliasGate().confirm_frames(M_TRUE @ H, sup) is None, "confirmed a super-cell alias"
    g = AliasGate(adopt=True)
    out = g.confirm_frames(M_TRUE @ H, sup)
    assert out is not None and same_lattice(out, M_TRUE), (g.info, out is None)


def test_confirm_frames_drops_untestable_frames():
    """A frame whose peaks are not in the leader's orientation testifies about nothing and is dropped, not
    counted as a refusal -- otherwise a handful of unregistered frames could sink a good lock."""
    rng = np.random.default_rng(SEED + 8)
    fr = _frames(M_TRUE, rng, k=8)
    junk = [(rng.normal(0.0, 0.02, (300, 3)), M) for _, M in fr[:8]]   # right cell, wrong peaks
    gate = AliasGate()
    out = gate.confirm_frames(M_TRUE, fr + junk)
    assert out is not None and same_lattice(out, M_TRUE), gate.info
    assert gate.info["frames_seen"] == 16 and gate.info["frames_tested"] == 8, gate.info


def test_confirm_frames_abstains_on_too_few_testable_frames():
    """A cross-frame gate that fires on one frame's opinion is not a cross-frame gate: if the untestable
    drop leaves fewer than min_frames, abstain rather than let a single frame refuse a lock."""
    rng = np.random.default_rng(SEED + 10)
    H = hnf_matrices(2)[-1]
    fr = _frames(M_TRUE, rng, k=2)
    sup = [(q, M @ H) for q, M in fr]                          # 2 frames that WOULD refuse the super-cell
    gate = AliasGate()
    out = gate.confirm_frames(M_TRUE @ H, sup)
    assert out is not None, "refused a lock on 2 frames"
    assert gate.info["verdict"] == "abstain" and gate.info["frames_tested"] == 2, gate.info
    assert AliasGate(min_frames=1).confirm_frames(M_TRUE @ H, sup) is None, "min_frames=1 should rule"


def test_min_coverage_zero_restores_prefix_behaviour():
    """The floor is a knob, not a hard-wired policy: min_coverage=0 reproduces the pre-fix verdict, which
    is how the old refusals stay reproducible for comparison."""
    rng = np.random.default_rng(SEED + 5)
    fr = _frames(M_TRUE, rng, k=24)
    Q = np.vstack([q for q, _ in fr])
    assert AliasGate(min_coverage=0.0).confirm(fr[0][1], Q) is None, "expected the old (wrong) refusal"


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
