"""CPU tests for glint.running_consensus (incremental cross-frame consensus + sequential early-stop).
No GPU, no driver. Run: `python experiments/test_running_consensus.py` or `pytest`.

Covers the module's three load-bearing claims: (1) fed every frame, verdict(gap=0) reproduces the batch
consensus_cell max-weight group exactly (same lattice + support); (2) on clean data the stop rule LOCKS in
a few frames on the true cell; (3) the gap knob is a specificity control -- a coincident ~1:1 alias race
refuses to lock, and a slowly-accumulating leader (the mosaic signature) raises the adaptive gap.
"""
import numpy as np
from glint.running_consensus import RunningConsensus
from glint.lattice import reduced_params
from glint.multishot import consensus_cell, same_lattice


def _rot(theta, axis=2):                                        # a rotation -- reduced_params is invariant,
    c, s = np.cos(theta), np.sin(theta)                         # so a rotated basis stays in the same group
    R = np.eye(3)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    R[i, i] = R[j, j] = c; R[i, j] = -s; R[j, i] = s
    return R

M_TRUE = np.diag([1 / 79.0, 1 / 79.0, 1 / 38.0])               # lyso reciprocal basis (one orientation)
M_ALIAS = np.diag([1 / 60.0, 1 / 72.0, 1 / 95.0])              # a DISTINCT persistent lattice (different reduced cell)
_SCATTER = [np.diag([1 / L, 1 / (L + 13), 1 / (L + 27)]) for L in (150.0, 175.0, 205.0, 235.0, 265.0, 295.0)]


def _rp_eq(A, B, rtol=1e-6):
    la, ca = reduced_params(A); lb, cb = reduced_params(B)
    return np.allclose(la, lb, rtol=rtol) and np.allclose(ca, cb, atol=1e-6)


def test_batch_equivalence():
    """Fed all frames, verdict(gap=0) == batch consensus_cell (same lattice fingerprint + support)."""
    rng = np.random.default_rng(3)
    frames = []                                                 # each frame: a few candidate bases
    for f in range(6):
        fr = [_rot(0.13 * f) @ M_TRUE]                          # the true cell recurs, varied orientation
        if f % 2 == 0:
            fr.append(_rot(0.2 * f, axis=0) @ M_ALIAS)          # alias recurs ~half the frames
        q = rng.normal(size=3); q /= np.linalg.norm(q)          # one fresh scatter cell per frame
        fr.append(np.diag(1.0 / rng.uniform(150, 300, 3)))
        frames.append(fr)
    pooled = [M for fr in frames for M in fr]                   # SAME order -> greedy grouping is identical
    # medoid=False so BOTH sides return the group's founder. consensus_cell now reports the medoid
    # of the winning group by default while RunningConsensus still reports the founder, so the
    # default settings compare two deliberately different representatives. That is not what this
    # test is about -- it claims the two GROUPINGS agree, so it pins the representative choice and
    # compares like with like. Without this the assertion below passes only because every member of
    # this test's group shares a reduced_params fingerprint (they are M_TRUE rotated, and the
    # fingerprint is rotation-invariant), so every medoid distance is 0 and argmin returns the
    # founder anyway. Add any scatter to the group and it would break for the wrong reason.
    rep_b, sup_b = consensus_cell(pooled, min_support=3, medoid=False)

    rc = RunningConsensus(min_support=3, adaptive=False)
    for fr in frames:
        rc.add_frame(fr)
    rep_r, sup_r, _ = rc.verdict(gap=0)
    assert rep_r is not None, "running verdict(gap=0) failed to reproduce a locked batch group"
    assert sup_r == sup_b, (sup_r, sup_b)                       # identical support count
    assert _rp_eq(rep_r, rep_b), (reduced_params(rep_r), reduced_params(rep_b))
    assert _rp_eq(rep_r, M_TRUE)                                # and it is the true cell

    # and the medoid, which IS the default, must still be the same lattice -- a different member of
    # the winning group, never a different group.
    rep_med, sup_med = consensus_cell(pooled, min_support=3)
    assert sup_med == sup_b, (sup_med, sup_b)                   # representative choice cannot move support
    assert same_lattice(rep_med, M_TRUE)


def test_early_stop_locks_clean():
    """Clean stream (true cell every frame + unique scatter): locks fast, on the true cell."""
    rc = RunningConsensus(min_support=3, gap=2, adaptive=True)
    locked_at = None
    for f in range(6):
        rc.add_frame([_rot(0.17 * f) @ M_TRUE, _SCATTER[f]])   # true recurs; each scatter appears once
        Mc, sup, lead = rc.verdict()
        if Mc is not None:
            locked_at = f + 1; break
    assert locked_at is not None and locked_at <= 4, locked_at  # min_support 3 + gap 2 => by frame 4
    assert _rp_eq(Mc, M_TRUE), reduced_params(Mc)


def test_refuses_ambiguous_race():
    """A coincident ~1:1 leader/runner-up race must NOT lock at gap>=2 (specificity)."""
    rc = RunningConsensus(min_support=3, gap=2, adaptive=True)
    for f in range(8):
        rc.add_frame([_rot(0.11 * f) @ M_TRUE, _rot(0.09 * f, axis=1) @ M_ALIAS])  # both recur every frame
        assert rc.verdict()[0] is None, f"false-locked on a tie at frame {f + 1}"
    _, w0, w1 = rc.verdict()
    assert w0 >= 3 and abs(w0 - w1) < 2, (w0, w1)               # both well-supported, lead never opens


def test_adaptive_gap_raises_when_slow():
    """A leader that accumulates slowly (rate<0.30) raises the adaptive gap above base (mosaic signature)."""
    rc = RunningConsensus(min_support=3, gap=2, adaptive=True)
    for f in range(10):                                         # true cell only in 2 of 10 frames -> rate 0.2
        fr = [_rot(0.1 * f) @ M_TRUE] if f in (3, 7) else []
        fr.append(_SCATTER[f % len(_SCATTER)])                 # rotate through fixed unique scatters
        rc.add_frame(fr)
    _, w0, _ = rc.verdict()
    assert w0 == 2, w0                                          # leader is the true cell, support 2
    assert rc.gap == rc.base_gap + 1, (rc.gap, rc.base_gap)     # 0.12 <= rate < 0.30  ->  +1


if __name__ == "__main__":
    tests = (test_batch_equivalence, test_early_stop_locks_clean,
             test_refuses_ambiguous_race, test_adaptive_gap_raises_when_slow)
    ok = 0
    for t in tests:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(tests)} passed")
