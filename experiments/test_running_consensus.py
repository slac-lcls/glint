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


def test_pool_switch_keys_scale_tests_on_pool_size():
    """pool_switch: min_frac/min_lead are SKIPPED below the switch (gap regime) and BIND above it.

    The failure this encodes: a wrong cell recurring at a ~0.4% rate cleared an absolute
    min_support=3 once the pool was large (23/6294 on mfxx49820 r0016) -- the random-agreement
    floor grows with pool size, so only a SHARE/RATIO test can hold at large n_pool, while at
    small n_pool the same tests add lock latency for nothing (measured on cxidb-120: always-on
    frac/lead stretched worst-case lock 18 -> 42 frames; switch >= 72 is bit-identical to gap-only).
    """
    from glint.running_consensus import consensus_accept
    # Below the switch: a small-pool lock that frac would block (0.5*18 = 9 > 3) goes through on
    # support+gap alone.
    assert consensus_accept(3, 0, 18, min_frac=0.5, min_lead=1.5, min_gap=2, pool_switch=72)
    # Above the switch: the SAME support in a big pool is a floor artifact -> frac refuses it.
    assert not consensus_accept(23, 8, 6294, min_frac=0.02, min_lead=1.5, min_gap=2, pool_switch=72)
    # pool_switch=0 (default) applies the scale tests at EVERY pool size -- existing callers unchanged.
    assert not consensus_accept(3, 0, 18, min_frac=0.5, pool_switch=0)

    # RATIO-ONLY, because everything above is carried by min_frac alone: below the switch the
    # runner-up is 0 and min_lead is inert by construction (`runner > 0` guards it), and above it
    # 23 >= 1.5 * 8 passes the ratio and only the share refuses. A regression that stopped keying
    # min_lead on the pool would survive all of it. Here the share is trivially satisfied
    # (0.02 * 100 = 2 <= 10) so the ratio decides: 10 < 1.5 * 8 = 12.
    assert consensus_accept(10, 8, 71, min_frac=0.02, min_lead=1.5, pool_switch=72)      # skipped
    assert not consensus_accept(10, 8, 72, min_frac=0.02, min_lead=1.5, pool_switch=72)  # binds
    # ...and the boundary is where it says it is: the test is `n_pool >= pool_switch`, so equality
    # BINDS. 71 vs 72 is the whole difference, and an off-by-one there would be invisible without it.
    assert consensus_accept(10, 8, 71, min_lead=1.5, pool_switch=72)
    assert not consensus_accept(10, 8, 72, min_lead=1.5, pool_switch=72)


def test_pool_switch_running_lock_unchanged_small_refuses_large():
    """RunningConsensus with the gated lock: clean small-pool locking is bit-identical to gap-only,
    and a leader that is a tiny share of a large pool is refused instead of locked."""
    # Clean stream: true cell every frame + fresh scatter. Locks identically with and without the gate.
    # Both arms are recorded and COMPARED, not each checked against "some correct lock" in
    # isolation: the claim is that the gate changes NOTHING here, and an isolated per-arm assertion
    # would still pass if the gated arm locked on a different frame, with different support, or on
    # a different member of the same lattice group.
    out = {}
    for tag, kw in (("gap-only", dict()),
                    ("pool-keyed", dict(min_frac=0.02, min_lead=1.5, pool_switch=72))):
        rc = RunningConsensus(min_support=3, gap=2, adaptive=True, **kw)
        out[tag] = None
        for f in range(12):
            rc.add_frame([_rot(0.13 * f) @ M_TRUE, _SCATTER[f % len(_SCATTER)]])
            M, w0, w1 = rc.verdict()
            if M is not None:
                out[tag] = (f + 1, w0, w1, M, rc.npool)
                break
        assert out[tag] is not None and _rp_eq(M, M_TRUE), (tag, out[tag])
        assert rc.npool < 72, rc.npool                          # clean locks live BELOW the switch
    a, b = out["gap-only"], out["pool-keyed"]
    assert a[:3] == b[:3], (a[:3], b[:3])                       # same frame, support, runner-up
    assert np.array_equal(a[3], b[3]), (a[3], b[3])             # and the SAME cell, not merely one
    assert a[4] == b[4], (a[4], b[4])                           # of the same lattice; same pool too
    # Floor regime: the "leader" recurs 1/25 frames while every frame adds 3 scatter cells that
    # are unique BY CONSTRUCTION (a 6%-spaced 12x12x12 length ladder; rtol=0.05 cannot group two
    # distinct rungs -- a random 150-300 draw birthday-collides into its own clusters and locks the
    # test on the wrong mechanism). Support crosses min_support + adapted gap only once the pool is
    # far past the switch; gap-only locks that floor artifact, the pool-keyed gate refuses it
    # (share ~1.3% < 2% at every pool size).
    def _unique_scatter(i):
        a = 150.0 * 1.06 ** (i % 12)
        b = 400.0 * 1.06 ** ((i // 12) % 12)
        c = 1100.0 * 1.06 ** ((i // 144) % 12)
        return np.diag([1.0 / a, 1.0 / b, 1.0 / c])
    def floor_frames(n):
        out = []
        for f in range(n):
            fr = [_unique_scatter(3 * f + j) for j in range(3)]
            if f % 25 == 0:
                fr[0] = _rot(0.1 * f) @ M_ALIAS
            out.append(fr)
        return out
    frames = floor_frames(200)
    locks = {}
    for name, kw in (("gap-only", dict()),
                     ("pool-keyed", dict(min_frac=0.02, min_lead=1.5, pool_switch=72))):
        rc = RunningConsensus(min_support=3, gap=2, adaptive=True, **kw)
        locks[name] = None
        for f, fr in enumerate(frames):
            rc.add_frame(fr)
            M, w0, w1 = rc.verdict()
            if M is not None:
                locks[name] = (f + 1, rc.npool)
                break
    assert locks["gap-only"] is not None, locks                 # the floor artifact DOES lock today
    assert locks["gap-only"][1] > 72, locks                     # ... and only in the large-pool regime
    assert locks["pool-keyed"] is None, locks                   # the pool-keyed gate refuses it


# ---------------------------------------------------------------------------------------------
# Non-transitivity of tolerance matching, and the merge-on-multi-match repair (glint: streaming
# path). Purely synthetic: the failure needs only three cells, so it reproduces with no data file.
# ---------------------------------------------------------------------------------------------

def _diag(a, b, c):
    return np.diag([1.0 / a, 1.0 / b, 1.0 / c])


def test_merge_repairs_split_vote():
    """A true lattice whose votes FRAGMENT must still win over a tighter-clustered spurious one.

    A and B are both inside rtol of the truth but NOT of each other, so first-match-wins keeps them
    apart and a spurious group with fewer total votes than A+B outranks each half. This is the shape
    of the real failure: on a refined-geometry run the true cell's 161+150 votes lost to a spurious
    217. The merge witness is a cell lying between A and B, which is exactly what a real refine
    produces and what proves the two groups are one lattice.
    """
    TRUE = _diag(79.0, 79.0, 38.0)
    A = _diag(79.0 * 0.972, 79.0 * 0.972, 38.0 * 0.972)      # -2.8% : within 5% of TRUE
    B = _diag(79.0 * 1.028, 79.0 * 1.028, 38.0 * 1.028)      # +2.8% : within 5% of TRUE...
    # ...but A and B are 5.8% apart, so they do NOT match each other -- tolerance is not transitive.
    rc = RunningConsensus(min_support=3, gap=2, merge=False)
    la, ca = reduced_params(A); lb, cb = reduced_params(B)
    lt, ct = reduced_params(TRUE)
    assert rc._match(la, ca, abs(np.linalg.det(A)), [TRUE, lt, ct, abs(np.linalg.det(TRUE)), 1]), "A should match TRUE"
    assert not rc._match(la, ca, abs(np.linalg.det(A)), [B, lb, cb, abs(np.linalg.det(B)), 1]), "A must NOT match B"

    SPUR = _diag(61.0, 67.0, 73.0)                            # a distinct, tightly-clustered alias
    frames = ([[A]] * 9) + ([[B]] * 9) + ([[SPUR]] * 12)      # A+B = 18 votes, SPUR = 12
    witness = [[_diag(79.0, 79.0, 38.0)]]                     # one on-truth cell: matches A and B both

    def winner(merge):
        rc = RunningConsensus(min_support=3, gap=2, adaptive=False, merge=merge)
        for f in frames + witness:
            rc.add_frame(f)
        top = max(rc.groups, key=lambda g: g[4])
        return top, len(rc.groups)

    top_off, n_off = winner(False)
    top_on, n_on = winner(True)
    assert not same_lattice(top_off[0], TRUE), "premise: without merge the split vote must LOSE"
    assert same_lattice(top_on[0], TRUE), "with merge the true lattice must win"
    assert top_on[4] >= 18, top_on[4]                        # the two halves, recombined
    assert 1 < n_on < n_off, (n_on, n_off)                   # folds what it must, not everything


def test_merge_false_is_bit_identical_to_pre_fix():
    """merge=False must reproduce first-match-wins EXACTLY, so the fix is auditable against it."""
    rng = np.random.default_rng(7)
    cells = [_diag(*(79.0 + rng.normal(0, 3), 79.0 + rng.normal(0, 3), 38.0 + rng.normal(0, 2)))
             for _ in range(60)]
    rc = RunningConsensus(min_support=3, gap=2, adaptive=False, merge=False)
    for c in cells:
        rc.add_frame([c])
    ref_w = sorted(g[4] for g in rc.groups)
    # replicate the pre-fix loop by hand
    groups = []
    for M in cells:
        l, c = reduced_params(np.asarray(M, float)); d = abs(np.linalg.det(np.asarray(M, float)))
        for g in groups:
            if rc._match(l, c, d, g):
                g[4] += 1; break
        else:
            groups.append([M, l, c, d, 1])
    assert ref_w == sorted(g[4] for g in groups), (ref_w, sorted(g[4] for g in groups))


def test_merge_is_on_by_default():
    """The DEFAULT must be the repaired grouping.

    Both other merge tests pass ``merge=`` explicitly, so neither notices if the default flips back
    to first-match-wins -- and the default is the entire point of the change: a caller who passes
    nothing must not silently get the partition that locks on the wrong lattice.
    """
    assert RunningConsensus().merge is True, "merge must default to True"
    assert RunningConsensus(merge=False).merge is False, "merge=False must remain available"

    TRUE = _diag(79.0, 79.0, 38.0)
    A = _diag(79.0 * 0.972, 79.0 * 0.972, 38.0 * 0.972)
    B = _diag(79.0 * 1.028, 79.0 * 1.028, 38.0 * 1.028)
    SPUR = _diag(61.0, 67.0, 73.0)
    frames = ([[A]] * 9) + ([[B]] * 9) + ([[SPUR]] * 12) + [[_diag(79.0, 79.0, 38.0)]]
    rc = RunningConsensus(min_support=3, gap=2, adaptive=False)      # NO merge= : the default
    for f in frames:
        rc.add_frame(f)
    top = max(rc.groups, key=lambda g: g[4])
    assert same_lattice(top[0], TRUE), "default grouping must recover the true lattice"


if __name__ == "__main__":
    tests = (test_batch_equivalence, test_early_stop_locks_clean,
             test_refuses_ambiguous_race, test_adaptive_gap_raises_when_slow,
             test_pool_switch_keys_scale_tests_on_pool_size,
             test_pool_switch_running_lock_unchanged_small_refuses_large,
             test_merge_repairs_split_vote, test_merge_false_is_bit_identical_to_pre_fix,
             test_merge_is_on_by_default)
    ok = 0
    for t in tests:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(tests)} passed")
    # The repo's script contract, stated in .github/workflows/ci.yml: print, then exit non-zero on
    # failure. Without this the file reports "2/6 passed" and still exits 0, so a CI step running it
    # goes green on a red suite -- which matters now that CI runs this file.
    raise SystemExit(0 if ok == len(tests) else 1)
