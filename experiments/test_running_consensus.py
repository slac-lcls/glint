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
    """Fed all frames, verdict(gap=0) == batch consensus_cell (same lattice fingerprint + support).

    merge=False is REQUIRED: the equivalence is to the LEGACY arrival-order batch grouper, which
    assigns a multi-match witness to the first group it matches and leaves the rest un-coalesced.
    The default coalesces, so it deliberately diverges -- see
    test_batch_equivalence_diverges_on_multi_match, which pins the divergence and its direction.
    """
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

    rc = RunningConsensus(min_support=3, adaptive=False, merge=False)
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
    """STREAMING: the default must lock on the true lattice where first-match-wins locks spurious.

    This drives verdict() after every add_frame -- the decide-after-every-frame behaviour the class
    exists for -- not just the pooled partition at the end. The order matters: A and B (the two
    halves of a split true lattice, each within rtol of truth but 5.8% from each other) arrive first
    but neither reaches min_support alone, then the WITNESS lands, then the spurious group
    accumulates. Without merge the halves stay apart and SPUR eventually wins the lock; with merge
    the witness coalesces them and the true lattice locks immediately, on fewer frames.
    """
    TRUE = _diag(79.0, 79.0, 38.0)
    A = _diag(79.0 * 0.972, 79.0 * 0.972, 38.0 * 0.972)      # -2.8% : within rtol of TRUE
    B = _diag(79.0 * 1.028, 79.0 * 1.028, 38.0 * 1.028)      # +2.8% : within rtol of TRUE...
    # ...but A and B are 5.8% apart, so they do NOT match each other -- tolerance is not transitive.
    rc = RunningConsensus(min_support=3, gap=2, merge=False)
    la, ca = reduced_params(A); lb, cb = reduced_params(B); lt, ct = reduced_params(TRUE)
    assert rc._match(la, ca, abs(np.linalg.det(A)),
                     [TRUE, lt, ct, abs(np.linalg.det(TRUE)), 1]), "premise: A matches TRUE"
    assert not rc._match(la, ca, abs(np.linalg.det(A)),
                         [B, lb, cb, abs(np.linalg.det(B)), 1]), "premise: A must NOT match B"

    SPUR = _diag(61.0, 67.0, 73.0)                           # a distinct, tightly-clustered alias
    WITNESS = _diag(79.0, 79.0, 38.0)                        # inside the tolerance of A and B both
    stream = [[A], [B], [WITNESS]] + [[SPUR]] * 4

    def run(merge):
        rc = RunningConsensus(min_support=3, gap=2, adaptive=False, merge=merge)
        for i, fr in enumerate(stream):
            rc.add_frame(fr)
            Mc, sup, _ = rc.verdict()                        # after EVERY frame, as the driver does
            if Mc is not None:
                return i + 1, Mc, sup
        return None, None, None

    at_off, Mc_off, _ = run(False)
    at_on, Mc_on, sup_on = run(True)
    assert Mc_off is not None and not same_lattice(Mc_off, TRUE), \
        "premise: first-match-wins must lock on the SPURIOUS lattice"
    assert Mc_on is not None and same_lattice(Mc_on, TRUE), "the default must lock on the true lattice"
    assert at_on < at_off, (at_on, at_off)                   # and sooner, not by waiting longer
    assert sup_on >= 3, sup_on

    # The merged group's representative must COVER the votes credited to it: the driver locks this
    # matrix and its alias gate only collects candidates matching it, so a founder that excludes half
    # its own voters would gate them away. The witness is inside the tolerance of every group it
    # merged, which is what selected them, so it is the representative.
    rc = RunningConsensus(min_support=3, gap=2, adaptive=False)
    for fr in ([[A]] * 9) + ([[B]] * 9) + ([[SPUR]] * 12) + [[WITNESS]]:
        rc.add_frame(fr)
    top = max(rc.groups, key=lambda g: g[4])
    assert same_lattice(top[0], TRUE), "pooled winner must be the true lattice"
    assert top[4] >= 18, top[4]                              # the two halves, recombined
    lr, cr = reduced_params(top[0]); dr = abs(np.linalg.det(top[0]))
    for nm, M in (("A", A), ("B", B)):
        lm, cm = reduced_params(M)
        assert rc._match(lm, cm, abs(np.linalg.det(M)), [top[0], lr, cr, dr, 1]), \
            f"counted voter {nm} must match the representative it is credited to"


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


def test_batch_equivalence_diverges_on_multi_match():
    """The default MUST diverge from the legacy batch grouper, and must diverge by being right.

    The pool contains a witness cell inside the tolerance of BOTH halves of a split true lattice --
    the case test_batch_equivalence's pool happens not to contain, which is why that test passed
    unchanged when the default flipped. Legacy batch and merge=False both hand the win to the
    spurious group; the default coalesces the halves and recovers the truth.
    """
    import os
    prev = os.environ.get("GLINT_CONSENSUS_STABLE")
    os.environ["GLINT_CONSENSUS_STABLE"] = "0"                  # legacy arrival-order batch grouping
    try:
        import importlib
        import glint.multishot as ms
        importlib.reload(ms)
        TRUE = _diag(79.0, 79.0, 38.0)
        A = _diag(79.0 * 0.972, 79.0 * 0.972, 38.0 * 0.972)
        B = _diag(79.0 * 1.028, 79.0 * 1.028, 38.0 * 1.028)
        SPUR = _diag(61.0, 67.0, 73.0)
        pool = [A] * 9 + [B] * 9 + [SPUR] * 12 + [_diag(79.0, 79.0, 38.0)]   # last matches A and B both

        rep_b, sup_b = ms.consensus_cell(pool, min_support=3, medoid=False)
        assert not ms.same_lattice(rep_b, TRUE), "premise: legacy batch must pick the spurious group"

        rc_off = RunningConsensus(min_support=3, gap=2, adaptive=False, merge=False)
        for M in pool:
            rc_off.add_frame([M])
        rep_off, sup_off, _ = rc_off.verdict(gap=0)
        assert _rp_eq(rep_off, rep_b) and sup_off == sup_b, (sup_off, sup_b)   # merge=False still equivalent

        rc_on = RunningConsensus(min_support=3, gap=2, adaptive=False)         # the DEFAULT
        for M in pool:
            rc_on.add_frame([M])
        rep_on, sup_on, _ = rc_on.verdict(gap=0)
        assert ms.same_lattice(rep_on, TRUE), "default must recover the true lattice"
        assert sup_on > sup_b, (sup_on, sup_b)                                 # and on more votes
    finally:
        if prev is None:
            os.environ.pop("GLINT_CONSENSUS_STABLE", None)
        else:
            os.environ["GLINT_CONSENSUS_STABLE"] = prev
        import importlib
        import glint.multishot as ms
        importlib.reload(ms)


def test_vectorised_filter_agrees_with_match():
    """The merge path's vectorised group filter must be EXACTLY _match, group for group.

    merge=False still calls _match in a Python loop; the default replaces it with one numpy
    expression over the parallel fingerprint arrays. If the two ever disagree the two modes silently
    partition differently for reasons unrelated to merging, so pin them against each other on a
    population containing near-misses on each of the three tests (volume, lengths, angles).
    """
    rng = np.random.default_rng(11)
    rc = RunningConsensus(min_support=3, gap=2, adaptive=False)
    for _ in range(150):                                     # a varied population of groups
        rc.add(np.diag(1.0 / rng.uniform(30.0, 250.0, 3)))
    # Seed the POPULATION with the probe family too, or the probes match nothing and the volume
    # term is never exercised. It is the term that discriminates a uniform rescale: at x1.049 every
    # length is inside rtol=5% while the volume is off by 15.4%, past vtol=10%, so lengths+angles
    # alone would accept a cell the real _match rejects.
    base = np.diag(1.0 / np.array([79.0, 79.0, 38.0]))
    for f in (1.0, 1.03, 1.049, 1.06):
        rc.add(np.diag(1.0 / (np.array([79.0, 79.0, 38.0]) * f)))
    probes = [base]
    for f in (1.0, 1.02, 1.049, 1.051, 1.2, 0.95, 2 ** (1 / 3)):   # straddle rtol and the volume test
        probes.append(np.diag(1.0 / (np.array([79.0, 79.0, 38.0]) * f)))
    probes += [np.diag(1.0 / rng.uniform(30.0, 250.0, 3)) for _ in range(60)]

    for M in probes:
        l, c = reduced_params(np.asarray(M, float)); d = abs(np.linalg.det(np.asarray(M, float)))
        vec = set(int(i) for i in rc._candidates(l, c, d))   # the MODULE's filter, not a copy
        ref = {i for i, g in enumerate(rc.groups) if rc._match(l, c, d, g)}
        assert vec == ref, (sorted(vec ^ ref), len(rc.groups))


def test_fingerprint_arrays_stay_aligned_with_groups():
    """The parallel fingerprint arrays must stay index-aligned with self.groups, including MERGES.

    A merge removes groups from the list, so the arrays have to be re-derived; if they are not, _n
    over-counts and _candidates() returns indices into a list that has shrunk -- silently selecting
    the wrong group, or raising. Nothing else in this file exercises that, because a stale array
    only misbehaves once a merge has actually removed something.
    """
    A = _diag(79.0 * 0.972, 79.0 * 0.972, 38.0 * 0.972)
    B = _diag(79.0 * 1.028, 79.0 * 1.028, 38.0 * 1.028)
    WITNESS = _diag(79.0, 79.0, 38.0)
    rng = np.random.default_rng(5)
    rc = RunningConsensus(min_support=3, gap=2, adaptive=False)

    def invariant(where):
        assert rc._n == len(rc.groups), (where, rc._n, len(rc.groups))
        for i, g in enumerate(rc.groups):
            assert abs(rc._dets[i] - g[3]) < 1e-12, (where, i)
            assert np.allclose(rc._lens[i], g[1]) and np.allclose(rc._cos[i], g[2]), (where, i)

    for _ in range(40):
        rc.add(np.diag(1.0 / rng.uniform(30.0, 250.0, 3)))
    invariant("after fills")
    before = len(rc.groups)
    rc.add(A); rc.add(B)
    invariant("after A,B")
    rc.add(WITNESS)                                          # forces a merge of the A and B groups
    assert len(rc.groups) < before + 2, (len(rc.groups), before)   # a merge really happened
    invariant("after merge")
    for _ in range(20):                                      # keep going: a stale array breaks here
        rc.add(np.diag(1.0 / rng.uniform(30.0, 250.0, 3)))
    invariant("after post-merge fills")


if __name__ == "__main__":
    tests = (test_batch_equivalence, test_early_stop_locks_clean,
             test_refuses_ambiguous_race, test_adaptive_gap_raises_when_slow,
             test_pool_switch_keys_scale_tests_on_pool_size,
             test_pool_switch_running_lock_unchanged_small_refuses_large,
             test_merge_repairs_split_vote, test_merge_false_is_bit_identical_to_pre_fix,
             test_merge_is_on_by_default,
             test_batch_equivalence_diverges_on_multi_match,
             test_vectorised_filter_agrees_with_match,
             test_fingerprint_arrays_stay_aligned_with_groups)
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
