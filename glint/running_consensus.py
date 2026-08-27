"""Running-histogram cross-frame consensus with a sequential early-stop.

The batch ``consensus_cell`` (glint.multishot) pools ALL frames' candidate cells, then greedily groups
them by ``same_lattice`` and returns the max-weight group if its support >= min_support. This module
runs that grouping INCREMENTALLY -- one frame at a time -- so a streaming driver (task #52) can test
after every frame and LOCK the consensus cell the instant it is unambiguous, instead of waiting for a
fixed batch. It is the batch path's ARRIVAL-ORDER grouping that is reproduced here, which was the
only kind there was when this was written and is now the legacy one: the batch path has seeded from
the densest neighbourhood since glint#102 and this path has not (GROUPING ORDER, below, is the whole
story and it is not a footnote -- the two can pick different winners from the same hypotheses).
Locking is what flips the driver from blind indexing (~26 ms/frame) to the batched known-cell
rescue path (~0.17 ms/frame).

Equivalence, and its EXPIRY DATE: feed every frame, then ``verdict(gap=0)`` returns the same
max-weight group as ``consensus_cell`` -- same ``reduced_params`` fingerprint, same (rtol, ctol,
vtol) tolerances -- but only under BOTH ``GLINT_CONSENSUS_STABLE=0`` (so the batch path also groups
in arrival order) AND ``merge=False`` (so this path does too). Under the default the batch path
seeds from the densest neighbourhood (glint#102) and the two can return DIFFERENT winners on the
same hypotheses; see GROUPING ORDER below, where they do. The claim held unconditionally when it
was written, became conditional on STABLE=0 when #102 landed, and became conditional on
``merge=False`` as well when MERGE-ON-MULTI-MATCH landed.

That second condition is not a regression, it is the point. The legacy batch grouper assigns a
multi-match witness to the FIRST group it matches and leaves the others un-coalesced, so on a pool
containing such a witness it reproduces the split-vote failure. Measured on the synthetic pool in
``test_batch_equivalence_diverges_on_multi_match``: legacy batch and ``merge=False`` both return the
SPURIOUS group at support 12, while the default returns the TRUE lattice at 19. Equivalence to a
partition that is known-broken is worth exactly what it costs to keep, which is why ``merge=False``
retains it for audit rather than the default preserving it.

GROUPING ORDER -- a real difference from the batch path. This module groups greedily in the order
hypotheses ARRIVE (first tolerance match wins), which is what ``consensus_cell`` also did until
glint#102 gave the batch path densest-neighbourhood seeding. Tolerance matching is not transitive,
so the two can partition the same hypotheses differently, and on bad geometry they do: on mfxx49820
r0016 (2228 frames, UNREFINED psana geometry) the pooled vote returns the true lattice at 27/6294
under densest seeding, and a DOUBLED cell at 23/6294 (volume 2.06x truth) under arrival order --
the recorded item-7 failure, which reproduces today only with ``GLINT_CONSENSUS_STABLE=0``. So the
number quoted throughout this repo is exact, but it is a PRE-#102 measurement; the batch path has
since outgrown that failure and this one has not.

Adopting #102 here is possible but not free: this class keeps only group representatives and
weights, so densest seeding would mean retaining the full hypothesis pool and re-grouping it
periodically -- O(n) work repeated on a path whose whole purpose is to decide after every frame.
It defends with the acceptance gate instead, and the defence is worth what it costs. Replaying that
run over 400 random arrival orders:

    gap-only (shipped)          locks 342/400, of which 194 are the WRONG lattice
                                (median volume 1.94x truth -- the doubled cell, live)
    + frac .02 / lead 1.5       locks   3/400, of which   2 wrong
    + the same, pool_switch=72  locks   3/400, of which   2 wrong -- identical protection,
                                and 0 of 400 clean-benchmark orders change (see consensus_accept)

The residual 2 are not a tuning failure. They lock at pools of 54-96, where a 2% share test asks
for 1.08 votes and a support of 3 clears it without meaning anything: neither scale-free test can
protect the small-pool tail, and only ``gap`` is doing work down there.

MERGE-ON-MULTI-MATCH (``merge=True``, the default) is the cheap repair that does not need the pool.
When an arriving cell falls inside the tolerance of SEVERAL groups it is a WITNESS that those groups
are one lattice the arrival order split, so they are folded together. It is NOT free: finding every
match cannot short-circuit the way first-match-wins did, so ``_candidates`` scans the whole group
set. The scan is one vectorised numpy expression over parallel fingerprint arrays rather than a
Python loop, which is what keeps the cost bounded -- 147 us/hypothesis against 121 for the
short-circuit path at ~700 groups, most of the remaining gap being the ``reduced_params`` call both
paths already pay (it was 328 us before the arrays). What it does NOT cost is memory: the witness is
the arriving cell, so nothing is retained. It fixes the failure this path was losing to. Measured on a real
refined-geometry run (small-molecule cell, 665 frames, 3325 hypotheses): the true lattice's votes
FRAGMENTED across two groups, each within 5% of truth but 5.5-5.9% from EACH OTHER, so neither
absorbed the other and a spurious group outranked both halves. Shipped grouping locked at frame 31
on the WRONG cell (pooled winner weight 217); with the merge it locks at frame 27 on the RIGHT one
(pooled winner 312 = the two halves recombined), and the partition changes only where it must --
1156 groups become 1119, 37 merges. That failure is the same shape as the recorded item-7 one but on
REFINED geometry, so it is NOT confined to bad geometry as previously believed.

The merge does not disturb the clean benchmark. Over 2000 random streaming orders on the cxidb
frames, merge and shipped are INDISTINGUISHABLE: at n=120 both lock 2000/2000 with 0 false-locks
(median 8 frames); at n=480 both lock 2000/2000 with 1 false-lock (0.05%, median 9 frames) -- and on
the SAME alias, a doubled c axis, so n=480 exposes a pre-existing supercell false-lock in BOTH rather
than a new one from the merge. ``merge=False`` restores first-match-wins exactly.


Stop rule: lock when the leading group's support >= ``min_support`` AND it leads the runner-up by >=
``gap``. The gap is the specificity knob: gap=0 occasionally false-locks on the coincident sublattice
aliases (real cxidb: the face-diagonal cell |q|=sqrt(a^2+c^2) recurs ~once/frame, a coherent ~2:1 rate
race with the true cell, NOT scatter); gap>=2 is clean on clean data. When ``adaptive`` is on, the gap is
raised as the leader accumulates slowly -- the live, model-free signature of the mosaic regime -- so
false-locks stay ~0 as the data hardens, at the cost of latency; on the hardest data it refuses to lock
(returns ``None``) rather than lock wrong. Measured (memory glint-sequential-stop-consensus): 120 sparse
cxidb frames -> median 6 frames to lock, 0/400 false-locks over random streaming orders; the mosaic knee
at sigma~=0.0015 needs gap 3, and sigma>=0.0020 correctly refuses.
"""
import numpy as np
from .lattice import reduced_params

DEF_RTOL, DEF_CTOL, DEF_VTOL = 0.05, 0.06, 0.10          # == consensus_cell defaults


def consensus_accept(win, runner, n_pool, min_support=3, min_frac=0.0, min_lead=1.0, min_gap=0,
                     pool_switch=0):
    """THE acceptance rule for a consensus cluster -- shared by BOTH consensus paths.

    It lives here, and `multishot.consensus_cell` imports it, because the two paths were each
    enforcing a different SUBSET of these tests and therefore disagreed on the same data. Measured on
    mfxx49820 r0016 triaged to the top-16 frames (leader 3 of 48 hypotheses): `consensus_cell`
    ACCEPTED (3 clears its 2% pool share, and 3 >= 1.5x a runner-up of <=2) while `RunningConsensus`
    REFUSED (its adaptive gap wanted an absolute lead of 3). Neither was wrong; they were answering
    different questions and calling both answers "the consensus".

    Four tests, and a caller may use any subset -- what matters is that there is now ONE definition
    of each, so "we require a margin over the runner-up" means the same thing on both paths:

      min_support  absolute floor on the leader.
      min_frac     leader as a SHARE of the pool. The random-agreement floor grows with the number
                   of hypotheses drawn, so a fixed absolute floor cannot hold from 360 hypotheses to
                   6300; 23/6294 = 0.37% cleared an absolute 3 and locked a wrong cell. That
                   measurement is PRE-glint#102 and reproduces today only with
                   GLINT_CONSENSUS_STABLE=0 -- see the module docstring, which matters because the
                   streaming path still groups the way it was taken.
      min_lead     leader / runner-up, a RATIO. Scale-free: meaningful whether the leader is 3 or 300.
      min_gap      leader - runner-up, a DIFFERENCE. Not scale-free, and that is the point at small
                   N: with a leader of 3 a ratio of 1.5 tolerates a runner-up of 2, which is a
                   coin-flip, while a gap of 3 demands the field be empty.

    Defaults (min_frac=0, min_lead=1.0, min_gap=0) are inert, so a caller passing none of them gets
    `win >= min_support` exactly.

    pool_switch keys the SHARE and RATIO tests on the pool size: when > 0, min_frac/min_lead apply
    only once n_pool >= pool_switch, so a caller can run gap-dominated at small pools (where an
    absolute margin is the meaningful test) and switch to the scale-free tests once the pool is
    large enough for the random-agreement floor to clear any absolute bar. pool_switch=0 (default)
    applies them at every pool size -- existing callers are unchanged.

    Why a switch rather than simply turning min_frac/min_lead on: they are not free at small pools.
    Measured over 400 random arrival orders of the cxidb-120 benchmark (experiments/nbest_120.npz),
    replayed through RunningConsensus:

        gap-only (shipped)        median 6 frames, p90 9, WORST 18   max n_pool at lock  54
        + frac/lead, always on    median 6 frames, p90 10, WORST 42   max n_pool at lock 126
        + frac/lead, switch 72    median 6 frames, p90 9, WORST 18   -- 0 of 400 orders differ
                                  from gap-only in lock frame, cell, support or runner-up

    72 is chosen against the clean side's DISTRIBUTION, not against a gap between the two regimes --
    there is no such gap. The clean benchmark never locks past a pool of 54, so a switch above 54
    leaves it untouched; but on mfxx49820 r0016 the earliest of 400 locks is at a pool of 54 as
    well, so the two overlap exactly at the tail and no switch can catch that tail. What the switch
    buys is the bulk: r0016's locks sit at a MEDIAN pool of 2480. 48 is already too low (5 of 400
    clean orders diverge); 72 and 96 are both bit-identical.
    """
    if win < min_support:
        return False
    scale_tests = n_pool >= pool_switch if pool_switch > 0 else True
    if scale_tests and min_frac > 0.0 and win < min_frac * n_pool:
        return False
    if scale_tests and min_lead > 1.0 and runner > 0 and win < min_lead * runner:
        return False
    if min_gap > 0 and (win - runner) < min_gap:
        return False
    return True


class RunningConsensus:
    """Incremental same_lattice vote histogram + sequential early-stop.

    Usage (streaming):
        rc = RunningConsensus(min_support=3, gap=2)
        for cells in per_frame_candidate_cells:      # cells = list of 3x3 bases for one frame
            rc.add_frame(cells)
            Mc, sup, lead = rc.verdict()
            if Mc is not None:                       # locked -> switch to known-cell rescue on Mc
                break
    """

    __slots__ = ("min_support", "base_gap", "gap", "rtol", "ctol", "vtol", "adaptive",
                 "groups", "nframes", "npool", "min_frac", "min_lead", "pool_switch", "merge", "_dets", "_lens", "_cos", "_n")

    def __init__(self, min_support=3, gap=2, rtol=DEF_RTOL, ctol=DEF_CTOL, vtol=DEF_VTOL,
                 adaptive=True, min_frac=0.0, min_lead=1.0, pool_switch=0, merge=True):
        self.min_support = int(min_support)
        self.base_gap = int(gap); self.gap = int(gap)
        self.rtol = rtol; self.ctol = ctol; self.vtol = vtol
        self.adaptive = bool(adaptive)
        # Pool-share and ratio tests, both INERT by default so the shipped streaming lock is
        # unchanged. They exist so this path can be made to agree with consensus_cell's, which
        # enforced them while this one did not -- see consensus_accept.
        self.min_frac = float(min_frac); self.min_lead = float(min_lead)
        self.pool_switch = int(pool_switch)
        self.merge = bool(merge)                          # non-transitivity repair; see add()
        # Parallel fingerprint arrays for the merge path's vectorised match: capacity-doubled so a
        # per-hypothesis query is one numpy expression instead of a Python loop over every group.
        self._n = 0
        self._dets = np.empty(64); self._lens = np.empty((64, 3)); self._cos = np.empty((64, 3))
        self.groups = []                                  # each: [rep_M, lens, cos, det, weight]
        self.nframes = 0
        self.npool = 0                                    # hypotheses added, not frames

    def _match(self, l, c, d, g):
        lg, cg, dg = g[1], g[2], g[3]
        return (abs(d - dg) <= self.vtol * dg
                and bool(np.all(np.abs(l - lg) <= self.rtol * lg))
                and bool(np.all(np.abs(c - cg) <= self.ctol)))

    def _grow(self):
        cap = self._dets.size
        if self._n < cap:
            return
        self._dets = np.resize(self._dets, cap * 2)
        self._lens = np.resize(self._lens, (cap * 2, 3))
        self._cos = np.resize(self._cos, (cap * 2, 3))

    def _push_fp(self, l, c, d):
        """Append one group's fingerprint to the parallel arrays."""
        self._grow()
        i = self._n
        self._dets[i] = d; self._lens[i] = l; self._cos[i] = c
        self._n = i + 1

    def _push(self, M, l, c, d):
        self.groups.append([M, l, c, d, 1]); self._push_fp(l, c, d)

    def _rebuild_fp(self):
        """Re-derive the fingerprint arrays from self.groups (after a merge changed the set)."""
        self._n = 0
        for g in self.groups:
            self._push_fp(g[1], g[2], g[3])

    def _candidates(self, l, c, d):
        """Indices of every group this fingerprint matches -- the vectorised form of ``_match``.

        Must agree with ``_match`` group for group: ``merge=False`` still walks groups with
        ``_match``, so a disagreement here would silently partition the two modes differently for
        reasons that have nothing to do with merging. Pinned by
        ``test_vectorised_filter_agrees_with_match``, which calls THIS method rather than
        re-deriving the expression. Volume is first because it is the cheapest and most selective
        term, and it is not redundant with the others: a uniform rescale of x1.049 leaves every
        length inside rtol=5% while the volume moves 15.4%, past vtol=10%.
        """
        n = self._n
        if not n:
            return ()
        dv = self._dets[:n]; lv = self._lens[:n]; cv = self._cos[:n]
        return np.nonzero((np.abs(d - dv) <= self.vtol * dv)
                          & np.all(np.abs(l - lv) <= self.rtol * lv, axis=1)
                          & np.all(np.abs(c - cv) <= self.ctol, axis=1))[0]

    def add(self, M):
        """Add ONE candidate cell (3x3 basis, columns a,b,c) to the running histogram.

        A cell inside the tolerance of SEVERAL groups is direct evidence that those groups are one
        lattice which arrival-order grouping split (tolerance matching is not transitive -- see
        GROUPING ORDER above), so they are folded together. The HEAVIEST group survives as the object
        that carries the combined weight, but its representative is REPLACED by the arriving witness:
        the witness is inside the tolerance of every group it merged, whereas the heaviest group's
        founder need not cover the others, and the representative is what gets locked and gated on.
        This is the repair for non-transitivity that fits THIS path: the witness is the arriving cell,
        so it needs no hypothesis pool and no periodic re-grouping, which is the cost that kept
        glint#102's densest-neighbourhood seeding in the batch path only. It is not free in time --
        see MERGE-ON-MULTI-MATCH above for the measured scan cost. ``merge=False`` restores the
        pre-fix first-match-wins behaviour exactly.
        """
        M = np.asarray(M, float)
        l, c = reduced_params(M); d = abs(np.linalg.det(M))
        self.npool += 1                                   # the denominator min_frac needs
        if not self.merge:
            for g in self.groups:
                if self._match(l, c, d, g):
                    g[4] += 1; return
            self.groups.append([M, l, c, d, 1]); self._push_fp(l, c, d)
            return
        # Volume is the cheap half of _match and rejects almost every group, so gate on it inline
        # before paying for the reduced-parameter comparisons. Without this the multi-match scan
        # cannot short-circuit the way first-match-wins did, and the per-hypothesis cost on a
        # thousand-group pool more than doubles on a path whose whole purpose is streaming latency.
        hits = [self.groups[i] for i in self._candidates(l, c, d)]
        if not hits:
            self._push(M, l, c, d); return
        keep = max(hits, key=lambda g: g[4])
        drop = set()
        for g in hits:
            if g is not keep:
                keep[4] += g[4]; drop.add(id(g))
        if drop:                                          # by IDENTITY: `==` is ambiguous on arrays
            self.groups = [g for g in self.groups if id(g) not in drop]
            # The WITNESS becomes the representative of the coalesced group. It is inside the
            # tolerance of every group it merged -- that is what selected them -- whereas the
            # heaviest group's founder need not cover the others, so keeping the founder would
            # expose a combined support whose voters do not all match the matrix that gets locked.
            # StreamDriver locks this matrix and its alias gate only collects candidates matching
            # it, so an uncovered representative would gate away a chunk of its own voters.
            keep[0] = M; keep[1] = l; keep[2] = c; keep[3] = d
            self._rebuild_fp()
        keep[4] += 1

    def add_frame(self, cells):
        """Add one frame's candidate cells (iterable of 3x3 bases; may be empty / contain None)."""
        for M in cells:
            if M is not None:
                self.add(M)
        self.nframes += 1
        if self.adaptive:
            self._adapt_gap()

    def leaders(self):
        """(leader_cell, leader_support, runner_up_support) -- (None, 0, 0) if no votes yet."""
        if not self.groups:
            return None, 0, 0
        g = sorted(self.groups, key=lambda x: x[4], reverse=True)
        return g[0][0], g[0][4], (g[1][4] if len(g) > 1 else 0)

    def _adapt_gap(self):
        """Raise the gap when the leader accumulates slowly (the mosaic signature).

        rate = leader_support / frames_seen; thresholds reproduce the measured remedies
        (clean lock ~6 frames -> base gap; mosaic knee sigma~=0.0015 ~15 frames -> +1 -> gap 3;
        severe sigma>=0.0020 ~81 frames -> +2 -> gap 4 => safe refusal). See the module docstring."""
        _, w0, _ = self.leaders()
        rate = w0 / max(self.nframes, 1)
        self.gap = self.base_gap + (0 if rate >= 0.30 else 1 if rate >= 0.12 else 2)

    def verdict(self, gap=None):
        """Return (locked_cell, support, lead) if the stop rule fires now, else (None, support, lead).

        gap overrides the (possibly adapted) gap -- pass gap=0 after the last frame to reproduce the
        batch consensus_cell result, which is exact only under GLINT_CONSENSUS_STABLE=0 (the module
        docstring's Equivalence note says why: since glint#102 the batch path seeds from the densest
        neighbourhood and this one still groups in arrival order)."""
        g = self.gap if gap is None else int(gap)
        rep, w0, w1 = self.leaders()
        if rep is not None and consensus_accept(w0, w1, self.npool, self.min_support,
                                                self.min_frac, self.min_lead, g,
                                                self.pool_switch):
            return rep, w0, w1
        return None, w0, w1
