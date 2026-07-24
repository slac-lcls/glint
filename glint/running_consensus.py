"""Running-histogram cross-frame consensus with a sequential early-stop.

The batch ``consensus_cell`` (glint.multishot) pools ALL frames' candidate cells, then greedily groups
them by ``same_lattice`` and returns the max-weight group if its support >= min_support. This module runs
the *identical* grouping INCREMENTALLY -- one frame at a time -- so a streaming driver (task #52) can test
after every frame and LOCK the consensus cell the instant it is unambiguous, instead of waiting for a
fixed batch. Locking is what flips the driver from blind indexing (~26 ms/frame) to the batched
known-cell rescue path (~0.26 ms/frame).

Equivalence: feed every frame, then ``verdict(gap=0)`` returns the same max-weight group as
``consensus_cell`` -- same ``reduced_params`` fingerprint, same (rtol, ctol, vtol) tolerances.

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
                 "groups", "nframes")

    def __init__(self, min_support=3, gap=2, rtol=DEF_RTOL, ctol=DEF_CTOL, vtol=DEF_VTOL,
                 adaptive=True):
        self.min_support = int(min_support)
        self.base_gap = int(gap); self.gap = int(gap)
        self.rtol = rtol; self.ctol = ctol; self.vtol = vtol
        self.adaptive = bool(adaptive)
        self.groups = []                                  # each: [rep_M, lens, cos, det, weight]
        self.nframes = 0

    def _match(self, l, c, d, g):
        lg, cg, dg = g[1], g[2], g[3]
        return (abs(d - dg) <= self.vtol * dg
                and bool(np.all(np.abs(l - lg) <= self.rtol * lg))
                and bool(np.all(np.abs(c - cg) <= self.ctol)))

    def add(self, M):
        """Add ONE candidate cell (3x3 basis, columns a,b,c) to the running histogram."""
        M = np.asarray(M, float)
        l, c = reduced_params(M); d = abs(np.linalg.det(M))
        for g in self.groups:
            if self._match(l, c, d, g):
                g[4] += 1; return
        self.groups.append([M, l, c, d, 1])

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
        batch consensus_cell result exactly."""
        g = self.gap if gap is None else int(gap)
        rep, w0, w1 = self.leaders()
        if rep is not None and w0 >= self.min_support and (w0 - w1) >= g:
            return rep, w0, w1
        return None, w0, w1
