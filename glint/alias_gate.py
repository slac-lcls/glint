"""Lock-time alias gate for the streaming driver's cross-frame consensus.

`RunningConsensus` locks a cell when the same lattice recurs across >= min_support frames with a margin
(the `gap`) over the runner-up (glint.running_consensus). That margin is a *statistical* defence against
the coincident sublattice aliases -- most importantly the face-diagonal cell |q|=sqrt(a^2+c^2), which is
NOT random scatter but a systematic lattice that recurs ~once/frame, racing the true cell ~2:1 (see the
RunningConsensus docstring). The gap wins that race on clean data, but it is thin at the mosaic knee and
cannot fire at all if an alias recurs at nearly the true rate.

This module adds a *deterministic, single-lock* confirmation that does not depend on the vote race. Given
the leader cell the consensus is about to lock and the observed reciprocal-space peaks of the frames that
voted for it, it enumerates the leader's small-index derivative lattices (its index<=N sub- and super-
cells -- the alias family the face-diagonal cell belongs to) and keeps the one that best explains the
data, where "best" is the Occam tightness that neither inlier-count nor vote-count alone can measure:

    tightness(M) = coverage(M) * occupancy(M)

  coverage  = fraction of observed peaks that index to near-integer hkl in M. A coarser (smaller) real
              cell has a sparser reciprocal lattice, so real peaks fall off its nodes and coverage drops.
  occupancy = (# distinct occupied nodes) / (# integer nodes inside the observed resolution shell). A
              finer (larger) super-cell predicts n x as many nodes with the same peaks on them, so a
              (n-1)/n fraction sit systematically ABSENT and its occupancy drops.

The true cell maximises BOTH: a super-cell keeps coverage but loses occupancy, a sub-cell keeps occupancy
but loses coverage. Their product is a clean discriminator that a single frame's inlier count (blind to
absences) and the vote histogram (blind to geometry) both miss -- exactly the case the `gap` handles only
statistically.

Cell convention matches glint.stream_driver._inliers: M's columns are the real cell vectors, so
`hkl = q @ M` and `q = hkl @ inv(M)`.

Opt-in: the streaming driver calls `AliasGate.confirm(M_leader, Q)` only when an `alias_gate=` is passed;
default None => the gate never runs => the lock path is bit-identical to today.
"""
import itertools
import numpy as np

from .lattice import reduced_params


# --------------------------------------------------------------------------- derivative lattices ----
def _diag_triples(index):
    """All ordered positive (a, d, f) with a*d*f == index."""
    out = []
    for a in range(1, index + 1):
        if index % a:
            continue
        rem = index // a
        for d in range(1, rem + 1):
            if rem % d:
                continue
            out.append((a, d, rem // d))
    return out


def hnf_matrices(index):
    """Column-style Hermite normal forms: upper-triangular integer matrices with positive diagonal whose
    product is `index`, off-diagonal entry (i,j<j) reduced modulo its column's diagonal -- one per
    distinct index-`index` sublattice of Z^3 (1 for index 1, 7 for index 2, 13 for index 3)."""
    mats = []
    for a, d, f in _diag_triples(index):
        for h12 in range(d):                     # column 2 diagonal = d
            for h13 in range(f):                 # column 3 diagonal = f
                for h23 in range(f):
                    mats.append(np.array([[a, h12, h13],
                                          [0.0, d, h23],
                                          [0.0, 0.0, f]], float))
    return mats


def _degenerate(M):
    """A basis is unusable if it is (near-)singular -- the blind indexer can return such a candidate on
    off-regime frames, and the gate must refuse it, never crash on inv()."""
    M = np.asarray(M, float)
    return not np.all(np.isfinite(M)) or abs(np.linalg.det(M)) < 1e-9


def _fingerprint(M):
    """same_lattice-style reduced-cell key, rounded so numerically-equal lattices collide. Degenerate
    cells get a unique id-based key so they neither collide nor crash reduced_params."""
    if _degenerate(M):
        return ("degenerate", id(M))
    l, c = reduced_params(M)
    if not (np.all(np.isfinite(l)) and np.all(np.isfinite(c))):
        return ("degenerate", id(M))
    return (tuple(np.round(l, 3)), tuple(np.round(c, 3)))


def derivative_lattices(M, max_index=2):
    """The leader M plus its index in [2, max_index] sub- and super-lattices (M@H and M@inv(H) for every
    Hermite form H), deduplicated by reduced cell. M is always element 0. A degenerate leader yields just
    [M] (nothing to derive from)."""
    M = np.asarray(M, float)
    out = [M]
    if _degenerate(M):
        return out
    seen = {_fingerprint(M)}
    for idx in range(2, max_index + 1):
        for H in hnf_matrices(idx):
            for cand in (M @ H, M @ np.linalg.inv(H)):
                if _degenerate(cand):
                    continue
                fp = _fingerprint(cand)
                if fp not in seen:
                    seen.add(fp)
                    out.append(cand)
    return out


# ------------------------------------------------------------------------------------- tightness ----
def tightness(M, Q, hkl_tol=0.15):
    """(score, coverage, occupancy) of cell M against observed reciprocal peaks Q (N x 3).

    coverage  = fraction of Q within `hkl_tol` of an integer hkl (q @ M near-integer -- same test as
                stream_driver._inliers).
    occupancy = distinct occupied integer nodes / integer nodes inside the observed |q| shell.
    score     = coverage * occupancy (0 if no peaks index)."""
    M = np.asarray(M, float)
    Q = np.asarray(Q, float)
    if Q.ndim != 2 or Q.shape[0] == 0 or _degenerate(M):
        return 0.0, 0.0, 0.0
    hf = Q @ M
    h = np.round(hf)
    inl = np.abs(hf - h).max(1) < hkl_tol
    coverage = float(inl.mean())
    ninl = int(inl.sum())
    if ninl == 0:
        return 0.0, coverage, 0.0

    occ_nodes = {tuple(x) for x in h[inl].astype(int)}
    qmax = float(np.sqrt(np.einsum("ij,ij->i", Q, Q)).max())
    Minv = np.linalg.inv(M)                                   # hkl -> q : q = hkl @ Minv
    rlen = np.sqrt((Minv ** 2).sum(1))                        # |dq| per unit step in h,k,l
    hb = np.ceil(qmax / np.maximum(rlen, 1e-12)).astype(int) + 1
    grid = np.array(list(itertools.product(range(-hb[0], hb[0] + 1),
                                           range(-hb[1], hb[1] + 1),
                                           range(-hb[2], hb[2] + 1))), float)
    qg = grid @ Minv
    qgn = np.sqrt(np.einsum("ij,ij->i", qg, qg))
    npred = int(((qgn > 1e-9) & (qgn <= qmax)).sum())
    occupancy = len(occ_nodes) / max(npred, 1)
    return coverage * occupancy, coverage, occupancy


# -------------------------------------------------------------------------------------- the gate ----
class AliasGate:
    """Confirm (or reject) a consensus lock by the Occam-tightness test over the leader's derivative
    lattices. Default `adopt=False` REFUSES an ambiguous lock (returns None -> the driver keeps
    accumulating votes so cross-frame consensus can resolve the cell -- the conservative streaming
    choice, and the one that matters: it reliably STOPS a super-cell alias from locking). `adopt=True`
    instead swaps in the tightest derivative; that is the true cell on rich or multi-frame data, but a
    SINGLE sparse frame can under-determine which derivative is correct (a smaller cell may explain the
    observed peaks just as tightly -- see the pseudo-centering case in the tests), so adopt is best-
    effort per-frame repair, not a guaranteed recovery. Prefer the default refuse in streaming."""

    __slots__ = ("max_index", "margin", "hkl_tol", "adopt", "last")

    def __init__(self, max_index=2, margin=1.10, hkl_tol=0.15, adopt=False):
        self.max_index = int(max_index)
        self.margin = float(margin)              # an alias must be >margin x tighter to overrule the leader
        self.hkl_tol = float(hkl_tol)
        self.adopt = bool(adopt)
        self.last = None                         # (leader_score, best_score, best_is_leader) for logging

    def confirm(self, M_leader, Q):
        """Return the cell to lock: M_leader if it is (near-)tightest; else the tighter alias (adopt) or
        None (refuse). Q = observed reciprocal peaks of the voting frames (N x 3)."""
        M_leader = np.asarray(M_leader, float)
        Q = np.asarray(Q, float)
        if Q.ndim != 2 or Q.shape[0] == 0:
            self.last = (0.0, 0.0, True)
            return M_leader                      # nothing to test against -> don't block the lock
        cands = derivative_lattices(M_leader, self.max_index)
        scores = [tightness(M, Q, self.hkl_tol)[0] for M in cands]
        lead = scores[0]
        bi = int(np.argmax(scores))
        best = scores[bi]
        self.last = (lead, best, bi == 0)
        if best <= lead * self.margin:           # no alias beats the leader by the required factor
            return M_leader
        return cands[bi] if self.adopt else None
