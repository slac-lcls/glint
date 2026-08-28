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

REGIME -- Q MUST BE ONE ORIENTATION. Both halves of the score are computed in the leader's frame, so
they only mean anything if the peaks actually sit in that frame. Hand `confirm` a *pooled* cloud from
many frames at different orientations and the test does not merely get noisy, it inverts into a pure
`1/V` prior on cell volume: coverage collapses to the chance value `(2*hkl_tol)^3` (0.027 by default,
see `null_coverage`) for EVERY candidate, so `score = coverage * occupancy` reduces to occupancy, and
occupancy is `#occupied / #nodes-in-shell` with a node count proportional to the real-cell volume.
Every half-volume derivative then scores ~2x the leader and the gate refuses whatever it is shown.
Measured on the cxidb-62 consensus lock (37,531 pooled peaks from 314 voting frames, job 34880744):
coverage 0.0276-0.0302 across the whole index-2 family against a 0.027 chance value, all five
half-volume derivatives at 1.5-1.8x the leader, and a randomly ROTATED copy of the leader scoring
within 1.26x of the as-locked one -- i.e. the pooled score carried essentially no orientation
information. The same gate on the same lock, evaluated per frame, confirmed the true cell on 30 of 40
frames with a median best/leader of exactly 1.000.

Two defences follow from that measurement. `confirm` ABSTAINS (returns the leader untouched) unless the
leader's coverage clears `coverage_floor`, which sits a safety factor above the coverage at which an
ideal half-volume alias starts winning on volume alone -- so an orientation-incoherent input can no
longer produce a verdict at all. And `confirm_frames` is the API a cross-frame consensus lock should
use: it scores each voting frame in ITS OWN orientation and refuses only when a derivative beats the
leader on a majority of frames, which is what a systematic alias does and what per-frame noise does not.

Note the abstain path deliberately gives up a refusal the old code did emit: a wholly spurious lattice
also sits at chance coverage, and used to be refused for the same wrong reason a correct one was.
Detecting *that* is the spurious meter's job (glint.spurious_meter.null_margin, wired as the driver's
lock_z), not the alias gate's -- this gate answers "right lattice, wrong scale?", not "is there any
signal here at all?".

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
def null_coverage(hkl_tol=0.15):
    """Coverage a cell scores on peaks it bears NO relation to -- the value `tightness` returns when the
    peaks are not in the cell's orientation. Each of the three fractional Miller coordinates lands within
    `hkl_tol` of an integer with probability 2*hkl_tol and they are independent, so chance coverage is
    (2*hkl_tol)^3 = 0.027 at the default tolerance, for ANY non-degenerate cell. That cell-independence
    is the whole problem with a pooled input: it leaves occupancy (~1/V) as the only live term.
    (Asymptotic in the usual sense -- it assumes the peaks spread over many nodes rather than clustering
    around hkl=000, which is the case whenever qmax covers more than a couple of reciprocal periods.)"""
    return float(min(2.0 * float(hkl_tol), 1.0) ** 3)


def coverage_floor(hkl_tol=0.15, margin=1.10, max_index=2, safety=2.0):
    """Least leader coverage at which a verdict is worth issuing -- derived, not tuned.

    Let f be the fraction of the peaks genuinely in the leader's orientation; the rest index at chance,
    `null_coverage` (c0). Away from saturation the occupied-node count is just the inlier count N*cov, so

        score = cov * occupancy = cov * (N*cov / npred)  ~  N * cov^2 / V

    since the node count in a fixed shell is proportional to the real-cell volume. For an index-k
    derivative of 1/k the volume, which keeps 1/k of the genuine peaks and all of the chance ones,

        cov_leader = f + (1-f)c0 ,  cov_alias = f/k + (1-f)c0 ,
        score_alias / score_leader = k * (cov_alias / cov_leader)^2

    -- 1/k at f=1 (the alias loses cleanly, which is why the test works when it is applicable) rising to
    k as f -> 0 (the score IS the 1/V prior). At the cxidb-62 lock f was 0.0028, predicting 1.82 against
    the 1.77 measured -- and inverting that f recovers ~360 frames' worth of dilution against 314 real
    voters, which is the arithmetic of pooling in one line. Requiring the ratio to stay within `margin`
    bounds f from below; the floor is the coverage there, with s = sqrt(margin/k):

        f >= c0(1-s) / [(s - 1/k) + c0(1-s)]

    0.054 at the defaults (k=2, margin=1.10, c0=0.027) = 2.0x chance, i.e. the leader must index ~3% of
    the peaks over and above chance. That is a break-even, though, and break-even is not where to stand:
    the verdict takes a MAX over ~10 derivatives, and the winner of a max is biased upward, so measured
    ratios cross the margin while the ideal-alias model still says the leader wins (1.16 at 2.5x chance
    on the synthetic pool in experiments/test_alias_gate.py, where the model predicts 0.97). Hence
    `safety`, default 2: the shipped floor is 0.109 = 4.0x chance.

    That factor costs nothing, because the two regimes are not close. Measured on the cxidb-62 voters
    (job 34906984): per-frame leader coverage 0.294 / 0.390 / 0.876 (min / p5 / median over 80 frames) =
    10.9-32.5x chance, against 1.10x for the pooled cloud of those same frames. The floor sits a factor
    2.7 below the weakest real frame and a factor 3.7 above the pooled input -- an empty decade, not a
    tuned edge.

    (The no-saturation assumption is the one to watch: if the peaks outnumber the nodes in the shell
    every occupancy pins at 1, the score collapses to coverage alone, and the leader then wins by
    construction -- a regime real sparse stills never reach, but synthetic generators that put a fixed
    fraction of EVERY node on every frame do.)"""
    c0 = null_coverage(hkl_tol)
    k = float(max_index)
    s = float(np.sqrt(float(margin) / k))
    denom = (s - 1.0 / k) + c0 * (1.0 - s)
    if s >= 1.0 or denom <= 0.0:                 # margin >= k: no alias can win on volume alone -> no floor
        return 0.0
    f = float(min(c0 * (1.0 - s) / denom, 1.0))
    return float(min(safety * (f + (1.0 - f) * c0), 1.0))


def _family_index(fam_params, W, rtol=0.05, ctol=0.06):
    """Index of the leader-family member matching cell W (reduced-cell params, so basis- and
    orientation-invariant), or -1. Needed because a per-frame winner is expressed in that frame's basis:
    the index-<=N derivative SET of a lattice is basis-independent, but the enumeration ORDER is not, so
    winners are matched back by cell, never by position."""
    lw, cw = reduced_params(W)
    if not (np.all(np.isfinite(lw)) and np.all(np.isfinite(cw))):
        return -1
    best, bd = -1, np.inf
    for j, (lj, cj) in enumerate(fam_params):
        dl = np.abs(lj - lw) / np.maximum(lw, 1e-9)
        dc = np.abs(cj - cw)
        if not (np.all(dl <= rtol) and np.all(dc <= ctol)):
            continue
        d = float(dl.max() + dc.max())
        if d < bd:
            bd, best = d, j
    return best


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
    """Confirm (or reject) a lock by the Occam-tightness test over the leader's derivative lattices.

    Two entry points, and picking the right one is not a style choice: `confirm` takes ONE frame's peaks
    in the leader's orientation, `confirm_frames` takes a CROSS-FRAME lock as (q, M) pairs and votes the
    per-frame verdicts. Pooling frames into `confirm` is the failure this class is now built around --
    see the module docstring.

    Default `adopt=False` REFUSES an ambiguous lock (returns None -> the driver keeps
    accumulating votes so cross-frame consensus can resolve the cell -- the conservative streaming
    choice, and the one that matters: it reliably STOPS a super-cell alias from locking). `adopt=True`
    instead swaps in the tightest derivative; that is the true cell on rich or multi-frame data, but a
    SINGLE sparse frame can under-determine which derivative is correct (a smaller cell may explain the
    observed peaks just as tightly -- see the pseudo-centering case in the tests), so adopt is best-
    effort per-frame repair, not a guaranteed recovery. Prefer the default refuse in streaming."""

    __slots__ = ("max_index", "margin", "hkl_tol", "adopt", "min_coverage", "frac_beat", "min_frames",
                 "last", "info")

    def __init__(self, max_index=2, margin=1.10, hkl_tol=0.15, adopt=False, min_coverage=None,
                 frac_beat=0.5, min_frames=3):
        self.max_index = int(max_index)
        self.margin = float(margin)              # an alias must be >margin x tighter to overrule the leader
        self.hkl_tol = float(hkl_tol)
        self.adopt = bool(adopt)
        # Applicability floor, derived from the other three by `coverage_floor` (default 0.109 = 4.0 x
        # chance): below it the score is the 1/V prior rather than a fit, so the gate abstains. Pass a
        # number to override, or 0.0 for the pre-fix behaviour of always answering.
        self.min_coverage = (coverage_floor(self.hkl_tol, self.margin, self.max_index)
                             if min_coverage is None else float(min_coverage))
        self.frac_beat = float(frac_beat)        # confirm_frames: share of frames an alias must win to overrule
        # ...and at least this many testable frames before that share means anything. A cross-frame gate
        # that fires on one frame's opinion is not a cross-frame gate; 3 mirrors consensus_cell's own
        # absolute min_support floor.
        self.min_frames = int(min_frames)
        self.last = None                         # (leader_score, best_score, best_is_leader) for logging
        self.info = None                         # dict: verdict + the numbers behind it

    # ------------------------------------------------------------------ single orientation (one frame)
    def confirm(self, M_leader, Q):
        """Return the cell to lock: M_leader if it is (near-)tightest; else the tighter alias (adopt) or
        None (refuse). Q = observed reciprocal peaks of ONE frame, in the orientation M_leader is
        expressed in. For a cross-frame consensus lock use `confirm_frames` -- pooling frames here is the
        exact input that degenerates to a 1/V prior (see the module docstring), and is now abstained on
        rather than answered."""
        M_leader = np.asarray(M_leader, float)
        Q = np.asarray(Q, float)
        if Q.ndim != 2 or Q.shape[0] == 0:
            self.last = (0.0, 0.0, True)
            self.info = {"verdict": "confirm", "reason": "no peaks to test against"}
            return M_leader                      # nothing to test against -> don't block the lock
        lead, cov, _ = tightness(M_leader, Q, self.hkl_tol)
        floor = self.min_coverage
        if cov < floor:
            # The leader does not index these peaks better than an unrelated cell would, so no comparison
            # among ITS derivatives can be evidence about aliasing -- every score here is the 1/V prior.
            self.last = (lead, lead, True)
            self.info = {"verdict": "abstain", "reason": "leader at chance coverage (input not one "
                         "orientation, or not this lattice)", "coverage": cov, "cov_floor": floor,
                         "null_coverage": null_coverage(self.hkl_tol)}
            return M_leader
        cands = derivative_lattices(M_leader, self.max_index)
        scores = [lead] + [tightness(M, Q, self.hkl_tol)[0] for M in cands[1:]]
        bi = int(np.argmax(scores))
        best = scores[bi]
        self.last = (lead, best, bi == 0)
        confirmed = best <= lead * self.margin   # no alias beats the leader by the required factor
        self.info = {"verdict": "confirm" if confirmed else ("adopt" if self.adopt else "refuse"),
                     "coverage": cov, "leader_score": lead, "best_score": best, "best_index": bi}
        if confirmed:
            return M_leader
        return cands[bi] if self.adopt else None

    # -------------------------------------------------------------- many orientations (a consensus lock)
    def confirm_frames(self, M_leader, frames):
        """Confirm a CROSS-FRAME lock by voting single-frame verdicts, each computed in its own frame's
        orientation.

        `frames` = iterable of (q, M) pairs: q the frame's reciprocal peaks (N x 3) and M the leader's
        lattice as oriented on THAT frame (in consensus this is free -- it is the frame's own N-best
        hypothesis that agreed with the lock). Frames whose leader coverage is at chance are dropped as
        untestable rather than counted as either verdict, and if fewer than `min_frames` survive that
        drop the gate abstains instead of ruling on a handful.

        A real alias is systematic: it out-tightens the leader on essentially every frame, because the
        absences that betray it are a property of the lattice, not of the shot. Per-frame noise is not,
        which is why the decision is a share (`frac_beat`, default: more than half the testable frames)
        and not a single frame's opinion. Returns the leader (confirm/abstain), the plurality alias
        (adopt), or None (refuse). `self.info` carries the counts; `self.last` is
        (n_tested, n_beaten, confirmed) here rather than confirm's (leader, best, best_is_leader),
        since there is no single pair of scores to report."""
        M_leader = np.asarray(M_leader, float)
        floor = self.min_coverage
        n_seen = n_tested = 0
        winners, ratios = [], []
        for q, M in frames:
            n_seen += 1
            q = np.asarray(q, float)
            M = np.asarray(M, float)
            if q.ndim != 2 or q.shape[0] == 0 or _degenerate(M):
                continue
            lead, cov, _ = tightness(M, q, self.hkl_tol)
            if cov < floor or lead <= 0.0:
                continue                          # this frame cannot testify about this lattice
            cands = derivative_lattices(M, self.max_index)
            scores = [lead] + [tightness(c, q, self.hkl_tol)[0] for c in cands[1:]]
            bi = int(np.argmax(scores))
            n_tested += 1
            ratios.append(scores[bi] / lead)
            if bi != 0 and scores[bi] > lead * self.margin:
                winners.append(cands[bi])
        frac = (len(winners) / n_tested) if n_tested else 0.0
        med = float(np.median(ratios)) if ratios else 0.0
        self.last = (float(n_tested), float(len(winners)), frac <= self.frac_beat)
        base = {"frames_seen": n_seen, "frames_tested": n_tested, "frames_beaten": len(winners),
                "frac_beaten": frac, "median_best_over_leader": med}
        if n_tested < max(self.min_frames, 1):
            self.info = dict(base, verdict="abstain",
                             reason=f"only {n_tested} frame(s) indexed the leader above chance")
            return M_leader
        if frac <= self.frac_beat:
            self.info = dict(base, verdict="confirm")
            return M_leader
        if not self.adopt:
            self.info = dict(base, verdict="refuse")
            return None
        # Adopt: the winners are in per-frame bases, so map each back onto the leader's own family by
        # reduced cell and take the plurality. No match (or no majority) means we know the leader is
        # wrong but not what to replace it with -> refuse rather than guess.
        fam = derivative_lattices(M_leader, self.max_index)
        fam_params = [reduced_params(F) for F in fam]
        counts = np.zeros(len(fam))
        for W in winners:
            j = _family_index(fam_params, W)
            if j >= 0:
                counts[j] += 1
        j = int(np.argmax(counts))
        if counts[j] == 0 or j == 0:
            self.info = dict(base, verdict="refuse", reason="no derivative won a plurality")
            return None
        self.info = dict(base, verdict="adopt", adopted_index=j, adopted_votes=int(counts[j]))
        return fam[j]
