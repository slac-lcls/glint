"""The retry ARMS, in one place, so the streaming driver and the offline measurement run the SAME
code (glint#75).

`experiments/offline_retry_arsenal.py` measured, on the 42 frames the batched known-cell pass rejects
of the cxidb-120 set, what each retry arm recovers:

    blind top-1               10      88/120
    blind N-best (k=3)        10      88/120
    blind N-best (k=10)       11      89/120
    known-cell, per-frame      9      87/120
    UNION, blind arms only    11      89/120
    UNION, all AVAILABLE      16      94/120   <-- beats offline's shipped 91

The headline is that the arms are COMPLEMENTARY, not redundant: 11 and 9 individually, 16 together.
They rescue different frames, so a CASCADE beats any single stronger retry. That is why this module
exists as a cascade of small arms rather than one "better retry".

Why the arms live here and not in the experiment script: the 94/120 above is a published number, and
a re-implementation in the driver would silently break its correspondence with the measurement. The
experiment now imports these functions, so the two cannot drift.

Two properties of the arm implementations are load-bearing and easy to lose:

  * `arm_blind_nbest` tests EVERY candidate and returns at the first one that clears the gate. The
    #76 code broke out of the loop unconditionally after candidate 0, which silently made an N-best
    retry a top-1 retry (fixed in e5ec454); with the break inside the test, k=10 reaches 11 rather
    than 10.
  * `arm_known_perframe` is the PER-FRAME known-cell indexer (replica_gpu.index_known_gpu_cell), NOT
    the batched one (replica_gpu_batch.index_fused). They do not agree frame by frame -- measured
    102/120 agreement at the voted cell, 9 frames each way -- so the per-frame arm recovers 9 frames
    the batched pass has already rejected. Substituting the batched indexer here makes this arm a
    no-op by construction.

Each arm takes an already-bound `gate(M) -> bool` and returns a cell matrix or None, so the caller
decides what "good enough" means: the offline measurement uses the strict research gate
(same_lattice + >=25% matched + >=10 reflections), the live driver uses StreamDriver._fits plus a
same_lattice check against the cell being retried against.
"""
import numpy as np

# The cascade the issue's measurement picks out as the best AVAILABLE pair: blind N-best at k=10
# (which by itself is the whole blind-arms union, 11) followed by the per-frame known-cell arm
# (+5 more, for 16). Stopping at the first blind arm forfeits 5 of the 16.
DEFAULT_NBEST = 10


def arm_blind_top1(q, gate, index_blind_fast):
    """Single blind index; accept if it clears the gate. (10 of 42 -- the weakest arm, and a strict
    subset of the k=10 N-best arm on the measured set, so the cascade does not run it.)"""
    try:
        M = index_blind_fast(q)
    except Exception:
        return None
    if M is None:
        return None
    M = np.asarray(M, float)
    return M if gate(M) else None


def arm_blind_nbest(q, gate, index_blind_nbest, nbest=DEFAULT_NBEST, candidates=None):
    """Blind N-best; accept the FIRST of the N candidates that clears the gate.

    `candidates` lets a caller that has already paid for the blind index (a fan-out over many frames,
    say) pass the [(cell, score), ...] list in instead of having it recomputed here.
    """
    try:
        nb = index_blind_nbest(q, nbest) if candidates is None else candidates
        for c, _s in nb:
            if c is None:
                continue
            c = np.asarray(c, float)
            if gate(c):
                return c                 # break INSIDE the test -- see the module docstring
    except Exception:
        pass
    return None


def arm_known_perframe(q, Mc, gate, index_known_gpu_cell):
    """Per-frame known-cell registration against Mc. NOT the batched indexer -- see the docstring."""
    try:
        M = index_known_gpu_cell(q, Mc)
    except Exception:
        return None
    if M is None:
        return None
    M = np.asarray(M, float)
    return M if gate(M) else None


# The escalation arm's defaults: T2 of the escalation experiment and its k = 32 null (exp/joint-ceiling,
# RESULTS_escalation.md and RESULTS_escalation_k32.md).
DEEP_TOPA = 128
DEEP_NC = 32
DEEP_K_NULL = 32
DEEP_SEED = 20260926


def arm_known_deep(q, Mc, index_known_gpu_cell, count, gate, seed, topa=DEEP_TOPA, nc=DEEP_NC,
                   k_null=DEEP_K_NULL, scramble=None):
    """Deep known-cell registration, accepted only if it beats every one of its own scrambled copies.

    A deeper search registers more of the frames the shipped search misses, but about half of what it
    adds is chance: on lattice-free (azimuth-scrambled) peak lists the observable gate passes 14 % of
    the time at this depth. So each fit is judged against its own frame. The frame's peaks are
    scrambled in azimuth k_null times (|q| and q_z, hence each peak's excitation error, kept exactly;
    lattice coherence destroyed); each copy is searched the same way. The fit is accepted only if no
    copy matches as many peaks, i.e. at p = 1/(k_null + 1).

    The null is SEQUENTIAL: it stops at the first copy that ties or beats the fit, and a fit that fails
    the gate costs one search and no null. On the 480-frame cxidb-17 set (job 39181473) this cost 7.6
    searches per missed frame instead of 33; it kept 21 of the 23 frames the k = 8 filter kept, and
    about 0.9 of those 21 are expected from chance.

    count(M, q) -> matched peaks (the strict matcher); gate(M, q) -> bool, the acceptance gate the caller
    uses (count and fraction, plus same lattice as Mc). seed: an entropy list, e.g. [DEEP_SEED, frame
    index]. Copy k is scrambled with np.random.default_rng([*seed, k]), which is the seeding of the
    experiment, so its accepts reproduce.

    Returns (M or None, record); the record holds m, n, searches, the null matched counts computed, and
    p = 1/(k_null + 1) when accepted.
    """
    if scramble is None:
        from glint.multilattice import scramble_azimuth as scramble
    q = np.asarray(q, float)
    rec = dict(m=0, n=int(len(q)), searches=1, null_m=[], p=None, accepted=False)
    try:
        M = index_known_gpu_cell(q, Mc, topa=topa, nc=nc)
    except Exception:
        return None, rec
    if M is None:
        return None, rec
    M = np.asarray(M, float)
    m = int(count(M, q))
    rec["m"] = m
    if not gate(M, q):
        return None, rec
    for k in range(int(k_null)):
        qs = scramble(q, np.random.default_rng([*seed, k]))
        rec["searches"] += 1
        try:
            Ms = index_known_gpu_cell(qs, Mc, topa=topa, nc=nc)
        except Exception:
            Ms = None
        ms = int(count(np.asarray(Ms, float), qs)) if Ms is not None else 0
        rec["null_m"].append(ms)
        if ms >= m:
            return None, rec                         # a lattice-free copy does as well: not evidence
    rec.update(accepted=True, p=1.0 / (int(k_null) + 1))
    return M, rec


# There is deliberately no `cascade(...)` convenience wrapper here composing the two arms. The
# ORDER is the finding (blind N-best k=10, then per-frame known-cell), but the composition is not
# reusable: StreamDriver has to try each candidate against every ACTIVE cell and remember which one
# accepted, so it can integrate the frame into that cell's accumulator. A wrapper would only serve
# the single-cell case and would sit uncalled and untested. The arms are the shared unit.
