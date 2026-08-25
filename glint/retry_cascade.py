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


# There is deliberately no `cascade(...)` convenience wrapper here composing the two arms. The
# ORDER is the finding (blind N-best k=10, then per-frame known-cell), but the composition is not
# reusable: StreamDriver has to try each candidate against every ACTIVE cell and remember which one
# accepted, so it can integrate the frame into that cell's accumulator. A wrapper would only serve
# the single-cell case and would sit uncalled and untested. The arms are the shared unit.
