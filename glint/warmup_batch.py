"""Parallel, peak-triaged blind warm-up for the streaming driver.

The blind path is ~100x the known-cell path, and consensus needs a handful of frames to lock, so
at a high source rate the SERIAL warm-up (one blind index at a time) stretches both the lock latency
and the raw-frame buffer that piles up meanwhile. This module makes the warm-up

  (1) PICKY  -- ranking the startup stack by Bragg-peak COUNT sends blind compute to the most
                indexable frames first (and skips blanks / water); and
  (2) PARALLEL -- the independent blind indexes fan across workers/GPUs and pool into ONE consensus
                round, so lock latency is ~one blind-frame-time instead of n of them.

The buffered frames not chosen for warm-up are not lost: after the cell locks they drain through the
fast known-cell path, so triage sets only the ORDER of blind attempts, not which frames are kept.

Ranking by a TEMPORAL MAD z-count (per-pixel median over the batch) was tried and rejected: it
separated hits from blanks on SIMULATED stacks, but on a liquid jet the score is swamped by
shot-varying water/jet scatter (see StreamDriver.warmup_batch for the cxic0415 calibration). The ring
peakfinder's spatial background is jet-robust and is the integration front end anyway, so its peak
count is both cheaper and better separated. Should the temporal route ever be revisited, the deleted
mad_triage() is in history: `git log -S'def mad_triage' -- glint/warmup_batch.py` (last on main in
caa254e).

Everything here is pure numpy and transport-agnostic (the multi-GPU fan-out is an injected callable),
so it is unit-testable on CPU with no GPU and no driver.
"""
import numpy as np


def triage_order(score, topk, floor=1):
    """Indices of the most promising events (descending score), at or above `floor`, capped at `topk`."""
    score = np.asarray(score)
    order = np.argsort(-score, kind="stable")
    return [int(i) for i in order if score[i] >= floor][:topk]


def warmup_consensus(qs, blind_index, rc, nbest=3, fanout=None):
    """Fan the independent blind indexes across workers, pool their N-best into ONE consensus round.

    qs          : list of per-frame reciprocal-vector arrays (the triaged picks).
    blind_index : callable(q, nbest) -> [(cell, score), ...]   (GLINT's index_blind_nbest).
    rc          : a RunningConsensus  (add_frame([cells]); verdict() -> (Mc, support, lead)).
    nbest       : keep this many candidate cells per frame for the vote.
    fanout      : callable(list_q, nbest) -> list of N-best lists, one per q, dispatched across
                  GPUs/workers. Default = serial single-GPU loop (the current behaviour), so nothing
                  breaks without a multi-GPU transport.

    Returns (Mc, support). Mc is None if the batch did not reach consensus -- widen `topk` or pull
    more of the buffer and call again (the vote accumulates across calls).
    """
    if not qs:
        return (None, 0)
    fan = fanout or (lambda Q, k: [blind_index(q, k) for q in Q])
    nbests = fan(qs, nbest)
    for nb in nbests:
        if nb:
            rc.add_frame([c for c, _ in nb])
    Mc, sup, _ = rc.verdict()
    return (Mc, sup)


def mpi_fanout(qs, nbest, blind_index):
    """Reference multi-GPU fan-out: one MPI rank per GPU indexes a slice, then allgather the N-best.

    Needs mpi4py and one visible GPU per rank (e.g. `srun --gpus-per-task=1`); not exercised by the
    CPU tests. Wire it as the `fanout` of warmup_consensus:

        from glint.glint_fast import index_blind_nbest
        warmup_consensus(qs, index_blind_nbest, rc, nbest,
                         fanout=lambda Q, k: mpi_fanout(Q, k, index_blind_nbest))

    The blind indexes are independent, so this is a single scatter of the work + one allgather; the
    only barrier is the consensus vote the caller runs on the reassembled result.
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    mine = [blind_index(qs[j], nbest) for j in range(rank, len(qs), size)]   # this rank's GPU, its slice
    gathered = comm.allgather(mine)
    out = [None] * len(qs)
    for r, chunk in enumerate(gathered):
        out[r::size] = chunk                                                 # reassemble original order
    return out
