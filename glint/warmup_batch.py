"""Parallel, peak-triaged blind warm-up for the streaming driver.

The blind path is ~100x the known-cell path, and consensus needs a handful of frames to lock, so
at a high source rate the SERIAL warm-up (one blind index at a time) stretches both the lock latency
and the raw-frame buffer that piles up meanwhile. This module makes the warm-up

  (1) PICKY  -- ranking the startup stack by Bragg-peak COUNT sends blind compute to the most
                indexable frames first (and skips blanks / water); and
  (2) PARALLEL -- the independent blind indexes fan across workers/GPUs and pool into ONE consensus
                round, so lock latency falls with the number of workers instead of costing n
                blind-frame-times.

MEASURED, because this file used to claim the floor was "~one blind-frame-time" and it is not.
S3DF job 37198405, one node, 4x A100, 32 triaged cxidb-17 frames, nbest=3, mpirun (OpenMPI here is
not built with SLURM PMI, so `srun` direct-launch aborts in MPI_Init):

    ranks   serial      fanout     speedup    fanout / one-frame-time
      1     737.7 ms    735.4 ms    1.00x       36.4
      2     728.2 ms    397.9 ms    1.83x       19.4
      4     737.7 ms    235.5 ms    3.13x       11.5

Answer-preserving at every rank count: identical cell [37.8 78.6 78.7] and identical support 28,
serial and fanned out.

What the numbers say. Subtracting the pure blind work (frames/ranks x ~20.4 ms) leaves a SERIAL
REMAINDER of ~76 ms that does not shrink with ranks (89 / 68 / 72 ms at 1 / 2 / 4) -- the pooling
and the consensus vote over frames*nbest hypotheses, plus the allgather. So the Amdahl floor is
~76 + ~20 = ~96 ms, about 4.8 blind-frame-times, and the ceiling on speedup is ~7.6x however many
workers are added. Efficiency is already falling at four ranks (91% at 2, 78% at 4).

Consequence for anyone sizing this: past roughly eight workers host work, not the blind indexes,
becomes the dominant term. More GPUs can still reduce latency toward the ~96 ms floor, but with
diminishing returns because the remaining cost is the same_lattice vote on the host, which is a
CPU-side algorithmic target -- and unlike the warp-per-candidate mapping in fused_kernels,
speeding it up need not cost bit-exactness.

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


def warmup_consensus(qs, blind_index, rc, nbest=3, fanout=None, sink=None):
    """Fan the independent blind indexes across workers, pool their N-best into ONE consensus round.

    qs          : list of per-frame reciprocal-vector arrays (the triaged picks).
    blind_index : callable(q, nbest) -> [(cell, score), ...]   (GLINT's index_blind_nbest).
    rc          : a RunningConsensus  (add_frame([cells]); verdict() -> (Mc, support, lead)).
    nbest       : keep this many candidate cells per frame for the vote.
    fanout      : callable(list_q, nbest) -> list of N-best lists, one per q, dispatched across
                  GPUs/workers. Default = serial single-GPU loop (the current behaviour), so nothing
                  breaks without a multi-GPU transport.
    sink        : optional list; if given, receives (q, [cells]) per indexed frame. The alias gate
                  needs the leader's lattice AS ORIENTED ON EACH FRAME, and each frame's own N-best
                  already carries it -- collecting it here costs nothing and saves re-indexing.

    Returns (Mc, support). Mc is None if the batch did not reach consensus -- widen `topk` or pull
    more of the buffer and call again (the vote accumulates across calls).
    """
    if not qs:
        return (None, 0)
    fan = fanout or (lambda Q, k: [blind_index(q, k) for q in Q])
    nbests = fan(qs, nbest)
    for q, nb in zip(qs, nbests):
        if nb:
            cells = [c for c, _ in nb if c is not None]
            rc.add_frame(cells)
            if sink is not None:
                sink.append((q, cells))
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
