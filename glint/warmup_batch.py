"""Parallel, MAD-triaged blind warm-up for the streaming driver.

The blind path is ~100x the known-cell path, and consensus needs a handful of frames to lock, so
at a high source rate the SERIAL warm-up (one blind index at a time) stretches both the lock latency
and the raw-frame buffer that piles up meanwhile. This module makes the warm-up

  (1) PICKY  -- a batched MAD triage over the startup stack ranks events by signal, so blind compute
                goes to the most indexable frames first (and blanks / water are skipped); and
  (2) PARALLEL -- the independent blind indexes fan across workers/GPUs and pool into ONE consensus
                round, so lock latency is ~one blind-frame-time instead of n of them.

The buffered frames not chosen for warm-up are not lost: after the cell locks they drain through the
fast known-cell path, so triage sets only the ORDER of blind attempts, not which frames are kept.

MAD (median-absolute-deviation) background is TEMPORAL -- the per-pixel median over the batch. This
is clean for SFX because random orientations put Bragg peaks at different pixels every shot, so the
median is a true common-background estimate and the peaks are the outliers. Scope it to warm-up
triage; the per-frame ring peakfinder (spatial background, sub-pixel, real I/sigma) stays the
integration front end -- temporal background breaks on shot-varying backgrounds and recurring peaks.

Everything here is pure (numpy or cupy via the `xp` arg) and transport-agnostic (the multi-GPU
fan-out is an injected callable), so it is unit-testable on CPU with no GPU and no driver.
"""
import numpy as np


def mad_triage(stack, z0=4.0, xp=np):
    """Batched MAD over a (B,H,W) stack -> (z, signal_mask, promise_score).

    z = (I - per-pixel median) / per-pixel MAD   -- a temporal-background SNR, batched over B.
    signal_mask = z > z0   (Holton's z-level; 4 is the reference default).
    promise_score[i] = number of signal pixels in event i -- an orientation-agnostic estimate that
    the event carries an indexable crystal (blanks / water score ~ the noise tail).
    Returned score is always host numpy, even when xp is cupy.
    """
    stack = xp.asarray(stack)
    if stack.ndim != 3:
        raise ValueError("stack must be (B, H, W)")
    med = xp.median(stack, axis=0, keepdims=True)                 # per-pixel temporal background
    mad = xp.median(xp.abs(stack - med), axis=0, keepdims=True)   # per-pixel noise
    mad = xp.where(mad > 0, mad, xp.asarray(1.0, stack.dtype if stack.dtype.kind == "f" else "float64"))
    z = (stack - med) / mad
    sig = z > z0
    score = sig.reshape(stack.shape[0], -1).sum(axis=1)
    score = score.get() if hasattr(score, "get") else np.asarray(score)
    return z, sig, score.astype(np.int64)


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
