"""Peakfinders on the 3D FFT volume.

This is the pluggable slot. v0 ships a classical local-maxima finder as the
baseline-to-beat. The learned detector (detector.LearnedPeakFinder) drops in here
behind the same `find_peaks(vol, xcoords, g, qmax) -> (vecs, amps)` interface --
g and qmax are passed because the learned features are computed from the spots,
not just the volume; the classical finder ignores them.
"""

from __future__ import annotations

import numpy as np

from .backend import asnumpy, maximum_filter, module_of


def find_peaks_classical(vol, xcoords, g=None, qmax=None, min_len=2.0,
                         max_peaks=400, rel_thresh=0.05):
    """Local maxima of |F(x)|, returned as candidate real-space vectors.

    vecs: (m,3) peak positions in Angstrom, sorted by descending amplitude.
    amps: (m,) peak amplitudes.
    Peaks with |x| < min_len (incl. the DC peak at the origin) are dropped.

    Backend-aware: when `vol` is a cupy array (gpu fft_volume), the n^3
    maximum_filter and reductions run on-device and only the small (m,3) peak
    list is copied back to host for the basis search.
    """
    xp = module_of(vol)
    mx = maximum_filter(vol, size=3)
    is_peak = (vol == mx) & (vol > rel_thresh * vol.max())
    ii = xp.argwhere(is_peak)
    if len(ii) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    vecs = xp.column_stack([xcoords[ii[:, 0]], xcoords[ii[:, 1]], xcoords[ii[:, 2]]])
    amps = vol[is_peak]
    lens = xp.linalg.norm(vecs, axis=1)
    keep = lens >= min_len
    vecs, amps = vecs[keep], amps[keep]
    order = xp.argsort(amps)[::-1][:max_peaks]
    return asnumpy(vecs[order]), asnumpy(amps[order])


def find_peaks_batch(vols, xs, min_len=2.0, max_peaks=128, rel_thresh=0.05):
    """Batched local-maxima peakfind on a (B,n,n,n) stack -> list of (vecs, amps).

    maximum_filter (size=(1,3,3,3), never crossing the batch axis) then a per-shot
    top-`max_peaks` selection, BOTH on the backend that owns `vols` (GPU for a cupy
    stack). The selection is an on-device argpartition over the flattened spatial
    axis, so only B*max_peaks amplitudes+indices cross to the host -- bounded and
    independent of how many local maxima a noisy |F(x)| has.

    (Tried a threshold-prefilter + argwhere instead, expecting to dodge the
    argpartition: it was SLOWER at every n. argwhere also scans all n^3, but adds a
    compaction + device sync + a variable, larger host transfer + a per-shot CPU
    grouping loop; argpartition keeps it all on-GPU with a fixed tiny transfer. The
    n=256 cost is just the 8x voxels -- inherent to the grid; the real lever there is
    a smaller grid (NUFFT crop to the basis-peak region), not the peakfind.)
    """
    xp = module_of(vols)
    B, n = vols.shape[0], vols.shape[1]
    mx = maximum_filter(vols, size=(1, 3, 3, 3))
    vmax = vols.reshape(B, -1).max(axis=1)                 # (B,) per-shot peak
    is_peak = (vols == mx) & (vols > rel_thresh * vmax[:, None, None, None])
    flat = xp.where(is_peak, vols, 0.0).reshape(B, -1)     # (B,n^3), non-peaks zeroed
    K = min(max_peaks, flat.shape[1])
    top = xp.argpartition(flat, flat.shape[1] - K, axis=1)[:, -K:]   # (B,K) flat idx
    amps = xp.take_along_axis(flat, top, axis=1)           # (B,K)
    ii, jj, kk = top // (n * n), (top // n) % n, top % n   # decode flat -> i,j,k

    ii, jj, kk, amps = asnumpy(ii), asnumpy(jj), asnumpy(kk), asnumpy(amps)
    xs = [asnumpy(x) for x in xs]
    out = []
    for b in range(B):
        a = amps[b]
        nz = a > 0                                         # fewer than K real peaks
        xb = xs[b]
        vecs = np.column_stack([xb[ii[b][nz]], xb[jj[b][nz]], xb[kk[b][nz]]])
        a = a[nz]
        keep = np.linalg.norm(vecs, axis=1) >= min_len
        vecs, a = vecs[keep], a[keep]
        order = np.argsort(a)[::-1]
        out.append((vecs[order], a[order]))
    return out


def find_peaks_batch_coarse(vols, xs, min_len=2.0, max_peaks=160, rel_thresh=0.05, pool=4):
    """Coarse-to-fine peakfind: instead of the full-n^3 top-K (the 76% bottleneck),
    max-pool by `pool` and select the top-K BLOCKS on the (n/pool)^3 coarse grid -- a
    ~pool^3-smaller argpartition -- then refine each winning block to its exact peak with
    a small batched argmax. The 3x3x3 max-filter mask is applied BEFORE pooling so the
    broad DC lobe contributes only its single local max (not a whole neighbourhood of
    high blocks). Falls back to dense find_peaks_batch when n isn't divisible by pool."""
    xp = module_of(vols)
    B, n = vols.shape[0], vols.shape[1]
    if n % pool:
        return find_peaks_batch(vols, xs, min_len=min_len, max_peaks=max_peaks)
    p, nc = pool, n // pool
    mx = maximum_filter(vols, size=(1, 3, 3, 3))
    vmax = vols.reshape(B, -1).max(axis=1)
    vp = xp.where((vols == mx) & (vols > rel_thresh * vmax[:, None, None, None]), vols, 0.0)
    vc = vp.reshape(B, nc, p, nc, p, nc, p).max(axis=(2, 4, 6))     # block max of local maxima
    vcf = vc.reshape(B, -1)
    K = min(max_peaks, vcf.shape[1])
    topc = xp.argpartition(vcf, vcf.shape[1] - K, axis=1)[:, -K:]   # (B,K) top coarse blocks
    Ic, Jc, Kc = topc // (nc * nc), (topc // nc) % nc, topc % nc
    oi, oj, ok = xp.meshgrid(xp.arange(p), xp.arange(p), xp.arange(p), indexing="ij")
    fi = Ic[..., None, None, None] * p + oi                         # (B,K,p,p,p) fine voxels
    fj = Jc[..., None, None, None] * p + oj
    fk = Kc[..., None, None, None] * p + ok
    bb = xp.arange(B)[:, None, None, None, None]
    blocks = vp[bb, fi, fj, fk].reshape(B, K, -1)                   # gather local-maxima volume
    am = blocks.argmax(axis=2)
    amps = blocks.max(axis=2)
    di, dj, dk = am // (p * p), (am // p) % p, am % p
    ii, jj, kk = Ic * p + di, Jc * p + dj, Kc * p + dk

    ii, jj, kk, amps = asnumpy(ii), asnumpy(jj), asnumpy(kk), asnumpy(amps)
    xs = [asnumpy(x) for x in xs]
    out = []
    for b in range(B):
        a = amps[b]
        nz = a > 0                                                 # blocks with a real local max
        xb = xs[b]
        vecs = np.column_stack([xb[ii[b][nz]], xb[jj[b][nz]], xb[kk[b][nz]]])
        a = a[nz]
        keep = np.linalg.norm(vecs, axis=1) >= min_len
        vecs, a = vecs[keep], a[keep]
        order = np.argsort(a)[::-1]
        out.append((vecs[order], a[order]))
    return out
