"""The single indexing volume: 3D FFT of the reciprocal point cloud.

We deposit the observed spots g_i onto a Cartesian grid in reciprocal space
(with a small Gaussian footprint to control aliasing), FFT to the conjugate
real-space variable x, and return |F(x)| -- whose peaks are direct-lattice
vectors (columns of M and their integer combinations).

central_rays() reads radial lines out of this *same* volume: by the
projection-slice theorem that is exactly the mosflm/DPS 1D-FFT-over-directions
method, so the 1D baseline costs us nothing extra.
"""

from __future__ import annotations

import numpy as np

from .backend import array_module, fft_mag_centered, scatter_add


def estimate_grid_n(g, qmax, margin=1.6, n_min=128, n_max=256):
    """Auto-size the FFT grid so the longest cell axis fits on the volume.

    The real-space grid spans |x| <= n/(4 qmax); a cell axis longer than that falls
    off and is unindexable. The longest real axis ~ 1/(shortest reciprocal spacing).
    We estimate that spacing as the few-th smallest pairwise spot difference (the
    minimum can miss the longest axis when no adjacent pair along it is observed, so
    `margin` provides headroom; using the 3rd-smallest drops a stray spurious pair).
    Clamped to [n_min, n_max]; only compute, never correctness, is lost by going big.
    """
    g = np.asarray(g, float)
    if len(g) < 4:
        return n_min
    iu, ju = np.triu_indices(len(g), 1)
    d = np.linalg.norm(g[iu] - g[ju], axis=1)
    d = d[d > 1e-9]
    if len(d) < 4:
        return n_min
    spacing = np.partition(d, 2)[2]              # 3rd-smallest difference
    n = int(np.ceil(4 * qmax * margin / spacing))   # = 4 qmax margin Lmax
    n = int(np.clip(n, n_min, n_max))
    return n + (n % 2)                            # keep even


def fft_volume(g, qmax, n=128, gpu=False):
    """Return (vol, xcoords).

    vol[i,j,k] = |F(x)| with x = (xcoords[i], xcoords[j], xcoords[k]) in Angstrom.

    Spots are deposited with trilinear (cloud-in-cell) interpolation rather than
    Gaussian smoothing: a Gaussian footprint of width sigma cells multiplies F(x)
    by exp(-2 pi^2 sigma_q^2 |x|^2), which exponentially crushes the *long* real-
    space axes we most need. CIC adds essentially no envelope.

    The deposit is a single scatter-add over all spots x 8 trilinear corners (no
    Python loop), so it vectorizes on CPU and runs unchanged on the GPU when
    gpu=True (cupy + cuFFT). With gpu=True, vol and xcoords come back as cupy
    arrays so find_peaks can keep the n^3 maximum_filter on-device; gpu=False (or
    no cupy) returns numpy and is bit-for-bit the old result.
    """
    xp, _ = array_module(gpu)
    g = xp.asarray(g, dtype=xp.float64)
    dq = 2 * qmax / n
    rho = xp.zeros((n, n, n), dtype=xp.float64)

    f = (g + qmax) / dq                       # fractional grid coordinates
    i0 = xp.floor(f).astype(xp.int64)
    frac = f - i0
    ii, jj, kk, ww = [], [], [], []
    for di in (0, 1):                         # 8 corners, each vectorized over spots
        for dj in (0, 1):
            for dk in (0, 1):
                w = (
                    (frac[:, 0] if di else 1 - frac[:, 0])
                    * (frac[:, 1] if dj else 1 - frac[:, 1])
                    * (frac[:, 2] if dk else 1 - frac[:, 2])
                )
                ii.append(i0[:, 0] + di)
                jj.append(i0[:, 1] + dj)
                kk.append(i0[:, 2] + dk)
                ww.append(w)
    ii, jj = xp.concatenate(ii), xp.concatenate(jj)
    kk, ww = xp.concatenate(kk), xp.concatenate(ww)
    m = (ii >= 0) & (ii < n) & (jj >= 0) & (jj < n) & (kk >= 0) & (kk < n)
    scatter_add(rho, (ii[m], jj[m], kk[m]), ww[m])

    vol = fft_mag_centered(rho)

    # conjugate real-space coordinates (Angstrom)
    x = xp.fft.fftshift(xp.fft.fftfreq(n, d=dq))
    return vol, x


def fft_volume_batch(gs, qmaxs, n=128, gpu=True, dtype="float32"):
    """Batched indexing volume for B shots at once: (vols, xcoords_list).

    vols[b] = |F_b(x)| on the same n^3 grid (one batched cuFFT over a (B,n,n,n)
    stack), each shot keeping its own qmax via a per-spot dq. All shots' spots are
    deposited in a SINGLE scatter-add (batch index = which shot), so the only
    Python loop is the fixed 8 trilinear corners. This amortizes the per-launch /
    transfer overhead that makes the single-shot GPU path latency-bound -> the A100
    fills up and throughput (shots/sec) climbs far past the per-shot speedup.

    dtype float32 (complex64 FFT) is the throughput default: half the memory of
    float64 and much faster on the A100; indexing peaks are robust to it. Pass
    dtype='float64' to match the single-shot path bit-closely.
    """
    xp, _ = array_module(gpu)
    real_t = xp.float32 if dtype == "float32" else xp.float64
    B = len(gs)
    counts = np.array([len(g) for g in gs])
    g_all = xp.asarray(np.concatenate([np.asarray(g, float) for g in gs]), dtype=real_t)
    batch = xp.asarray(np.repeat(np.arange(B), counts), dtype=xp.int64)
    qm = xp.asarray(np.repeat(np.asarray(qmaxs, float), counts), dtype=real_t)
    dq = 2 * qm / n
    f = (g_all + qm[:, None]) / dq[:, None]          # (Ntot,3) fractional coords
    i0 = xp.floor(f).astype(xp.int64)
    frac = f - i0

    rho = xp.zeros((B, n, n, n), dtype=real_t)
    bb, ii, jj, kk, ww = [], [], [], [], []
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                w = (
                    (frac[:, 0] if di else 1 - frac[:, 0])
                    * (frac[:, 1] if dj else 1 - frac[:, 1])
                    * (frac[:, 2] if dk else 1 - frac[:, 2])
                )
                bb.append(batch)
                ii.append(i0[:, 0] + di)
                jj.append(i0[:, 1] + dj)
                kk.append(i0[:, 2] + dk)
                ww.append(w)
    bb, ii, jj = xp.concatenate(bb), xp.concatenate(ii), xp.concatenate(jj)
    kk, ww = xp.concatenate(kk), xp.concatenate(ww)
    m = (ii >= 0) & (ii < n) & (jj >= 0) & (jj < n) & (kk >= 0) & (kk < n)
    scatter_add(rho, (bb[m], ii[m], jj[m], kk[m]), ww[m])

    ax = (1, 2, 3)                                    # FFT the spatial axes, not batch
    F = xp.fft.fftshift(xp.fft.fftn(xp.fft.ifftshift(rho, axes=ax), axes=ax), axes=ax)
    vols = xp.abs(F)
    xs = [xp.fft.fftshift(xp.fft.fftfreq(n, d=2 * float(q) / n)) for q in qmaxs]
    return vols, xs


def central_rays(g, qmax, directions, n=512):
    """1D baseline via the projection-slice theorem.

    For each unit direction u, project the cloud onto u and 1D-FFT it; the
    transform equals the central ray x = s*u of the 3D volume. Returns a list
    of (s, |amp|) 1D spectra, one per direction.
    """
    g = np.asarray(g, float)
    out = []
    edges = np.linspace(-qmax, qmax, n + 1)
    dt = edges[1] - edges[0]
    for u in np.asarray(directions, float):
        u = u / np.linalg.norm(u)
        t = g @ u
        hist, _ = np.histogram(t, bins=edges)
        amp = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(hist.astype(float)))))
        s = np.fft.fftshift(np.fft.fftfreq(n, d=dt))
        out.append((s, amp))
    return out
