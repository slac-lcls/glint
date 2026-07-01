"""Difference-cloud GRIDDING as a candidate generator (user idea, 2026-06-26).

Wiener-Khinchin: the difference/autocorrelation cloud A(u) carries the same info as
|F|^2 but in the domain where the long DIRECT axis a (hard for the real-space FFT
grid: digitization err prop dq*x kills large x) becomes a SHORT reciprocal vector
a*=1/a (easy). A(u) has ~P^2 differences that PILE UP on the reciprocal lattice, so
each reciprocal basis vector gets multiplicity ~#pairs along it = SIGNAL that grows
with P. Hypothesis: grid the difference cloud on a FINE grid focused on the short-
vector region; when P is high the pile-up towers over the gridding/digitization
smear, so 3D gridding works (opposite scaling to the direct FFT).

This measures the candidate CEILING (does the pool contain a lyso cell at all) per
peak-count stratum for: FFT |F| peaks | DPS projection | difference-grid (new) |
combined. Run on Mac.
"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, "/Users/smarches/git/glint")
from glint.transform import fft_volume
from glint.peakfind import find_peaks_classical
from glint.seed import projection_axis_seeds
from glint.index import refine
from glint.lattice import cell_to_Ar
from glint.multishot import same_lattice

LYSO = cell_to_Ar(78.6, 78.6, 37.9, 90, 90, 90)
FR = "/tmp/frames_cxidb.txt"


def load(p):
    fr = []; f = open(p); l = f.readline()
    while l:
        t = l.split()
        if t and t[0] == "FRAME":
            n = int(t[2]); fr.append(np.array([list(map(float, f.readline().split())) for _ in range(n)]))
        l = f.readline()
    return fr


def difference_grid_seeds(g, qmax, n=128, frac=0.28, smooth=0.8, max_seeds=16):
    """Densest SHORT difference vectors = reciprocal basis candidates (a*,b*,c*,...).

    Grid the pairwise-difference cloud on a fine grid over |d| <= frac*qmax (the
    short-vector region where the reciprocal basis lives) with trilinear deposit, so
    the pile-up multiplicity at each reciprocal lattice vector forms a clean peak.
    Returns RECIPROCAL vectors."""
    g = np.asarray(g, float); P = len(g)
    if P < 4:
        return np.zeros((0, 3))
    iu, ju = np.triu_indices(P, 1)
    d = g[iu] - g[ju]; d = np.vstack([d, -d])
    dmax = frac * qmax
    d = d[np.linalg.norm(d, axis=1) <= dmax]
    if len(d) < 4:
        return np.zeros((0, 3))
    dq = 2 * dmax / n
    hist = np.zeros((n, n, n))
    f = (d + dmax) / dq
    i0 = np.floor(f).astype(np.int64); frac3 = f - i0
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                w = ((frac3[:, 0] if di else 1 - frac3[:, 0])
                     * (frac3[:, 1] if dj else 1 - frac3[:, 1])
                     * (frac3[:, 2] if dk else 1 - frac3[:, 2]))
                ii = i0[:, 0] + di; jj = i0[:, 1] + dj; kk = i0[:, 2] + dk
                m = (ii >= 0) & (ii < n) & (jj >= 0) & (jj < n) & (kk >= 0) & (kk < n)
                np.add.at(hist, (ii[m], jj[m], kk[m]), w[m])
    if smooth > 0:
        from scipy.ndimage import gaussian_filter
        hist = gaussian_filter(hist, smooth, mode="constant")
    coords = -dmax + (np.arange(n) + 0.5) * dq
    vecs, _ = find_peaks_classical(hist, coords, min_len=2.5 * dq)
    return vecs[:max_seeds]


def any_lyso_from_real(cands, q, tol):
    cands = np.asarray(cands, float).reshape(-1, 3)
    if len(cands) < 3:
        return False
    for tri in itertools.combinations(range(len(cands)), 3):
        M0 = cands[list(tri)].T
        sc = np.prod([np.linalg.norm(cands[t]) for t in tri])
        if sc <= 0 or abs(np.linalg.det(M0)) < 0.1 * sc:
            continue
        try:
            M, C, hkl, inl = refine(q, M0, tol)
        except np.linalg.LinAlgError:
            continue
        if C is not None and same_lattice(M, LYSO):
            return True
    return False


def any_lyso_from_recip(seeds, q, tol):
    seeds = np.asarray(seeds, float).reshape(-1, 3)
    if len(seeds) < 3:
        return False
    for tri in itertools.combinations(range(len(seeds)), 3):
        C0 = seeds[list(tri)].T
        sc = np.prod([np.linalg.norm(seeds[t]) for t in tri])
        if sc <= 0 or abs(np.linalg.det(C0)) < 0.1 * sc:
            continue
        try:
            M0 = np.linalg.inv(C0).T
            M, C, hkl, inl = refine(q, M0, tol)
        except np.linalg.LinAlgError:
            continue
        if C is not None and same_lattice(M, LYSO):
            return True
    return False


if __name__ == "__main__":
    frames = load(FR)
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else len(frames)
    frames = frames[:lim]
    strata = {"sparse(<70)": [], "mod(70-150)": [], "dense(>150)": []}
    for i, q in enumerate(frames):
        p = len(q); k = "sparse(<70)" if p < 70 else "mod(70-150)" if p <= 150 else "dense(>150)"
        strata[k].append(i)
    methods = ["FFT", "DPS", "DIFFGRID", "ANY"]
    counts = {k: {m: 0 for m in methods} for k in strata}
    t0 = time.time()
    for i, q in enumerate(frames):
        if len(q) < 6:
            continue
        qmax = float(np.linalg.norm(q, axis=1).max()); tol = 0.02 * qmax
        vol, x = fft_volume(q, qmax, n=256, gpu=False)
        fft_c, _ = find_peaks_classical(vol, x, q, qmax, min_len=3.0)
        fft_c = np.asarray(fft_c[:12], float)
        dps_c = projection_axis_seeds(q, topk=12) if len(q) <= 350 else np.zeros((0, 3))
        dg_c = difference_grid_seeds(q, qmax)
        cf = any_lyso_from_real(fft_c, q, tol)
        cd = any_lyso_from_real(dps_c, q, tol)
        cg = any_lyso_from_recip(dg_c, q, tol)
        k = "sparse(<70)" if len(q) < 70 else "mod(70-150)" if len(q) <= 150 else "dense(>150)"
        counts[k]["FFT"] += cf; counts[k]["DPS"] += cd; counts[k]["DIFFGRID"] += cg
        counts[k]["ANY"] += (cf or cd or cg)
        if (i + 1) % 30 == 0:
            print(f"  ..{i+1}/{len(frames)} {time.time()-t0:.0f}s", flush=True)
    print(f"\nCANDIDATE CEILING by stratum (frames whose pool contains a lyso cell):")
    print(f"{'stratum':14}{'n':>4}{'FFT':>8}{'DPS':>8}{'DIFFGRID':>10}{'ANY':>8}")
    tot = {m: 0 for m in methods}; ntot = 0
    for k, idxs in strata.items():
        nn = len([i for i in idxs if len(frames[i]) >= 6]); ntot += nn
        row = "".join(f"{counts[k][m]:>8}" if m != 'DIFFGRID' else f"{counts[k][m]:>10}" for m in methods)
        print(f"{k:14}{nn:>4}{row}")
        for m in methods: tot[m] += counts[k][m]
    print(f"{'TOTAL':14}{ntot:>4}{tot['FFT']:>8}{tot['DPS']:>8}{tot['DIFFGRID']:>10}{tot['ANY']:>8}")
    print(f"wall {time.time()-t0:.0f}s")
