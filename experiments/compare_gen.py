"""Candidate-generator family on the SAME 120 cxidb peaks, SAME assembler
(fftindex.search_basis) and SAME gate as compare3.py. These are runnable stand-ins for
the external programs (not installed on S3DF) keyed to their core principle:
  FFT volume        = asdf / cuFFT principle  (gridded 3D Fourier peak)
  difference-vector = DiRAX principle         (Patterson / pairwise-difference votes)
  projection (DPS)  = MOSFLM principle        (projection-slice 1D periodicity)
DIALS-FFT3D (the external cctbx/DIALS = labelit lineage) is gated separately by
compare_dials.py. xgandalf/ffbidx full indexers come from their driver output files.

  /sdf/group/lcls/ds/tools/cctbx-psana2/build/bin/dials.python compare_gen.py [N]
"""
import sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
import numpy as np
from glint.transform import fft_volume
from glint.peakfind import find_peaks_classical
from glint.seed import difference_vector_seeds, projection_axis_seeds
from glint.index import search_basis
from glint.multishot import same_lattice
from glint.lattice import cell_to_Ar

LYSO = np.asarray(cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90), float)
TOL = 0.15
EMPTY = np.zeros((0, 3))
FR = "/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt"


def load(p):
    fr = []; L = open(p).read().split("\n"); i = 0
    while i < len(L):
        if not L[i].startswith("FRAME"): i += 1; continue
        _, fid, n = L[i].split(); n = int(n)
        fr.append(np.array([[float(x) for x in L[i + 1 + j].split()] for j in range(n)])); i += 1 + n
    return fr


def valid(M, q):
    if M is None or not same_lattice(M, LYSO): return False
    H = q @ M; inl = (np.abs(H - np.rint(H)).max(1) < TOL).sum()
    return inl / len(q) >= 0.25 and inl >= 10


def gen_fft(q, qmax):
    vol, x = fft_volume(q, qmax, n=128)
    vecs, _ = find_peaks_classical(vol, x, min_len=3.0)
    return search_basis(vecs, EMPTY, q, 0.02 * qmax)


def gen_diff(q, qmax):
    seeds = difference_vector_seeds(q, qmax)
    if len(seeds) < 3: return None
    return search_basis(EMPTY, seeds, q, 0.02 * qmax)


def gen_dps(q, qmax):
    vecs = projection_axis_seeds(q)
    if len(vecs) < 3: return None
    return search_basis(vecs, EMPTY, q, 0.02 * qmax)


GENS = [("FFT (asdf-principle)", gen_fft),
        ("diff-vector (DiRAX-principle)", gen_diff),
        ("projection/DPS (MOSFLM-principle)", gen_dps)]

frames = [q for q in load(FR) if len(q) >= 6]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
frames = frames[:N]; n = len(frames)
print("generator-family, same %d peaks / common assembler+gate (TOL=0.15, frac>=.25, >=10 inl)" % n)
for name, fn in GENS:
    ok = 0; ms = 0.0
    for q in frames:
        qmax = float(np.linalg.norm(q, axis=1).max())
        t0 = time.time(); M = None
        try:
            res = fn(q, qmax)
            if res is not None: M = res[0]
        except Exception:
            pass
        ms += (time.time() - t0) * 1000.0
        if valid(M, q): ok += 1
    print("  %-34s %3d/%d = %2.0f%%   %.1f ms/frame" % (name, ok, n, 100.0 * ok / n, ms / n))
