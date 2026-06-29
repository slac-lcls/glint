"""DIALS FFT3D basis-vector search (cctbx/DIALS = modern labelit lineage) on the SAME
120 cxidb frames, SAME peaks, SAME downstream assembler (fftindex.search_basis) and SAME
gate as compare3.py [same_lattice(LYSO) AND frac>=.25 AND >=10 inliers, TOL=0.15].

Run under CCP4-9 dials.python:
  /sdf/group/lcls/ds/tools/cctbx/ccp4-9/bin/dials.python compare_dials.py [N]
"""
import sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex")
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex/experiments")
import numpy as np
from scitbx.array_family import flex
from dials.algorithms.indexing.basis_vector_search import FFT3D
from fftindex.index import search_basis
from fftindex.multishot import same_lattice
from fftindex.lattice import cell_to_Ar

LYSO = np.asarray(cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90), float)
TOL = 0.15
FR = "/sdf/home/s/smarches/git/fftindex/experiments/frames_cxidb_clean.txt"


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


frames = [q for q in load(FR) if len(q) >= 6]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
frames = frames[:N]
fft = FFT3D(max_cell=100.0, min_cell=3.0)
ok = 0; ms = 0.0
for q in frames:
    t0 = time.time(); M = None
    try:
        rlp = flex.vec3_double(q.copy())
        out = fft.find_basis_vectors(rlp)
        vecs = np.array([list(v) for v in out[0]], float)
        if len(vecs) >= 3:
            qmax = float(np.linalg.norm(q, axis=1).max())
            res = search_basis(vecs, np.zeros((0, 3)), q, 0.02 * qmax)
            if res is not None:
                M = res[0]
    except Exception:
        pass
    ms += (time.time() - t0) * 1000.0
    if valid(M, q):
        ok += 1
n = len(frames)
print("DIALS-FFT3D blind (same peaks, common assembler+gate): %d/%d = %.0f%%, %.1f ms/frame"
      % (ok, n, 100.0 * ok / n, ms / n))
