"""Original LBL labelit/phenix DPS (rstbx 1D-FFT projection) on the SAME 120 sparse cxidb frames,
SAME peaks, SAME downstream assembler (glint.search_basis) and SAME gate as compare_dials.py --
the labelit row for Table 2. NOT a basis-vector search (it's Steller-Rossmann-Sauter direction
periodograms); fed the identical peaks so the front end (DPS vs FFT3D) is what varies.
Run: /sdf/group/lcls/ds/tools/cctbx/ccp4-9/bin/dials.python labelit_sparse.py [N]"""
import sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
import numpy as np
from scitbx.array_family import flex
from rstbx.indexing_api.lattice import DPS_primitive_lattice
import iotbx.phil
from rstbx.phil.phil_preferences import indexing_api_defs
from glint.index import search_basis
from glint.multishot import same_lattice
from glint.lattice import cell_to_Ar

LYSO = np.asarray(cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90), float)
TOL = 0.15
PHIL = iotbx.phil.parse(input_string=indexing_api_defs).extract()
FR = "/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt"


def load(p):
    fr = []; L = open(p).read().split("\n"); i = 0
    while i < len(L):
        if not L[i].startswith("FRAME"):
            i += 1; continue
        _, fid, n = L[i].split(); n = int(n)
        fr.append(np.array([[float(x) for x in L[i + 1 + j].split()] for j in range(n)])); i += 1 + n
    return fr


def to_flex(q):
    return flex.vec3_double([(float(a), float(b), float(c)) for a, b, c in q])


def sol_vec(s):
    d = s.dvec                              # unit real-space direction
    return [d[0] * s.real, d[1] * s.real, d[2] * s.real]


def dps_vecs(q):
    Lt = DPS_primitive_lattice(max_cell=100.0, recommended_grid_sampling_rad=None, horizon_phil=PHIL)
    Lt.index(reciprocal_space_vectors=to_flex(q))
    return np.array([sol_vec(s) for s in Lt.getSolutions()], float)


def valid(M, q):
    if M is None or not same_lattice(M, LYSO):
        return False
    H = q @ M; inl = (np.abs(H - np.rint(H)).max(1) < TOL).sum()
    return inl / len(q) >= 0.25 and inl >= 10


frames = [q for q in load(FR) if len(q) >= 6]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
frames = frames[:N]

try:                                        # warmup + one-time API introspection
    Lt = DPS_primitive_lattice(max_cell=100.0, recommended_grid_sampling_rad=None, horizon_phil=PHIL)
    Lt.index(reciprocal_space_vectors=to_flex(frames[0]))
    sols = list(Lt.getSolutions())
    print("warmup: %d DPS solutions; solution attrs: %s" %
          (len(sols), [a for a in dir(sols[0]) if not a.startswith("_")][:14]))
    _ = dps_vecs(frames[0])
    print("dps_vecs OK, shape", _.shape)
except Exception as e:
    print("WARMUP ERR:", repr(e)[:300])

ok = 0; ms = 0.0
for q in frames:
    t0 = time.time(); M = None
    try:
        vecs = dps_vecs(q)
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
print("labelit/DPS blind (same peaks, common assembler+gate): %d/%d = %.0f%%, %.1f ms/frame"
      % (ok, n, 100.0 * ok / n, ms / n))
