import sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
import numpy as np
from scitbx.array_family import flex
from dials.algorithms.indexing.basis_vector_search import FFT3D
from glint.index import search_basis
from glint.multishot import same_lattice
from glint.lattice import cell_to_Ar
LYSO = np.asarray(cell_to_Ar(79.02,79.02,37.98,90,90,90), float)
def read_frames(path):
    lines = open(path).read().split("\n"); frames=[]; i=0
    while i < len(lines):
        if not lines[i].startswith("FRAME"): i+=1; continue
        _,fid,npk = lines[i].split(); npk=int(npk)
        q=np.array([[float(x) for x in lines[i+1+j].split()] for j in range(npk)])
        frames.append((int(fid), q)); i+=1+npk
    return frames
frames = read_frames("/sdf/home/s/smarches/git/glint/experiments/xgandalf/frames.txt")
fft = FFT3D(max_cell=100.0, min_cell=3.0)
ok=0; tot_ms=0.0
for fid, q in frames:
    rlp = flex.vec3_double(q.copy())
    t0=time.time()
    try: out = fft.find_basis_vectors(rlp)
    except Exception: tot_ms += (time.time()-t0)*1000; continue
    tot_ms += (time.time()-t0)*1000
    vecs = np.array([list(v) for v in out[0]], float)
    if len(vecs) < 3: continue
    qmax = float(np.linalg.norm(q, axis=1).max())
    res = search_basis(vecs, np.zeros((0,3)), q, 0.02*qmax, n_real=12, min_inlier_frac=0.7)
    if res is not None and same_lattice(res[0], LYSO): ok += 1
print("DIALS fft3d (same spots, common assembler): indexed %d/%d = %.0f%%, %.1f ms/frame" % (
    ok, len(frames), 100.0*ok/len(frames), tot_ms/len(frames)))
