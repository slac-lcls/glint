import sys, time
import numpy as np
from scitbx.array_family import flex
from dials.algorithms.indexing.basis_vector_search import FFT3D

def read_frames(path):
    lines = open(path).read().split("\n"); frames=[]; i=0
    while i < len(lines):
        if not lines[i].startswith("FRAME"): i+=1; continue
        _,fid,npk = lines[i].split(); npk=int(npk)
        q=np.array([[float(x) for x in lines[i+1+j].split()] for j in range(npk)])
        frames.append((int(fid), q)); i+=1+npk
    return frames

frames = read_frames(sys.argv[1])
fft = FFT3D(max_cell=100.0, min_cell=3.0)
out = open(sys.argv[2], "w")
for fid, q in frames:
    rlp = flex.vec3_double(q.copy())
    t0 = time.time()
    try:
        res = fft.find_basis_vectors(rlp); vecs = list(res[0])
    except Exception as e:
        vecs = []
    ms = (time.time()-t0)*1000.0
    out.write("FRAME %d %d %.2f\n" % (fid, len(vecs), ms))
    for v in vecs:
        out.write("%.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
out.close()
print("done", sys.argv[1])
