"""DIALS/phenix DPS across a range of cells (corroboration). Run with cctbx dials.python (Py2)."""
import sys, time
import numpy as np
from scitbx.array_family import flex
from rstbx.indexing_api.lattice import DPS_primitive_lattice
import iotbx.phil
from rstbx.phil.phil_preferences import indexing_api_defs
PHIL = iotbx.phil.parse(input_string=indexing_api_defs).extract()

D = np.load("/pscratch/sd/s/smarches/glint_real/cells.npz")
names = [str(x) for x in D["names"]]; REPS = int(D["reps"]); MAXCELL = 200.0


def to_flex(q):
    return flex.vec3_double([(float(a), float(b), float(c)) for a, b, c in q])


def dps_lens(q):
    L = DPS_primitive_lattice(max_cell=MAXCELL, recommended_grid_sampling_rad=None, horizon_phil=PHIL)
    L.index(reciprocal_space_vectors=to_flex(q))
    return sorted([s.real for s in L.getSolutions()])


def found(lens, axes):                                     # all 3 truth axes among candidate lengths
    return all(any(abs(l - ax) <= 0.05 * ax for l in lens) for ax in axes)


try:
    dps_lens(D["%s_0" % names[0]])
except Exception as e:
    print("warmup:", repr(e)[:150])
print("DIALS/phenix DPS (CPU)")
print("%-11s%8s%7s%9s" % ("cell", "n_rlps", "rate", "ms"))
for n in names:
    axes = [float(x) for x in D["%s_axes" % n]]; ts, oks, ns = [], 0, []
    for r in range(REPS):
        q = D["%s_%d" % (n, r)]; ns.append(len(q))
        t0 = time.time()
        try:
            okk = found(dps_lens(q), axes)
        except Exception:
            okk = False
        ts.append(time.time() - t0); oks += okk
    print("  %-9s%8d%6d%%%9.1f" % (n, int(np.median(ns)), 100 * oks // REPS, 1e3 * np.median(ts)))
