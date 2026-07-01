"""DIALS/phenix side of the head-to-head (run with the cctbx dials.python, Py2). Index the SAME shared
clouds with the labelit/phenix DPS 1D-FFT engine (rstbx) -> rate + wall-time, to compare with GLINT's
cluster-FFT (GPU). cctbx is CPU; this is the apples-to-apples FFT-indexing comparison on dense data.

  /global/cfs/cdirs/lcls/cctbx/build/bin/dials.python dials_clouds.py
"""
import sys, time
import numpy as np
from scitbx.array_family import flex

D = np.load("/pscratch/sd/s/smarches/glint_real/clouds.npz")
CELL = sorted([float(x) for x in D["cell"]]); REPS = int(D["reps"])
regimes = [str(x) for x in D["regimes"]]
MAXCELL = 150.0


def to_flex(q):
    return flex.vec3_double([(float(a), float(b), float(c)) for a, b, c in q])


# labelit/phenix DPS primitive-lattice engine
from rstbx.indexing_api.lattice import DPS_primitive_lattice
import iotbx.phil
from rstbx.phil.phil_preferences import indexing_api_defs
PHIL = iotbx.phil.parse(input_string=indexing_api_defs).extract()


def dps_lengths(q):
    L = DPS_primitive_lattice(max_cell=MAXCELL, recommended_grid_sampling_rad=None, horizon_phil=PHIL)
    L.index(reciprocal_space_vectors=to_flex(q))           # the DPS 1D-FFT work (timed)
    return sorted([s.real for s in L.getSolutions()])      # candidate lattice-vector lengths (A)


def found(lens):                                           # cell axes 38 & 79 both among candidates
    return (any(abs(l - 38.0) <= 0.05 * 38 for l in lens) and
            any(abs(l - 79.0) <= 0.05 * 79 for l in lens))


print("DIALS/phenix DPS (rstbx 1D-FFT, CPU)   cell %s" % CELL)
print("%-10s%8s%9s%10s" % ("regime", "n_rlps", "found", "ms"))
try:
    dps_lengths(D["%s_0" % regimes[0]])                    # warmup
except Exception as e:
    print("WARMUP err:", repr(e)[:200])
for g in regimes:
    ts, oks, ns = [], 0, []
    for r in range(REPS):
        q = D["%s_%d" % (g, r)]; ns.append(len(q))
        t0 = time.time()
        try:
            ok = found(dps_lengths(q))
        except Exception as e:
            ok = False
            if r == 0:
                print("  (%s err: %s)" % (g, repr(e)[:140]))
        ts.append(time.time() - t0); oks += ok
    print("  %-8s%8d%8d%%%10.1f" % (g, int(np.median(ns)), 100 * oks // REPS, 1e3 * np.median(ts)))
