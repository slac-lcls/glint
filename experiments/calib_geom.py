"""Calibrate the detector->reciprocal conversion against CrystFEL's indexed stream.

For an indexed frame the stream gives the reciprocal basis (astar/bstar/cstar, nm^-1)
and each reflection's (h,k,l) AND (fs,ss). So CrystFEL's q = h*astar+k*bstar+l*cstar
is ground truth; our peaks_to_q(fs,ss) must reproduce it. Comparing pins the bug
(distance, axis swap, sign).
"""
import sys

sys.path.insert(0, "..")
import numpy as np

from glint.lute_bridge import lambda_from_eV, parse_geom, peaks_to_q

GEOM = "/sdf/group/lcls/ds/tools/lute/test_utilities/jf16mgeom.geom"
astar = np.array([-0.0023779, 0.0600627, -0.4403932])      # nm^-1
bstar = np.array([-0.0036161, -0.3527004, 0.0069991])
cstar = np.array([-0.2155720, 0.0110920, 0.0249098])
lam = lambda_from_eV(14975.836726)
REFL = [(0, -5, -10, 573.0, 9126.5), (0, -5, -9, 622.7, 9128.2),
        (0, -4, -12, 476.2, 9042.8), (0, -4, -10, 574.1, 9046.5),
        (0, -4, -5, 818.2, 9055.9)]

panels, _ = parse_geom(GEOM)
hkl = np.array([[h, k, l] for h, k, l, _, _ in REFL])
fs = np.array([r[3] for r in REFL])
ss = np.array([r[4] for r in REFL])
# CrystFEL truth (nm^-1 -> A^-1 = *0.1)
qc = (hkl @ np.array([astar, bstar, cstar])) * 0.1

print("CrystFEL q (A^-1):")
for i, (h, k, l, _, _) in enumerate(REFL):
    print(f"  ({h},{k},{l}): {np.round(qc[i],4)}  |q|={np.linalg.norm(qc[i]):.4f}")

for clen in [-0.05, 0.0, 0.05, 0.1]:                       # actual z = clen + coffset(0.1)
    qm = peaks_to_q(fs, ss, panels, clen, lam)
    err = np.linalg.norm(qm - qc, axis=1)
    print(f"\nclen={clen} (z={clen+0.1:.2f} m):  mean|q_mine-q_crystfel|={np.nanmean(err):.4f}")
    for i in range(len(REFL)):
        print(f"  mine {np.round(qm[i],4)} |{np.linalg.norm(qm[i]):.4f}|  vs cryst |{np.linalg.norm(qc[i]):.4f}|")
