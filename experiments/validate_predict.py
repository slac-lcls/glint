"""Validate the spot-prediction bridge (fftindex.predict) against CrystFEL ground truth.

Ground truth = an indexed reflection list lifted from a real CrystFEL stream (the same one
calib_geom.py used): the jf16m geometry, a true orientation (astar/bstar/cstar), and 5 reflections
with their (h,k,l, fs, ss). Two checks:
  (1) PROJECTION is the exact inverse of lute_bridge.peaks_to_q: peaks_to_q(fs,ss)->q->project_q(q)
      must return (fs,ss). Independent of orientation; isolates the detector math.
  (2) ORIENTATION/EWALD: predicting q = hkl @ R for the true hkl must land on CrystFEL's (fs,ss)
      with small excitation error. Sweep clen to confirm the detector distance and the chain.

  source psconda.sh ; python validate_predict.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np
from glint.lute_bridge import parse_geom, peaks_to_q, lambda_from_eV
from glint.predict import project_q, predict_spots

GEOM = "/sdf/group/lcls/ds/tools/lute/test_utilities/jf16mgeom.geom"
astar = np.array([-0.0023779, 0.0600627, -0.4403932])   # nm^-1 (CrystFEL truth)
bstar = np.array([-0.0036161, -0.3527004, 0.0069991])
cstar = np.array([-0.2155720, 0.0110920, 0.0249098])
R = np.array([astar, bstar, cstar]) * 0.1               # -> 1/A, rows a*,b*,c*
lam = lambda_from_eV(14975.836726)
REFL = [(0, -5, -10, 573.0, 9126.5), (0, -5, -9, 622.7, 9128.2),
        (0, -4, -12, 476.2, 9042.8), (0, -4, -10, 574.1, 9046.5),
        (0, -4, -5, 818.2, 9055.9)]
hkl = np.array([[h, k, l] for h, k, l, _, _ in REFL])
fs_true = np.array([r[3] for r in REFL]); ss_true = np.array([r[4] for r in REFL])

panels, _ = parse_geom(GEOM)
print(f"geom: {len(panels)} panels   lambda={lam:.5f} A")

# ---- (2) sweep clen: q-match, excitation error, and predicted-vs-true (fs,ss) ----
print("\n(2) ORIENTATION/EWALD  (predict hkl @ R -> detector, vs CrystFEL fs,ss)")
print(f"  {'clen':>7}{'mean|dq|':>11}{'mean|exc|':>11}{'fs,ss err px':>15}")
best = None
for clen in [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15]:
    q_obs = peaks_to_q(fs_true, ss_true, panels, clen, lam)     # from CrystFEL fs,ss
    q_pred = hkl @ R                                            # from orientation
    dq = np.linalg.norm(q_pred - q_obs, axis=1).mean()
    exc = np.abs(q_pred[:, 2] + 0.5 * lam * np.einsum("ij,ij->i", q_pred, q_pred)).mean()
    fp, sp, pan = project_q(q_pred, panels, clen, lam)
    err = np.sqrt((fp - fs_true) ** 2 + (sp - ss_true) ** 2)
    me = np.nanmean(err)
    print(f"  {clen:7.2f}{dq:11.4f}{exc:11.4f}{me:15.3f}")
    if best is None or dq < best[1]:
        best = (clen, dq, me)
clen = best[0]
print(f"  -> best clen = {clen:.2f} m  (mean|dq|={best[1]:.4f} 1/A, fs/ss err={best[2]:.3f} px)")

# ---- (1) projection inverse round-trip at best clen ----
q_obs = peaks_to_q(fs_true, ss_true, panels, clen, lam)
fp, sp, pan = project_q(q_obs, panels, clen, lam)
rt = np.sqrt((fp - fs_true) ** 2 + (sp - ss_true) ** 2)
print(f"\n(1) PROJECTION inverse round-trip: max|fs,ss err| = {np.nanmax(rt):.2e} px  "
      f"(panels {pan.tolist()})")

# ---- full predict_spots sanity: how many spots, are the 5 truth hkl recovered ----
pred = predict_spots(R, panels, clen, lam, dmin=1.6, tol=0.02, is_recip=True)
got = {(p["h"], p["k"], p["l"]) for p in pred}
hit = sum((h, k, l) in got for h, k, l, _, _ in REFL)
print(f"\nfull predict_spots(dmin=1.6, tol=0.02): {len(pred)} on-detector reflections; "
      f"{hit}/5 truth hkl recovered; res range {pred['res'].min():.2f}-{pred['res'].max():.2f} A")
