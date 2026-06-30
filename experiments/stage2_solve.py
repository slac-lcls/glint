"""Stage 2 (solve): pin nanoBragg's lab frame vs the detector frame by hkl-integrality.

nanoBragg renders with reciprocal orientation A_recip (columns a*,b*,c*) in ITS lab frame; my
detector model (peaks_to_q: beam +z, panel at z=clen, fs/ss in xy) is a fixed rotation G away from
that frame. For the correct detector geom (right scale) and the correct G, every bright Bragg pixel
maps to a near-INTEGER hkl:  q_det = G @ A_recip @ hkl  =>  hkl = A_recip^-1 @ G^T @ q_det.
Search G over the 24 proper cube rotations; the winner has ~zero integer residual -> convention pinned.

  source psconda.sh ; conda activate ana-4.0.58-py3-minipytorch ; python stage2_solve.py
"""
import os, sys, itertools
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np
from scipy.ndimage import maximum_filter
from fftindex.lute_bridge import peaks_to_q

HOME = "/sdf/home/s/smarches"
img = np.load(f"{HOME}/sim_still0.npy")
A_recip = np.load(f"{HOME}/sim_still0_Arecip.npy")    # columns a*,b*,c* (1/A)
Ainv = np.linalg.inv(A_recip)
WAVE, CLEN = 1.32, 0.15


def signed_perms():
    """All 48 signed permutation matrices (proper AND improper -> catches a handedness flip)."""
    out = []
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3))
        for i, j in enumerate(perm):
            P[i, j] = 1
        for s in itertools.product((1, -1), repeat=3):
            out.append(P * np.array(s))
    return out


# matching single-panel geom (0.2mm px -> res 5000; beam at 512.5 -> corner -512.5; 1024^2)
panels = [dict(name="p0", fs=np.array([1., 0, 0]), ss=np.array([0, 1., 0]), res=5000.0,
               cx=-512.5, cy=-512.5, coffset=0.0, min_fs=0, max_fs=1023, min_ss=0, max_ss=1023)]

mx = maximum_filter(img, size=7)
ys, xs = np.where((img == mx) & (img > 0.10 * img.max()))
q_det = peaks_to_q(xs.astype(float), ys.astype(float), panels, CLEN, WAVE)   # (n,3), my frame
ok = ~np.isnan(q_det).any(1)
q_det = q_det[ok]
print(f"{len(q_det)} bright maxima; |q_det| range {np.linalg.norm(q_det,axis=1).min():.3f}-"
      f"{np.linalg.norm(q_det,axis=1).max():.3f} 1/A")

# nanoBragg's reciprocal matrix C (columns a*,b*,c*) is either A_recip or its transpose
# (the Amatrix transpose convention); q_det = G @ C @ hkl  =>  hkl = C^-1 @ G^T @ q_det
best = None
for cname, C in (("A", A_recip), ("A^T", A_recip.T)):
    Cinv = np.linalg.inv(C)
    for G in signed_perms():
        hkl = (Cinv @ G.T @ q_det.T).T
        resid = np.abs(hkl - np.round(hkl))
        score = np.median(resid.max(1))
        frac = (resid.max(1) < 0.15).mean()
        if best is None or score < best[0]:
            best = (score, frac, cname, G)
score, frac, cname, G = best
print(f"\nbest: C={cname}, G=\n{np.array2string(G.astype(int))}")
print(f"  median worst-axis hkl residual = {score:.4f}; {100*frac:.0f}% spots <0.15 of integer")
C = A_recip if cname == "A" else A_recip.T
hkl = (np.linalg.inv(C) @ G.T @ q_det.T).T
print(f"  example hkl (rounded): {np.round(hkl[:6]).astype(int).tolist()}")
print("PASS: convention pinned" if score < 0.1 else f"FAIL: residual {score:.3f} (scan scale next)")
