"""Run fftindex on REAL mfx100848724 SFX frames via psana2 (exact geometry, no .geom).
Peakfind on calibrated jungfrau -> q (psana per-pixel coords) -> index_shot. The data's
cell is unknown, so the test is CONSISTENCY: does fftindex recover the same cell across
frames? A dominant cell both validates fftindex on real data AND reveals the cell."""
import sys

import numpy as np
from scipy.ndimage import maximum_filter

sys.path.insert(0, "..")
from psana import DataSource
from psana.pscalib.geometry.GeometryAccess import GeometryAccess

from fftindex import index_shot
from fftindex.multishot import cell_signature, consensus_cell, same_lattice

import os

from fftindex.lattice import cell_to_Ar
from fftindex.multishot import index_known_pairangle, reference_lattice

HC = 12398.42
N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 80
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
ZDIST = float(os.environ.get("ZDIST", "0.246"))         # detector distance [m] (Brewster: 246mm)
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)      # tetragonal lysozyme (run 51)

ds = DataSource(exp="mfx100848724", run=51)
myrun = next(ds.runs())
det = myrun.Detector("jungfrau")
eb = myrun.Detector("ebeamh")
geo = GeometryAccess()
geo.load_pars_from_str(det.calibconst["geometry"][0])
X, Y, Z = (a.astype(float).reshape(32, 512, 1024) for a in geo.get_pixel_coords())
X, Y = X * 1e-6, Y * 1e-6                                # um -> m (transverse positions)
Zc = np.sign(np.nanmean(Z)) * ZDIST                     # override distance (psana's was nominal)
kin = np.array([0.0, 0.0, np.sign(Zc)])                 # incident beam direction
print(f"detector distance set to {ZDIST} m (psana gave {np.nanmean(Z)/1e6:.3f})", flush=True)
status = det.calibconst["pixel_status"][0].reshape(-1, 32, 512, 1024)  # per gain stage
badmask = (status != 0).any(axis=0)                     # bad if bad in any gain
print(f"geometry+status loaded; bad pixels {100*badmask.mean():.1f}%", flush=True)
MAXPK = 600


def find_q(img, lam, son_min=6.0):
    """Per-panel SNR peakfinding (peakfinder8-like): local maxima well above the
    panel's robust background (median + son_min*MAD)."""
    img = np.where(badmask, 0.0, img)
    mx = maximum_filter(img, size=5)
    keep = np.zeros(img.shape, bool)
    for p in range(32):
        panel = img[p]
        med = np.median(panel)
        sig = 1.4826 * np.median(np.abs(panel - med)) + 1e-6
        keep[p] = (panel == mx[p]) & (panel > med + son_min * sig) & (panel > med + THR)
    idx = np.argwhere(keep)
    if len(idx) < 8:
        return None, len(idx)
    p, r, c = idx.T
    if len(idx) > MAXPK:
        sel = np.argsort(img[p, r, c])[::-1][:MAXPK]
        p, r, c = p[sel], r[sel], c[sel]
    rr = np.stack([X[p, r, c], Y[p, r, c], np.full(len(p), Zc)], axis=1)
    s = rr / np.linalg.norm(rr, axis=1, keepdims=True)
    return (s - kin) / lam, len(idx)


cells, used, indexed, npk, lyso_match = [], 0, 0, [], [0, 0]
for i, evt in enumerate(myrun.events()):
    if used >= N_FRAMES:
        break
    img = det.raw.calib(evt)
    if img is None:
        continue
    pe = eb.raw.ebeamPhotonEnergy(evt)
    lam = HC / (pe if pe and pe > 1000 else 11624.0)
    q, nfound = find_q(img, lam)
    if q is None:
        continue
    used += 1
    npk.append(len(q))
    qn = np.linalg.norm(q, axis=1)
    iu, ju = np.triu_indices(len(q), 1)
    dd = np.sort(np.linalg.norm(q[iu] - q[ju], axis=1))
    qmax = float(qn.max())
    try:
        res = index_shot(q, qmax, n_max=176, tol_frac=0.05, min_inlier_frac=0.5)
    except Exception as e:
        print(f"  frame {used}: index_shot ERR {type(e).__name__}: {e}", flush=True)
        continue
    lyso_blind = res.M is not None and same_lattice(res.M, LYSO)
    if res.M is not None:
        indexed += 1
        cells.append(res.M)
        lyso_match[0] += lyso_blind
    # known-cell (taketwo) constrained to the lysozyme cell (loose tol for real data)
    kn = index_known_pairangle(q, qmax, LYSO, Vref=reference_lattice(LYSO, qmax),
                               tol_frac=0.04, len_tol=0.06, ang_tol=5.0)
    lyso_known = kn.M is not None and same_lattice(kn.M, LYSO)
    lyso_match[1] += lyso_known
    bc = np.round(np.sort(np.linalg.norm(res.M, axis=0)), 1) if res.M is not None else None
    print(f"  frame {used}: peaks={len(q)} qmax={qmax:.3f} dmin={dd[2]:.4f}  "
          f"blind={bc} lyso?={lyso_blind}  known-cell-lyso={lyso_known}", flush=True)

u = max(used, 1)
print(f"\nframes used: {used}; median peaks/frame: {np.median(npk):.0f}")
print(f"fftindex BLIND found a cell in {indexed}/{used} ({100*indexed/u:.0f}%); "
      f"matches lysozyme in {lyso_match[0]}/{used} ({100*lyso_match[0]/u:.0f}%)")
print(f"fftindex KNOWN-CELL (lyso) indexed {lyso_match[1]}/{used} ({100*lyso_match[1]/u:.0f}%)")
if cells:
    Mc, sup = consensus_cell(cells)
    if Mc is not None:
        agree = sum(same_lattice(M, Mc) for M in cells)
        print(f"DOMINANT cell: {agree}/{len(cells)} indexed frames AGREE -> "
              f"{np.round(np.sort(np.linalg.norm(Mc, axis=0)),1)} A; sig {np.round(cell_signature(Mc),1)}")
    else:
        print("no dominant cell (recovered cells scattered)")
