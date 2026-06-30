"""Phase B (ana): predict + integrate the simulated stills -> a CrystFEL .stream with REAL I/sigma.

ORIENT=truth : use the known nanoBragg orientation per still (M = inv(A_recip @ G^T), the validated
               frame relation) -> predicted hkl ARE nanoBragg's true hkl. Upper bound: isolates the
               predict+integrate+merge machinery from indexing.
ORIENT=glint : GLINT blind-indexes each still's peaks (its own job), then we predict+integrate from
               GLINT's orientation -- the actual frontier-#1 claim. hkl are relabelled into the truth
               setting by the per-frame lattice-symmetry op (known-cell merge consistency).

  source psconda.sh ; conda activate ana-... ; ORIENT=truth python phaseB_integrate.py [TOL]
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
os.environ.setdefault("STEPS", "8")
import numpy as np
from scipy.ndimage import maximum_filter
from fftindex.lute_bridge import peaks_to_q
from fftindex.predict import predict_spots, integrate_spots, write_stream_integrated

ORIENT = os.environ.get("ORIENT", "truth")
TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.004
SIM = "/sdf/home/s/smarches/glint_sim"
OUTSTREAM = f"/sdf/home/s/smarches/glint_sim/glint_{ORIENT}.stream"
G = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], float)   # detector-frame -> nanoBragg-lab (pinned)

T = np.load(f"{SIM}/truth.npz")
imgs = np.load(f"{SIM}/images.npy")
K, N = imgs.shape[0], int(T["det_n"])
clen = float(T["dist_mm"]) / 1000.0
wave = float(T["wave_A"]); dmin = float(T["dmin"]); bg0 = float(T["bg"])
res = 1.0 / (float(T["pix_mm"]) / 1000.0)
corner = -(N / 2.0 - 0.5)
panels = [dict(name="p0", fs=np.array([1., 0, 0]), ss=np.array([0, 1., 0]), res=res,
               cx=corner, cy=corner, coffset=0.0, min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
Amats = T["Amats"]


def peakfind(img, nmax=140):
    mx = maximum_filter(img, size=5)
    thr = bg0 + 5.0 * np.sqrt(bg0)
    ys, xs = np.where((img == mx) & (img > thr))
    if len(xs) == 0:
        return np.empty((0, 2))
    order = np.argsort(img[ys, xs])[::-1][:nmax]
    return np.column_stack([xs[order], ys[order]]).astype(float)   # (fs, ss)


def glint_M(img):
    from glint_fast import index_blind_fast
    pk = peakfind(img)
    if len(pk) < 6:
        return None
    q = peaks_to_q(pk[:, 0], pk[:, 1], panels, clen, wave)
    q = q[~np.isnan(q).any(1)]
    return index_blind_fast(q)


results = []
n_idx = 0
tot_refl = 0
for k in range(K):
    img = imgs[k].astype(float)
    if ORIENT == "truth":
        M = np.linalg.inv(Amats[k].T @ G.T)
    else:
        Mg = glint_M(img)
        if Mg is None:
            results.append({"image": f"sim_{k}", "event": k, "M": None}); continue
        # relabel GLINT's basis into the truth setting: S = round(inv(M_true) @ M_glint) (lattice sym op)
        Mt = np.linalg.inv(Amats[k].T @ G.T)
        S = np.round(np.linalg.inv(Mt) @ Mg)
        M = Mg @ np.linalg.inv(S) if abs(np.linalg.det(S)) > 0.5 else Mg
    pred = predict_spots(M, panels, clen, wave, dmin=dmin, tol=TOL)
    I, sig, peak, bg = integrate_spots(img, pred, half=3, gap=2, ring=3)
    keep = (I > 0) & np.isfinite(sig) & (sig > 0)
    pred, I, sig, peak, bg = pred[keep], I[keep], sig[keep], peak[keep], bg[keep]
    results.append({"image": f"sim_{k}", "event": k, "M": M, "pred": pred,
                    "I": I, "sigma": sig, "peak": peak, "bg": bg})
    n_idx += 1; tot_refl += len(pred)
    print(f"  still {k:2d}: {len(pred):4d} refl integrated  (>3sig: {(I>3*sig).sum()})", flush=True)

geom_text = (f"clen = {clen}\nres = {res}\nadu_per_photon = 1\n"
             f"photon_energy = {12398.42/wave:.1f}\n"
             f"p0/min_fs = 0\np0/max_fs = {N-1}\np0/min_ss = 0\np0/max_ss = {N-1}\n"
             f"p0/corner_x = {corner}\np0/corner_y = {corner}\np0/fs = x\np0/ss = y\n")
n = write_stream_integrated(results, OUTSTREAM, geom_text=geom_text)
print(f"\nORIENT={ORIENT}  indexed {n_idx}/{K}  total refl {tot_refl}  (~{tot_refl//max(n_idx,1)}/frame)")
print(f"wrote {OUTSTREAM}")
