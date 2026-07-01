"""Phase B on REAL data (NERSC): GLINT-index cxidb-45 Proteinase K (SACLA MPCCD) -> predict ->
integrate -> CrystFEL .stream with real I/sigma.

Per frame: peakfind the MPCCD image -> q (mnasser MPCCD geom, per-shot lambda) -> GLINT blind index.
Keep frames whose lattice matches ProK (P43212 ~68/68/109); CANONICALISE the basis (sort axes by
length so c=longest is consistent, fix handedness) so hkl are consistent across frames -- the residual
a<->b / sign ambiguity is absorbed by the 4/mmm merge symmetry. Then predict_spots + integrate_spots
-> stream. (Sub-lattice frames are dropped here; known-cell rescue would add them back.)

  module load pytorch/2.6.0 ; python phaseB_prok.py [N] [out.stream]
"""
import os, sys, re
sys.path.insert(0, "/pscratch/sd/s/smarches/glint_real")
os.environ.setdefault("STEPS", "8"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import h5py
from scipy.ndimage import maximum_filter
from glint.lute_bridge import peaks_to_q
from glint.glint_fast import index_blind_fast
from glint.multishot import same_lattice
from glint.predict import predict_spots, integrate_spots, write_stream_integrated

GEOM = "/global/cfs/cdirs/lcls/mnasser/cxidb_62/mpccd-optimized.geom"
H5DIR = "/global/cfs/cdirs/lcls/dermen/cxidb45/data"
H5S = [f"{H5DIR}/run296940-0.h5", f"{H5DIR}/run296940-1.h5", f"{H5DIR}/run296940-2.h5"]
PROK = np.diag([68.7, 68.7, 108.6])              # ProK reference metric (P43212)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = sys.argv[2] if len(sys.argv) > 2 else "/pscratch/sd/s/smarches/glint_real/prok_glint.stream"
DMIN = float(os.environ.get("DMIN", "2.1")); TOL = float(os.environ.get("PTOL", "0.010"))  # PTOL != glint_fast's TOL knob


def _vec(s):
    out = [0.0, 0.0, 0.0]
    for m in re.finditer(r"([-+0-9.eE]+)\s*([xyz])", s):
        out["xyz".index(m.group(2))] = float(m.group(1))
    return np.array(out)


def load_geom(path):
    glob, pan = {}, {}
    for line in open(path):
        line = line.split(";")[0].strip()
        if "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        if "/" in k:
            p, prop = k.split("/", 1); pan.setdefault(p, {})[prop] = v
        else:
            glob[k] = v
    rg = float(glob.get("res", 1.0)); cg = float(glob.get("coffset", 0.0))
    panels = []
    for nm, d in pan.items():
        if "fs" not in d or "corner_x" not in d:
            continue
        panels.append(dict(name=nm, fs=_vec(d["fs"]), ss=_vec(d["ss"]),
                           res=float(d.get("res", rg)), cx=float(d["corner_x"]), cy=float(d["corner_y"]),
                           coffset=float(d.get("coffset", cg)),
                           min_fs=int(d["min_fs"]), max_fs=int(d["max_fs"]),
                           min_ss=int(d["min_ss"]), max_ss=int(d["max_ss"])))
    return panels, glob


def peakfind(img, nmax=170, snr=8.0):
    bg = np.median(img)
    sig = 1.4826 * np.median(np.abs(img - bg)) + 1e-3
    mx = maximum_filter(img, size=5)
    ys, xs = np.where((img == mx) & (img > bg + snr * sig))
    if len(xs) == 0:
        return np.empty((0, 2))
    order = np.argsort(img[ys, xs])[::-1][:nmax]
    return np.column_stack([xs[order], ys[order]]).astype(float)


def refine_orient(q, M, tol_abs, iters=6):
    """Predict-refine: assign peaks to integer hkl, least-squares re-fit R (q = hkl @ R), iterate.
    Tightens the blind orientation so predicted spots land on real peaks (better integration)."""
    R = np.linalg.inv(M)
    inl = np.ones(len(q), bool)
    for _ in range(iters):
        hkl = np.round(q @ np.linalg.inv(R))
        res = np.linalg.norm(q - hkl @ R, axis=1)
        inl = res < tol_abs
        if inl.sum() < 8:
            break
        R, *_ = np.linalg.lstsq(hkl[inl], q[inl], rcond=None)
    return np.linalg.inv(R), int(inl.sum())


def canonicalize(M):
    """Consistent basis: columns sorted by axis length (c=longest last); proper handedness."""
    order = np.argsort(np.linalg.norm(M, axis=0))
    Mc = M[:, order].copy()
    if np.linalg.det(Mc) < 0:
        Mc[:, 0] = -Mc[:, 0]
    return Mc


panels, glob = load_geom(GEOM)
clen = float(glob.get("clen", 0.055))
print(f"geom {len(panels)} panels clen={clen} res={panels[0]['res']}  DMIN={DMIN} TOL={TOL}")
index_blind_fast(np.random.rand(20, 3) * 0.1)    # warmup

results = []
n_seen = n_idx = n_keep = tot = 0
for h5 in H5S:
    if n_seen >= N:
        break
    f = h5py.File(h5, "r")
    for t in [k for k in f if k.startswith("tag")]:
        if n_seen >= N:
            break
        n_seen += 1
        img = f[t]["data"][()].astype(np.float32)
        lam = float(f[t]["photon_wavelength_A"][()])
        pk = peakfind(img)
        if len(pk) < 8:
            results.append({"image": h5, "event": n_seen, "M": None}); continue
        q = peaks_to_q(pk[:, 0], pk[:, 1], panels, clen, lam)
        q = q[~np.isnan(q).any(1)]
        M = index_blind_fast(q)
        if M is None:
            results.append({"image": h5, "event": n_seen, "M": None}); continue
        n_idx += 1
        if not same_lattice(M, PROK):
            results.append({"image": h5, "event": n_seen, "M": None}); continue
        qmax = np.linalg.norm(q, axis=1).max()
        M, ninl = refine_orient(q, M, tol_abs=0.012 * qmax)   # predict-refine the orientation
        M = canonicalize(M)
        pred = predict_spots(M, panels, clen, lam, dmin=DMIN, tol=TOL)
        I, sig, peak, bg = integrate_spots(img, pred, half=3, gap=2, ring=3)
        keep = (I > 0) & np.isfinite(sig) & (sig > 0)
        pred, I, sig, peak, bg = pred[keep], I[keep], sig[keep], peak[keep], bg[keep]
        results.append({"image": os.path.basename(h5), "event": n_seen, "M": M,
                        "pred": pred, "I": I, "sigma": sig, "peak": peak, "bg": bg})
        n_keep += 1; tot += len(pred)
        if n_keep <= 6:
            a, b, c = (np.linalg.norm(M[:, i]) for i in range(3))
            print(f"  {t}: {len(pred)} refl  cell {a:.1f} {b:.1f} {c:.1f}")

eV = 12398.42 / lam
gt = (f"clen = {clen}\nres = {panels[0]['res']}\nadu_per_photon = 1\nphoton_energy = {eV:.1f}\n"
      "p0/min_fs = 0\np0/max_fs = 511\np0/min_ss = 0\np0/max_ss = 8191\n"
      "p0/corner_x = -256\np0/corner_y = -256\np0/fs = x\np0/ss = y\n")
n = write_stream_integrated(results, OUT, geom_text=gt, photon_eV=eV, clen_m=clen)
print(f"\nseen {n_seen}  blind-indexed {n_idx}  ProK-lattice kept {n_keep}  "
      f"refl {tot} (~{tot//max(n_keep,1)}/frame)")
print(f"wrote {OUT}  (indexed chunks {n})")
