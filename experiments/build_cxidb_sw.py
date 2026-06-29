"""cxidb-17 weak-peak Stage C extractor (S3DF). Parse the WORKING CrystFEL geometry
(cspad_asm.geom, 64 CSPAD ASICs in the assembled /data/data array) -> per-pixel lab
x,y; peakfind each run0349/*.h5 image at a STRONG and a WEAK (sub-threshold) SNR; map
peaks -> reciprocal q (1/A). Strong = the normal set; weak = lower-SNR peaks NOT in
strong = the sub-threshold reflections for the soft-completeness test. Known cell =
lysozyme P 79/79/38. Validate by glint-indexing the strong q.

  source psconda.sh ; python build_cxidb_sw.py [N] [out.npz]
"""
import sys, os, glob, re
import numpy as np
from scipy.ndimage import maximum_filter
import h5py

C = "/sdf/data/lcls/ds/mfx/mfx101629726/scratch/smarches/cxidb17/crystfel"
GEOM = C + "/cspad_asm.geom"
HC = 12398.42
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
OUT = sys.argv[2] if len(sys.argv) > 2 else "/sdf/home/s/smarches/cxidb_sw.npz"
S_SON, S_ABS = 6.0, 100.0
W_SON, W_ABS = float(os.environ.get("W_SON", "3.0")), float(os.environ.get("W_ABS", "20.0"))
CAP_S, CAP_W = 500, 1000


def parse_geom(path):
    glob_, panels = {}, {}
    for line in open(path):
        line = line.split(";")[0].strip()
        if "=" not in line:
            continue
        key, val = [s.strip() for s in line.split("=", 1)]
        if "/" in key and not key.startswith("/"):
            p, prop = key.split("/", 1)
            panels.setdefault(p, {})[prop] = val
        else:
            glob_[key] = val
    return glob_, panels


def _vec(s):
    xm = re.search(r"([-+0-9.eE]+)\s*x", s); ym = re.search(r"([-+0-9.eE]+)\s*y", s)
    return (float(xm.group(1)) if xm else 0.0, float(ym.group(1)) if ym else 0.0)


def build_maps(panels, shape):
    H, W = shape
    AX = np.full((H, W), np.nan, np.float32); AY = np.full((H, W), np.nan, np.float32)
    valid = np.zeros((H, W), bool)
    for p in panels.values():
        mnf, mxf = int(p["min_fs"]), int(p["max_fs"])
        mns, mxs = int(p["min_ss"]), int(p["max_ss"])
        res = float(p["res"]); cx, cy = float(p["corner_x"]), float(p["corner_y"])
        fsx, fsy = _vec(p["fs"]); ssx, ssy = _vec(p["ss"])
        LF, LS = np.meshgrid(np.arange(mxf - mnf + 1), np.arange(mxs - mns + 1))
        AX[mns:mxs + 1, mnf:mxf + 1] = (cx + fsx * LF + ssx * LS) / res
        AY[mns:mxs + 1, mnf:mxf + 1] = (cy + fsy * LF + ssy * LS) / res
        valid[mns:mxs + 1, mnf:mxf + 1] = True
    return AX, AY, valid, [(int(p["min_ss"]), int(p["max_ss"]), int(p["min_fs"]), int(p["max_fs"]))
                           for p in panels.values()]


def peaks(img, valid, blocks, son, absmin):
    im = np.where(valid, img.astype(np.float32), 0.0)
    mx = maximum_filter(im, size=5)
    keep = np.zeros(img.shape, bool)
    for (a, b, c, d) in blocks:
        pan = im[a:b + 1, c:d + 1]
        med = np.median(pan); sig = 1.4826 * np.median(np.abs(pan - med)) + 1e-6
        loc = (im[a:b + 1, c:d + 1] == mx[a:b + 1, c:d + 1]) & \
              (pan > med + son * sig) & (pan > med + absmin)
        keep[a:b + 1, c:d + 1] = loc
    return np.argwhere(keep)


if __name__ == "__main__":
    glob_, panels = parse_geom(GEOM)
    clen = float(glob_["clen"])
    files = sorted(glob.glob(C + "/run0349/*.h5"))
    print(f"{len(panels)} panels, clen={clen} m, {len(files)} images", flush=True)
    AX = AY = valid = blocks = None
    kin = np.array([0.0, 0.0, 1.0])
    frames = []
    for f in files:
        if len(frames) >= N:
            break
        with h5py.File(f, "r") as h:
            img = h["/data/data"][:]
            try:
                lam = float(h["/LCLS/photon_wavelength_A"][0])
            except Exception:
                lam = HC / float(h["/LCLS/photon_energy_eV"][0])
        if AX is None:
            AX, AY, valid, blocks = build_maps(panels, img.shape)
        si = peaks(img, valid, blocks, S_SON, S_ABS)
        if len(si) < 8:
            continue
        wi = peaks(img, valid, blocks, W_SON, W_ABS)

        def to_q(idx, cap):
            if len(idx) == 0:
                return np.zeros((0, 3), np.float32)
            r, c = idx[:, 0], idx[:, 1]
            inten = img[r, c]
            if cap and len(idx) > cap:
                sel = np.argsort(inten)[::-1][:cap]; r, c = r[sel], c[sel]
            rr = np.stack([AX[r, c], AY[r, c], np.full(len(r), clen)], 1)
            s = rr / np.linalg.norm(rr, axis=1, keepdims=True)
            return ((s - kin) / lam).astype(np.float32)

        qs = to_q(si, CAP_S); qw_all = to_q(wi, CAP_W)
        if len(qw_all) and len(qs):
            d = np.linalg.norm(qw_all[:, None] - qs[None], axis=2).min(1)
            qw = qw_all[d > 1e-5]
        else:
            qw = qw_all
        frames.append((qs, qw))
        print(f"frame {len(frames)}: strong={len(qs)} weak_sub={len(qw)} lam={lam:.4f}", flush=True)
    blob = {"n": len(frames)}
    blob.update({f"s{i}": f[0] for i, f in enumerate(frames)})
    blob.update({f"w{i}": f[1] for i, f in enumerate(frames)})
    np.savez(OUT, **blob)
    print(f"wrote {len(frames)} -> {OUT}", flush=True)
