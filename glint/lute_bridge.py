"""Bridge from CrystFEL geometry + Cheetah/CXI peak lists to reciprocal q-vectors,
so fftindex can be benchmarked on the SAME real LCLS peaks an indexer (xgandalf) sees.

CrystFEL .geom defines, per panel: fs/ss basis vectors (lab frame), res (px/m),
corner_(x,y) in px, coffset (m), and the panel's [min/max fs,ss] slab in the data
array. A peak at data-array (fs,ss) -> find its panel -> lab position
    r = (corner + (fs-min_fs)*fs_vec + (ss-min_ss)*ss_vec)/res   (x,y in m)
    z = clen + coffset                                            (m)
-> scattered unit vector s_hat = r/|r|, q = (1/lambda)(s_hat - z_hat).

lambda from photon_energy (eV); clen and photon_energy are usually h5 paths in the
geom (read per-event from the CXI). NOTE: the peak (fs,ss) -> panel mapping for a
panel-stacked (dim0='%') detector needs confirming against a real CXI -- see
experiments/bench_lute.py `inspect`.
"""

from __future__ import annotations

import re

import numpy as np

HC_EV_A = 12398.419843320026     # h*c in eV*Angstrom -> lambda[A] = HC/E[eV]


def _vec(s):
    v = [0.0, 0.0, 0.0]
    for m in re.finditer(r"([+-]?[\d.eE]+)\s*([xyz])", s):
        v["xyz".index(m.group(2))] = float(m.group(1))
    return np.array(v)


def parse_geom(path):
    """Return (panels, globals). panels: list of dicts; globals: raw key->value."""
    panels, glob = {}, {}
    with open(path) as f:
        for line in f:
            line = line.split(";")[0].strip()
            if not line or "=" not in line:
                continue
            key, val = (x.strip() for x in line.split("=", 1))
            if "/" in key and not key.startswith("/"):
                pn, prop = key.split("/", 1)
                panels.setdefault(pn, {})[prop] = val
            else:
                glob[key] = val
    out = []
    gres = glob.get("res")                                  # res may be global or per-panel
    for name, d in panels.items():
        if "fs" not in d or "corner_x" not in d:
            continue
        out.append(dict(
            name=name, fs=_vec(d["fs"]), ss=_vec(d["ss"]), res=float(d.get("res", gres)),
            cx=float(d["corner_x"]), cy=float(d["corner_y"]),
            coffset=float(d.get("coffset", glob.get("coffset", 0.0))),
            min_fs=int(d["min_fs"]), max_fs=int(d["max_fs"]),
            min_ss=int(d["min_ss"]), max_ss=int(d["max_ss"])))
    return out, glob


def panel_of(fs, ss, panels):
    fi, si = int(np.floor(fs)), int(np.floor(ss))   # fractional peak -> the pixel it lands in
    for i, p in enumerate(panels):
        if p["min_fs"] <= fi <= p["max_fs"] and p["min_ss"] <= si <= p["max_ss"]:
            return i
    return -1


def peaks_to_q(fs_arr, ss_arr, panels, clen_m, wavelength_A):
    """Detector peak (fs,ss) arrays -> (n,3) reciprocal vectors q [1/A].

    clen_m: detector distance [m] (per event). wavelength_A: [A].
    """
    fs_arr = np.asarray(fs_arr, float)
    ss_arr = np.asarray(ss_arr, float)
    r = np.zeros((len(fs_arr), 3))
    for k, (fs, ss) in enumerate(zip(fs_arr, ss_arr)):
        i = panel_of(fs, ss, panels)
        if i < 0:
            r[k] = np.nan
            continue
        p = panels[i]
        lf, ls = fs - p["min_fs"], ss - p["min_ss"]
        xy = (np.array([p["cx"], p["cy"], 0.0]) + lf * p["fs"] + ls * p["ss"]) / p["res"]
        r[k] = [xy[0], xy[1], clen_m + p["coffset"]]
    s_hat = r / np.linalg.norm(r, axis=1, keepdims=True)
    z_hat = np.array([0.0, 0.0, 1.0])
    return (s_hat - z_hat) / wavelength_A      # q in 1/A


def lambda_from_eV(eV):
    return HC_EV_A / float(eV)


def _meta(spec, h5, i, default):
    """A .geom global value: a plain float, or an h5 path ('/...') read per event from the CXI."""
    if spec is None:
        return default
    try:
        return float(spec)
    except (TypeError, ValueError):
        pass
    if isinstance(spec, str) and spec.startswith("/") and h5 is not None and spec in h5:
        v = h5[spec]
        return float(v[i]) if getattr(v, "ndim", 0) >= 1 and v.shape[0] > i else float(v[()])
    return default


def frames_from_cxi(cxi_path, geom_path, wavelength_A=None, n=0, min_peaks=6,
                    data_key=None, clen_scale=None, **pf_kw):
    """Self-contained GLINT front end: read raw detector images from a jf16m .cxi, GPU peak-find them with
    our peakfinder_v4, and bridge to reciprocal q-vectors -- no CrystFEL peak-search stream in between.
    Returns (frames [(N,3) q in 1/A], images [{image,event}]). clen/photon_energy may be per-event h5 paths
    (read from the CXI); clen_scale converts the encoder units to metres (auto: >10 => assume mm)."""
    import h5py
    from glint.peakfinder_v4 import peakfinder_v4
    panels, glob = parse_geom(geom_path)
    data_key = data_key or glob.get("data", "/entry_1/data_1/data")
    clen_spec, en_spec, mask_key = glob.get("clen"), glob.get("photon_energy"), glob.get("mask")
    coff = float(glob.get("coffset", 0.0))
    f = h5py.File(cxi_path, "r")
    data = f[data_key]
    nfr = data.shape[0] if data.ndim >= 3 else 1
    if n:
        nfr = min(n, nfr)
    cmask = None
    if mask_key and mask_key in f:
        m = f[mask_key]
        m = np.asarray(m[0] if m.ndim >= 3 else m)
        cmask = (m == int(str(glob.get("mask_good", "0")), 0))     # True = good pixel
    frames, images = [], []
    for i in range(nfr):
        img = np.asarray(data[i] if data.ndim >= 3 else data, np.float32)
        pk = peakfinder_v4(img, mask=cmask, **pf_kw)
        images.append({"image": cxi_path, "event": i})
        if len(pk["x"]) < min_peaks:
            frames.append(np.empty((0, 3)))
            continue
        clen = _meta(clen_spec, f, i, 0.1)
        scale = clen_scale if clen_scale is not None else (0.001 if abs(clen) > 10 else 1.0)
        clen = clen * scale + coff
        wl = wavelength_A
        if wl is None:
            eV = _meta(en_spec, f, i, None)
            wl = lambda_from_eV(eV) if eV else None
        if wl is None:
            raise SystemExit("frames_from_cxi: no wavelength (geom photon_energy path or --wavelength)")
        frames.append(peaks_to_q(np.asarray(pk["x"]), np.asarray(pk["y"]), panels, clen, wl))
    return frames, images
