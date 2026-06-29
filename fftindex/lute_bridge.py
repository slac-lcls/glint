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
    for name, d in panels.items():
        if "fs" not in d or "corner_x" not in d:
            continue
        out.append(dict(
            name=name, fs=_vec(d["fs"]), ss=_vec(d["ss"]), res=float(d["res"]),
            cx=float(d["corner_x"]), cy=float(d["corner_y"]),
            coffset=float(d.get("coffset", 0.0)),
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
