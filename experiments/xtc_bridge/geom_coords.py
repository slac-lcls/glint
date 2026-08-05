"""Per-pixel lab coords from a CrystFEL `.geom`, as a drop-in for psana's `det.coords_x/y/z`.

WHY THIS EXISTS. psana's deployed geometry is often the UNREFINED starting calibration, while the
refinement that downstream processing actually trusts (btx / BayFAI / a CrystFEL refine) lives only
in a `.geom` and is never written back into psana's `*-end.data`. `--zdist` cannot paper over the
difference: it replaces Z, while X and Y still come from psana.

Measured on mfxx49820 r0016 -- pushing btx's OWN peak pixels through psana's `0-end.data` and
comparing to the `1/d` btx recorded for them:

    median |rel| 3.16 %,  max 22.6 %,  but mean rel +0.003 %

i.e. NOT a scale error (refitting one global ratio moved the median only 3.161 -> 3.138 %), and
signed per QUADRANT of the Epix10ka2M -- segments 0-3 and 12-15 one way, 4-11 the other, +-3-5 % --
which is exactly the per-quadrant term a detector refinement fits. That is enough to stop blind
indexing converging on the true cell.

CONVENTIONS, chosen so this is a true drop-in:
  * returns MICROMETRES (psana's unit), so `prep_geometry`'s 1e-6 still applies;
  * returns Z with psana's sign (negative, i.e. detector downstream), so `prep_geometry`'s
    `sign(nanmean(Zf))` and the `kin` it derives are unchanged. The `.geom`'s own z is per-panel
    `clen + coffset`, and `clen` is usually a DAQ path rather than a number -- `--zdist` remains the
    single source of truth for the distance, as it is on the psana path;
  * pixel (fs, ss) of a panel sits at `(corner + fs*fs_vec + ss*ss_vec) / res`, with corner in
    PIXELS and res in pixels/m (CrystFEL's convention);
  * panels are written into the flat data array through their `min_fs/min_ss`, which is the same
    (rows, cols) order psana's `calib` reshapes to -- so the result indexes identically.

The frame is CrystFEL's lab frame, NOT psana's cframe=0. For BLIND indexing that is harmless: cell
lengths and angles are invariant under the rotation between them, and a handedness flip only swaps
the enantiomorph. Do NOT feed the resulting orientations to `indexamajig --indexing=file` unfixed.
"""
from __future__ import annotations

import re

import numpy as np

_VEC = re.compile(r"([+-]?[\d.eE+-]+)\s*([xyz])")


def _vec(s):
    """'-0.999882x -0.000169y -0.015358z' -> (x, y, z); missing components are 0."""
    out = {"x": 0.0, "y": 0.0, "z": 0.0}
    for val, ax in _VEC.findall(s):
        try:
            out[ax] = float(val)
        except ValueError:
            pass
    return out["x"], out["y"], out["z"]


def parse_geom(path):
    """-> {panel_name: {min_fs,max_fs,min_ss,max_ss,corner_x,corner_y,res,fs,ss,coffset}}."""
    panels, defaults = {}, {}
    for raw in open(path, errors="ignore"):
        line = raw.split(";")[0].strip()                 # ';' starts a comment
        if not line or "=" not in line:
            continue
        key, val = (x.strip() for x in line.split("=", 1))
        if "/" not in key:
            defaults[key] = val                          # file-level default (e.g. a global res)
            continue
        name, prop = key.rsplit("/", 1)
        p = panels.setdefault(name, {})
        if prop in ("fs", "ss"):
            p[prop] = _vec(val)
        else:
            try:
                p[prop] = float(val)
            except ValueError:
                p[prop] = val                            # e.g. clen = /LCLS/detector_1/EncoderValue
    for p in panels.values():                            # inherit file-level numeric defaults
        for k, v in defaults.items():
            if k not in p:
                try:
                    p[k] = float(v)
                except ValueError:
                    pass
    return {k: v for k, v in panels.items() if "min_fs" in v and "corner_x" in v}


def coords_from_geom(path, shape, zdist):
    """CrystFEL `.geom` -> (Xf, Yf, Zf) flat arrays in MICROMETRES, sized for `shape` (nseg,H,W).

    Z is a constant -zdist (psana sign convention); see the module docstring for why the .geom's own
    clen/coffset is deliberately not used. Raises if the panel map does not tile `shape` exactly --
    a silent size or ordering mismatch is the failure mode this whole module exists to avoid."""
    panels = parse_geom(path)
    if not panels:
        raise ValueError(f"no panels with min_fs/corner_x parsed from {path}")
    nseg, H, W = shape
    n_rows = max(int(p["max_ss"]) for p in panels.values()) + 1
    n_cols = max(int(p["max_fs"]) for p in panels.values()) + 1
    if n_rows * n_cols != nseg * H * W:
        raise ValueError(
            f"{path} tiles {n_rows}x{n_cols} = {n_rows*n_cols} px, but the detector array is "
            f"{nseg}x{H}x{W} = {nseg*H*W}. The .geom does not describe this detector.")
    if n_cols != W:
        raise ValueError(f"{path} is {n_cols} px wide but the calib array is {W}; the panel map "
                         f"would be transposed against psana's row-major reshape.")

    X = np.full((n_rows, n_cols), np.nan)
    Y = np.full((n_rows, n_cols), np.nan)
    for name, p in panels.items():
        try:
            f0, f1 = int(p["min_fs"]), int(p["max_fs"])
            s0, s1 = int(p["min_ss"]), int(p["max_ss"])
            res = float(p["res"])
            fsv, ssv = p["fs"], p["ss"]
        except (KeyError, ValueError) as e:
            raise ValueError(f"panel {name} in {path} is missing a required key: {e}")
        ss, fs = np.meshgrid(np.arange(s1 - s0 + 1), np.arange(f1 - f0 + 1), indexing="ij")
        # corner is in PIXELS; divide the whole lot by res (px/m) -> metres, then 1e6 -> um
        X[s0:s1 + 1, f0:f1 + 1] = (p["corner_x"] + fs * fsv[0] + ss * ssv[0]) / res * 1e6
        Y[s0:s1 + 1, f0:f1 + 1] = (p["corner_y"] + fs * fsv[1] + ss * ssv[1]) / res * 1e6

    if np.isnan(X).any():
        raise ValueError(f"{path} leaves {int(np.isnan(X).sum())} pixels uncovered -- the panels do "
                         f"not tile the array")
    Zf = np.full(X.size, -abs(zdist) * 1e6)              # psana sign convention, um
    return X.ravel(), Y.ravel(), Zf
