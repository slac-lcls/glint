"""CrystFEL .geom parsing + peak(fs,ss) -> reciprocal q bridge, so GLINT ingests exactly what
a LUTE/CrystFEL peakfinder8 run produces (a .geom + per-frame peak lists in data-array fs/ss).

  from glint.geom import parse_geom, peaks_to_q
  geom = parse_geom("detector.geom")
  q = peaks_to_q(peaks_fs_ss, geom)        # peaks: (N,2) [fs, ss] in data-array coords -> (N,3) 1/A

Convention (CrystFEL): a pixel at data-array (fs, ss) lies on the panel whose [min_fs..max_fs] x
[min_ss..max_ss] contains it; panel-local (lfs,lss)=(fs-min_fs, ss-min_ss) map to lab pixel
coords  X = corner_x + lfs*fsx + lss*ssx ,  Y = corner_y + lfs*fsy + lss*ssy ; metres via /res;
z = clen + coffset. Scattered unit s_hat = R/|R|; incident beam +z (s0=(0,0,1)); q=(s_hat-s0)/lambda.
"""
import numpy as np

_HC_eV_A = 12398.419843320026          # h*c in eV.A  ->  lambda_A = _HC_eV_A / E_eV
_GLOBAL = ("photon_energy", "wavelength", "clen", "res", "coffset", "adu_per_eV")
_PANEL_INHERIT = ("res", "clen", "coffset")


def _vec(s):
    """Parse a CrystFEL direction string like '+1.0x -0.0y' or '-0.5x +0.866y +0.0z' -> (x,y,z)."""
    v = [0.0, 0.0, 0.0]
    import re
    for val, axis in re.findall(r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([xyz])", s):
        v["xyz".index(axis)] += float(val)
    return np.array(v, float)


def _wavelength_A(d):
    """Resolve wavelength in Angstrom from photon_energy (eV) or wavelength (m or A)."""
    if "photon_energy" in d:
        return _HC_eV_A / float(d["photon_energy"])
    if "wavelength" in d:
        w = float(d["wavelength"])
        return w * 1e10 if w < 1e-6 else w     # metres -> A, else already A
    return None


def parse_geom(path):
    """Parse a CrystFEL .geom into {'panels': {name: {...}}, 'wavelength_A': float}.
    Panel fields: min_fs,max_fs,min_ss,max_ss,corner_x,corner_y,fsx,fsy,ssx,ssy,res,clen,coffset."""
    g = {}
    panels = {}
    for raw in open(path):
        line = raw.split(";", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, val = (s.strip() for s in line.split("=", 1))
        if "/" in key:
            pname, sub = key.split("/", 1)
            p = panels.setdefault(pname, {})
            if sub in ("fs", "ss"):
                v = _vec(val)
                p[sub + "x"], p[sub + "y"], p[sub + "z"] = v
            else:
                try:
                    p[sub] = float(val)
                except ValueError:
                    p[sub] = val
        elif key in _GLOBAL:
            try:
                g[key] = float(val)
            except ValueError:
                g[key] = val
    # apply global defaults to panels
    for p in panels.values():
        for k in _PANEL_INHERIT:
            if k not in p and k in g:
                p[k] = g[k]
        p.setdefault("coffset", 0.0)
        p.setdefault("fsx", 1.0); p.setdefault("fsy", 0.0)
        p.setdefault("ssx", 0.0); p.setdefault("ssy", 1.0)
    return {"panels": panels, "wavelength_A": _wavelength_A(g), "global": g}


def read_crystfel_peaks(path):
    """Read a CrystFEL stream's 'Peaks from peak search' blocks (what peakfinder8/LUTE emits).
    Returns [{'image':str, 'event':int|str, 'peaks':(N,2) fs,ss}, ...] -- one per chunk."""
    chunks = []
    image = None; event = 0; peaks = []; inpk = False
    for line in open(path):
        s = line.strip()
        if s.startswith("----- Begin chunk"):
            image = None; event = 0; peaks = []; inpk = False
        elif s.startswith("Image filename:"):
            image = s.split(":", 1)[1].strip()
        elif s.startswith("Event:"):
            ev = s.split(":", 1)[1].strip().lstrip("/")
            try:
                event = int(ev)
            except ValueError:
                event = ev
        elif s.startswith("Peaks from peak search"):
            inpk = True
        elif s.startswith("End of peak list"):
            inpk = False
        elif s.startswith("----- End chunk"):
            chunks.append({"image": image, "event": event,
                           "peaks": np.array(peaks, float).reshape(-1, 2)})
        elif inpk:
            p = s.split()
            try:
                peaks.append((float(p[0]), float(p[1])))   # header 'fs/px ss/px ...' -> ValueError, skipped
            except (ValueError, IndexError):
                continue
    return chunks


def _find_panel(fs, ss, panels):
    for name, p in panels.items():
        if (p.get("min_fs", -np.inf) <= fs <= p.get("max_fs", np.inf) and
                p.get("min_ss", -np.inf) <= ss <= p.get("max_ss", np.inf)):
            return name, p
    return None, None


def peaks_to_q(peaks, geom, wavelength_A=None):
    """peaks: (N,2) [fs, ss] in data-array coords (single panel ok). Returns (N,3) q in 1/A.
    Peaks outside every panel are dropped."""
    peaks = np.asarray(peaks, float).reshape(-1, 2)
    panels = geom["panels"] if isinstance(geom, dict) and "panels" in geom else geom
    lam = wavelength_A or (geom.get("wavelength_A") if isinstance(geom, dict) else None)
    if lam is None:
        raise ValueError("no wavelength: pass wavelength_A or set photon_energy/wavelength in .geom")
    out = []
    for fs, ss in peaks:
        name, p = _find_panel(fs, ss, panels)
        if p is None:
            continue
        lfs = fs - p.get("min_fs", 0.0); lss = ss - p.get("min_ss", 0.0)
        X = p["corner_x"] + lfs * p["fsx"] + lss * p["ssx"]
        Y = p["corner_y"] + lfs * p["fsy"] + lss * p["ssy"]
        res = p["res"]
        R = np.array([X / res, Y / res, p["clen"] + p["coffset"]])
        shat = R / np.linalg.norm(R)
        out.append((shat - np.array([0.0, 0.0, 1.0])) / lam)
    return np.array(out) if out else np.empty((0, 3))
