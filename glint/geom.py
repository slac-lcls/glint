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
# `data` is the HDF5 dataset path of the frame images -- a string, consumed by
# glint.predict.integrate_frames. It was missing from this tuple until glint#143, so a .geom's
# `data = /some/path` line was silently dropped and integration always fell back to /data/data.
_GLOBAL = ("photon_energy", "wavelength", "clen", "res", "coffset", "adu_per_eV", "data")
_PANEL_INHERIT = ("res", "clen", "coffset")


def _vec(s):
    """Parse a CrystFEL direction string like '+1.0x -0.0y' or '-0.5x +0.866y +0.0z' -> (x,y,z)."""
    v = [0.0, 0.0, 0.0]
    import re
    for val, axis in re.findall(r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([xyz])", s):
        v["xyz".index(axis)] += float(val)
    return np.array(v, float)


def _wavelength_A(d):
    """Resolve wavelength in Angstrom from photon_energy (eV) or wavelength (m or A).

    Returns None when the geometry does not carry a LITERAL value -- either the key is absent, or
    it is an HDF5 path (`photon_energy = /LCLS/photon_energy_eV`), which is the NORMAL form in an
    LCLS .geom because the energy is per-shot. There is nothing to resolve it against on the
    `--peaks` route, so the caller supplies `--wavelength`; this used to raise on the path form
    instead, which made every stock LCLS .geom unparseable before that override was ever read."""
    if "photon_energy" in d:
        try:
            return _HC_eV_A / float(d["photon_energy"])
        except (TypeError, ValueError):
            return None
    if "wavelength" in d:
        try:
            w = float(d["wavelength"])
        except (TypeError, ValueError):
            return None
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
    image = None; event = 0; peaks = []; inpk = False; open_chunk = False
    for line in open(path):
        s = line.strip()
        if s.startswith("----- Begin chunk"):
            image = None; event = 0; peaks = []; inpk = False; open_chunk = True
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
            # One frame per Begin/End PAIR. Without the latch a stream carrying a stray second
            # 'End chunk' emits the frame again -- same image, same event, same peaks, because
            # nothing is reset until the next 'Begin chunk'. Silent duplicates do not change a
            # percentage but they double every count and every consensus pool, which reads as more
            # evidence than there is. Measured: a 3000-chunk stream parsed as N=6000.
            if not open_chunk:
                continue
            open_chunk = False
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
    """First panel containing the PIXEL that (fs,ss) lands in. CrystFEL's min_fs/max_fs are integer
    pixel indices, so the test floors first: fs=511.7 lies in pixel 511, i.e. on a max_fs=511 panel.
    Kept for callers that want the panel name; peaks_to_q uses the vectorised _q_from_panels."""
    fi, si = np.floor(fs), np.floor(ss)
    for name, p in panels.items():
        if (p.get("min_fs", -np.inf) <= fi <= p.get("max_fs", np.inf) and
                p.get("min_ss", -np.inf) <= si <= p.get("max_ss", np.inf)):
            return name, p
    return None, None


def _q_from_panels(fs_arr, ss_arr, specs, wavelength_A):
    """Shared core behind BOTH public peaks_to_q entry points (this module's and lute_bridge's).

    They exist because there are two panel schemas -- this module's dict-of-dicts from a .geom, and
    lute_bridge's list-of-dicts with per-event clen -- not because the geometry differs. Keeping the
    maths in one place is the point: the two used to carry independent copies and silently drifted
    apart (geom's membership test omitted the floor, so it dropped every peak in the last fractional
    pixel of each panel).

    specs: list of dicts, one per panel, in priority order (first match wins), with
      lo_fs/hi_fs/lo_ss/hi_ss  inclusive PIXEL bounds for the membership test (may be +-inf)
      off_fs/off_ss            panel-local origin subtracted from (fs,ss)  -- NOT always the bound:
                               a .geom panel may omit min_fs, in which case the bound is -inf but
                               the offset is 0
      fsx/fsy/ssx/ssy          panel basis vectors in lab pixel coords
      cx/cy                    panel corner in lab pixel coords
      res                      pixels per metre
      z                        lab z of the panel [m] (clen + coffset)

    Returns (n,3) q [1/A] with a NaN row wherever a peak matched no panel; callers choose whether to
    keep those rows (positional correspondence) or drop them.
    """
    fs_arr = np.asarray(fs_arr, float)
    ss_arr = np.asarray(ss_arr, float)
    n = len(fs_arr)
    r = np.full((n, 3), np.nan)
    fi = np.floor(fs_arr); si = np.floor(ss_arr)      # fractional peak -> the pixel it lands in
    free = np.ones(n, bool)                            # reproduces "first panel that catches it"
    for p in specs:
        m = (free & (fi >= p["lo_fs"]) & (fi <= p["hi_fs"])
             & (si >= p["lo_ss"]) & (si <= p["hi_ss"]))
        if not m.any():
            continue
        lf = fs_arr[m] - p["off_fs"]; ls = ss_arr[m] - p["off_ss"]
        r[m, 0] = (p["cx"] + lf * p["fsx"] + ls * p["ssx"]) / p["res"]
        r[m, 1] = (p["cy"] + lf * p["fsy"] + ls * p["ssy"]) / p["res"]
        r[m, 2] = p["z"]
        free &= ~m
    s_hat = r / np.linalg.norm(r, axis=1, keepdims=True)
    return (s_hat - np.array([0.0, 0.0, 1.0])) / wavelength_A


def _panel_z(name, p):
    """clen + coffset [m], insisting BOTH are literals.

    `clen` is frequently an HDF5 path in an LCLS .geom (`clen = /LCLS/detector_1/EncoderValue`)
    because the distance is per-run. This route has no file to resolve it against, and adding a str
    to a float raises `can only concatenate str` -- which names the language, not the geometry."""
    out = 0.0
    for k in ("clen", "coffset"):
        v = p.get(k, 0.0)
        if isinstance(v, str):
            raise ValueError(
                f"panel {name}: '{k} = {v}' is an HDF5 path, not a distance. The --peaks route has "
                f"no file to read it from -- edit the .geom to a literal, or use a route that "
                f"supplies the distance per event.")
        out += float(v)
    return out


def _specs_from_geom_panels(panels):
    """.geom dict-of-dicts -> _q_from_panels specs. Bounds default to +-inf (a single-panel .geom
    need not declare min_fs) while the panel-local origin defaults to 0 -- they are different
    defaults for the same missing key, which is why the core takes them separately."""
    return [dict(lo_fs=p.get("min_fs", -np.inf), hi_fs=p.get("max_fs", np.inf),
                 lo_ss=p.get("min_ss", -np.inf), hi_ss=p.get("max_ss", np.inf),
                 off_fs=p.get("min_fs", 0.0), off_ss=p.get("min_ss", 0.0),
                 fsx=p["fsx"], fsy=p["fsy"], ssx=p["ssx"], ssy=p["ssy"],
                 cx=p["corner_x"], cy=p["corner_y"], res=p["res"],
                 z=_panel_z(name, p))
            for name, p in panels.items()]


def peaks_to_q(peaks, geom, wavelength_A=None):
    """peaks: (N,2) [fs, ss] in data-array coords (single panel ok). Returns (M,3) q in 1/A.

    Peaks outside every panel are DROPPED, so M <= N and rows do not correspond positionally to the
    input. That is this entry point's contract; `lute_bridge.peaks_to_q` shares the same geometry
    (via `_q_from_panels`) but keeps NaN rows instead, because its streaming caller needs the
    correspondence. Pick by which contract you want, not by which import is closer to hand.
    """
    peaks = np.asarray(peaks, float).reshape(-1, 2)
    panels = geom["panels"] if isinstance(geom, dict) and "panels" in geom else geom
    lam = wavelength_A or (geom.get("wavelength_A") if isinstance(geom, dict) else None)
    if lam is None:
        raise ValueError("no wavelength: pass wavelength_A or set photon_energy/wavelength in .geom")
    if len(peaks) == 0:
        return np.empty((0, 3))
    q = _q_from_panels(peaks[:, 0], peaks[:, 1], _specs_from_geom_panels(panels), lam)
    return q[np.isfinite(q).all(1)]
