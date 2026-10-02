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
import sys

import numpy as np

from glint.geom import Q_FLOOR, _q_from_panels, clean_q   # one geometry core, two panel schemas -- see geom.py

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
        p = dict(
            name=name, fs=_vec(d["fs"]), ss=_vec(d["ss"]), res=float(d.get("res", gres)),
            cx=float(d["corner_x"]), cy=float(d["corner_y"]),
            coffset=float(d.get("coffset", glob.get("coffset", 0.0))),
            min_fs=int(d["min_fs"]), max_fs=int(d["max_fs"]),
            min_ss=int(d["min_ss"]), max_ss=int(d["max_ss"]))
        # Carry the dimN keys through in geom.parse_geom's convention -- integers as floats, axis
        # names ('%', 'ss', 'fs') as strings -- so predict._panel_slab reads the slab mapping off
        # THESE panels too. This schema dropped them, which left integrate_cxi's layout decision
        # blind to a slab-mapped geometry: a 2-slab/4-panel stack then fell through the
        # "leading axis != n_panels -> events" rule and data[ev] integrated slab ev as an
        # assembled frame, which is glint#148 on the --images route (Copilot review of #183).
        for k in ("dim0", "dim1", "dim2", "dim3"):
            if k in d:
                try:
                    p[k] = float(d[k])
                except ValueError:
                    p[k] = d[k]
        out.append(p)
    return out, glob


def slab_rects(panels, shape):
    """Each panel's rectangle in a 2-D data array, as half-open (ss0, ss1, fs0, fs1), when the geometry
    tiles that array with MORE than one panel -- the case where a local peak finder (v4/pf9) has to run
    per panel, or its ring, local-max window and labelling reach across a panel seam into rows that
    are somewhere else in the lab. See peakfinder_v4.PerPanelFinder.

    None means "one finder over the whole array", exactly as before, for:
      * one panel, or panels that all share one rectangle;
      * a 3-D/4-D layout, i.e. a panel with an integer dimN key (its slab is on a leading axis, which
        a 2-D frame does not have);
      * a rectangle that does not fit inside `shape`, rectangles that overlap, or panels without the
        min/max_fs/ss keys -- geometries this cannot split honestly, so they are left alone."""
    H, W = int(shape[0]), int(shape[1])
    rects = []
    try:
        for p in panels:
            for k in ("dim0", "dim1", "dim2", "dim3"):
                v = p.get(k)
                if v is not None and not isinstance(v, str) and float(v).is_integer():
                    return None
            rc = (int(p["min_ss"]), int(p["max_ss"]) + 1, int(p["min_fs"]), int(p["max_fs"]) + 1)
            if not (0 <= rc[0] < rc[1] <= H and 0 <= rc[2] < rc[3] <= W):
                return None
            if rc not in rects:
                rects.append(rc)
    except (KeyError, TypeError, ValueError):
        return None
    if len(rects) < 2:
        return None
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            if a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]:
                return None
    return rects


def panel_of(fs, ss, panels):
    fi, si = int(np.floor(fs)), int(np.floor(ss))   # fractional peak -> the pixel it lands in
    for i, p in enumerate(panels):
        if p["min_fs"] <= fi <= p["max_fs"] and p["min_ss"] <= si <= p["max_ss"]:
            return i
    return -1


def peaks_to_q(fs_arr, ss_arr, panels, clen_m, wavelength_A):
    """Detector peak (fs,ss) arrays -> (n,3) reciprocal vectors q [1/A].

    clen_m: detector distance [m] (per event). wavelength_A: [A].

    Peaks that land on no panel come back as NaN rows, and so do NON-FINITE inputs (NaN/inf
    fs or ss) -- they match no panel rather than raising, which the per-peak predecessor did
    via int(np.floor(x)). Callers are expected to filter, as stream_driver and frames_from_cxi do
    with glint.geom.q_rows_ok (which also drops a beam-centre row, |q| ~ 0); a bad coordinate is
    dropped exactly like an off-panel peak.
    """
    # Geometry lives in glint.geom._q_from_panels, shared with geom.peaks_to_q -- the two used to
    # carry independent copies and drifted (see that function). This wrapper only adapts the schema:
    # panels are a LIST here (first match wins, as panel_of did), the basis vectors are 3-vectors,
    # and z comes from the per-event clen rather than the .geom.
    specs = [dict(lo_fs=p["min_fs"], hi_fs=p["max_fs"], lo_ss=p["min_ss"], hi_ss=p["max_ss"],
                  off_fs=p["min_fs"], off_ss=p["min_ss"],
                  fsx=p["fs"][0], fsy=p["fs"][1], ssx=p["ss"][0], ssy=p["ss"][1],
                  cx=p["cx"], cy=p["cy"], res=p["res"], z=clen_m + p["coffset"])
             for p in panels]
    return _q_from_panels(fs_arr, ss_arr, specs, wavelength_A)


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


def _get_finder(name):
    """Peakfinder dispatch. v4/pf9 self-peak-find the image (portable numpy/cupy DRP finders)."""
    if name == "v4":
        from glint.peakfinder_v4 import peakfinder_v4; return peakfinder_v4
    if name == "pf9":
        from glint.peakfinder9 import peakfinder9; return peakfinder9
    if name == "pf8":
        raise SystemExit("frames_from_cxi: peakfinder='pf8' needs a per-pixel q map + radial.py (not yet "
                         "vendored). Use 'stored' to reuse the .cxi's own peakfinder8 peaks, or 'v4'/'pf9'.")
    raise SystemExit("frames_from_cxi: unknown peakfinder %r (use v4|pf9|pf8|stored)" % name)


def frames_from_cxi(cxi_path, geom_path, wavelength_A=None, n=0, min_peaks=6, data_key=None,
                    clen_scale=None, peakfinder="v4", top_n=0, ring_focus=None, per_panel=False,
                    event_axis=None, **pf_kw):
    """Self-contained GLINT front end: read a .cxi and bridge detector peaks to reciprocal q-vectors -- no
    CrystFEL peak-search stream in between. Returns (frames [(N,3) q in 1/A], images [{image,event}]).

    peakfinder: 'v4' (default) / 'pf9' -> self peak-find each image with the vendored DRP finder; 'stored'
    -> REUSE the .cxi's own peakfinder8/Cheetah peaks in /entry_1/result_1 (no redundant peak-find -- the
    efficient path when FindPeaksSFX/Cheetah already stored them); 'pf8' -> not yet vendored (needs a q-map).
    top_n: keep only the N strongest peaks per frame (0 = all; guards a finder that over-finds on background).
    per_panel: on a multi-panel slab, run the finder once per panel rectangle (slab_rects), so nothing it
    computes reaches across a panel seam. Off by default: one finder per panel costs a fixed overhead per panel.
    clen/photon_energy may be per-event h5 paths; clen_scale converts encoder units to metres (auto: >10 => mm).

    cxi_path may also be a CrystFEL .list/.lst of .cxi files (FindPeaksSFX's result); frames from all listed
    .cxi are concatenated so IndexGLINT is a drop-in for the .list that feeds CrystFELIndexer.

    LAYOUT (self peak-find). The image dataset goes through the SAME layout decision as integrate_cxi and
    the --peaks route's _load_image (``predict._leading_axis_is_events``), so the three cannot read one
    file differently. An ``(event, ss, fs)`` stack or a 2-D frame is one assembled frame per event, as
    before. An un-assembled ``(panel, ss, fs)`` stack is ONE event, and a 4-D ``(event, panel, ss, fs)``
    file is one panel stack per event: each slab is peak-found on its own and its peaks are mapped only
    through the panels the .geom's integer ``dimN`` keys put on that slab, then the slabs of an event are
    joined into one frame. Until this was added, ``data[i]`` of a panel stack was read as event ``i`` and
    every panel's peaks went through panel 0's corner and basis, silently; a 4-D file crashed in the
    finder. A panel stack with no slab mapping is refused by name, and so is the ambiguous case
    (leading axis == panel count, no per-event metadata). ``event_axis`` (True = events, False = panels,
    None = ask the file) is the override, ``--event-axis`` on the CLI. glint#148."""
    if str(cxi_path).endswith((".list", ".lst")):
        with open(cxi_path) as fh:
            paths = [ln.split()[0] for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
        frames, images = [], []
        for pth in paths:
            remaining = (n - len(frames)) if n else 0   # pass the REMAINING budget so we don't read whole files
            fr, im = frames_from_cxi(pth, geom_path, wavelength_A=wavelength_A, n=remaining, min_peaks=min_peaks,
                                     data_key=data_key, clen_scale=clen_scale, peakfinder=peakfinder,
                                     top_n=top_n, ring_focus=ring_focus, per_panel=per_panel,
                                     event_axis=event_axis, **pf_kw)
            frames += fr; images += im
            if n and len(frames) >= n:
                break
        return (frames[:n], images[:n]) if n else (frames, images)
    import h5py
    from glint.predict import _leading_axis_is_events, _panel_slab
    panels, glob = parse_geom(geom_path)
    # The slab mapping, built exactly as integrate_cxi builds it: all-or-nothing, multi-panel only.
    _slabs = [_panel_slab(p) for p in panels]
    panel_slabs = _slabs if len(panels) > 1 and all(s is not None for s in _slabs) else None
    n_slabs_geom = len(set(panel_slabs)) if panel_slabs is not None else 0
    data_key = data_key or glob.get("data", "/entry_1/data_1/data")
    clen_spec, en_spec, mask_key = glob.get("clen"), glob.get("photon_energy"), glob.get("mask")
    coff = float(glob.get("coffset", 0.0))
    f = h5py.File(cxi_path, "r")
    dropped = [0, 0]                                                # rows dropped by _q, frames they were in

    def _q(xarr, yarr, i, pans=panels, clean=True):                 # (fs,ss) peaks -> q for event i
        clen = _meta(clen_spec, f, i, 0.1)
        scale = clen_scale if clen_scale is not None else (0.001 if abs(clen) > 10 else 1.0)
        clen = clen * scale + coff
        wl = wavelength_A
        if wl is None:
            eV = _meta(en_spec, f, i, None)
            wl = lambda_from_eV(eV) if eV else None
        if wl is None:
            raise SystemExit("frames_from_cxi: no wavelength (geom photon_energy path or --wavelength)")
        # peaks_to_q keeps a NaN row for a peak on no panel (its streaming caller needs the positions);
        # these frames have no positional use for it, and handed on, ONE such row made hybrid_index drop
        # the whole frame without a word (glint review s7-05). Drop it here, with a beam-centre row
        # (|q| <= Q_FLOOR), by the same rule every indexer applies (glint.geom.q_rows_ok).
        q = peaks_to_q(np.asarray(xarr, float), np.asarray(yarr, float), pans, clen, wl)
        return _clean(q) if clean else q

    def _clean(q):                                                  # drop and count unusable rows
        kept = clean_q(q)
        if len(kept) < len(q):
            dropped[0] += len(q) - len(kept); dropped[1] += 1
        return kept

    def _report_dropped():
        if dropped[0]:                                              # a .geom that does not tile the data array
            sys.stderr.write(
                f"glint: {cxi_path}: dropped {dropped[0]} peak(s) in {dropped[1]} frame(s) that fall on no "
                f".geom panel or at |q| <= {Q_FLOOR:g} 1/A (the beam centre); the rest of each frame is kept.\n")

    frames, images = [], []
    if peakfinder == "stored":                                     # reuse the .cxi's own peakfinder8/Cheetah peaks
        if n_slabs_geom > 1:
            # A stored peak is an (x, y) and nothing else. Under a slab-mapped .geom the panel windows
            # are slab-LOCAL and overlap across slabs, so (x, y) does not say which slab it came from,
            # and peaks_to_q's first-match-wins put every peak through the slab-0 panels' geometry --
            # the self-peak-find defect below, on this branch (glint#148). Refuse rather than guess.
            raise NotImplementedError(
                f"{geom_path} maps its {len(panels)} panels onto {n_slabs_geom} data slabs (integer "
                f"dimN keys), so a stored peak's (x, y) is slab-local and does not name its slab: "
                f"mapping it would put every peak through the first matching panel's geometry "
                f"(glint#148). Use peakfinder='v4' or 'pf9', which peak-find each slab and map its "
                f"peaks through that slab's own panels.")
        rl = glob.get("peak_list", "/entry_1/result_1")
        px, py, npk = f[rl + "/peakXPosRaw"], f[rl + "/peakYPosRaw"], f[rl + "/nPeaks"]
        # top_n MUST work here too. It used to be applied only on the self-peak-find path below, so
        # with peakfinder='stored' -- the LUTE default -- `top_peaks` was a silent no-op: the config
        # validated and the peak list came through untruncated. Cheetah/peakfinder8 write the
        # intensities alongside the positions, so rank by them and match what v4/pf9 mean by
        # "strongest"; a flag must not change meaning when the peak SOURCE changes. Files without
        # the dataset fall back to stored order rather than failing.
        # Ranking needs intensities that actually VARY. experiments/cf_peaks.cxi carries
        # peakTotalIntensity = 1000.0 for all 16545 peaks (std 0, one unique value) -- a placeholder
        # some writers emit. Sorting a constant array yields an ARBITRARY subset unrelated to peak
        # strength, so a present-but-degenerate dataset is worse than an absent one: it looks like a
        # ranked selection and is not. Detect it and fall back to stored order.
        ipath = rl + "/peakTotalIntensity"
        pint = f[ipath] if (top_n and ipath in f) else None
        if pint is not None:
            probe = np.concatenate([np.asarray(pint[i, :int(npk[i])], float)
                                    for i in range(min(len(npk), 32))]) if len(npk) else np.empty(0)
            if probe.size == 0 or np.ptp(probe) == 0:
                sys.stderr.write(
                    f"glint: {ipath} is constant ({probe[0] if probe.size else 'empty'}) -- it carries "
                    f"no ranking information, so top_n={top_n} falls back to stored peak order.\n")
                pint = None
        nfr = min(n, px.shape[0]) if n else px.shape[0]
        for i in range(nfr):
            k = int(npk[i]); images.append({"image": cxi_path, "event": i})
            x, y = np.asarray(px[i, :k], float), np.asarray(py[i, :k], float)
            if top_n and k > top_n:
                if pint is not None:
                    keep = np.argsort(np.asarray(pint[i, :k], float))[::-1][:top_n]
                else:
                    keep = np.arange(top_n)
                x, y = x[keep], y[keep]
            q = _q(x, y, i) if len(x) >= min_peaks else np.empty((0, 3))
            frames.append(q if len(q) >= min_peaks else np.empty((0, 3)))     # min_peaks counts usable rows
        _report_dropped()
        return frames, images

    data = f[data_key]
    # LAYOUT: what ONE EVENT is in this dataset, decided as integrate_cxi and _load_image decide it
    # (glint#148). `stack` = each event is an un-assembled (slab, ss, fs) panel stack, peak-found slab by
    # slab; otherwise each event is one assembled 2-D frame -- the only reading this front end had, so a
    # (panel, ss, fs) stack came back as one "event" per panel, all mapped through panel 0's geometry.
    stack = False
    if data.ndim >= 4:
        if data.ndim > 4 or event_axis is False:
            raise NotImplementedError(
                f"{cxi_path}:{data_key} has shape {data.shape}; the self peak-find front end reads a 4-D "
                f"dataset as (event, panel, ss, fs) only, so it cannot honour "
                f"{'event_axis=False / --event-axis panel' if event_axis is False else 'more than 4 axes'} "
                f"here (glint#148).")
        nfr, stack = data.shape[0], True
    elif data.ndim == 3:
        stacked = data.shape[0] > 1 or event_axis is not None
        is_event = (_leading_axis_is_events(f, data, cxi_path, data_key, n_panels=len(panels),
                                            event_axis=event_axis, panel_slabs=panel_slabs)
                    if stacked else None)
        if is_event is False or (is_event is None and panel_slabs is not None):
            nfr, stack = 1, True                                   # ONE event: (slab, ss, fs)
        else:
            nfr = data.shape[0]                                    # (event, ss, fs), or a (1, ss, fs) frame
    else:
        nfr = 1
    H, W = data.shape[-2], data.shape[-1]
    if not stack and n_slabs_geom > 1:
        raise NotImplementedError(
            f"{cxi_path}:{data_key} {data.shape} is read as one assembled 2-D frame per event, but "
            f"{geom_path} maps its panels onto {n_slabs_geom} data slabs with integer dimN keys, so its "
            f"panel windows are slab-local and overlap: a peak's (fs, ss) names no slab and would go "
            f"through the first matching panel's geometry (glint#148). The data and the geometry "
            f"disagree about the layout.")
    if stack:
        nslab = data.shape[-3]
        if panel_slabs is None:
            if nslab > 1 and len(panels) > 1:
                raise NotImplementedError(
                    f"{cxi_path}:{data_key} {data.shape} reads as a stack of {nslab} PANELS per event, "
                    f"but none of the geometry's {len(panels)} panels maps to a slab, so nothing says "
                    f"whose corner and basis a slab's peaks take; reading slab i as event i put every "
                    f"panel's peaks through panel 0's geometry (glint#148). Declare each panel's slab "
                    f"in the .geom with an integer dimN key (e.g. `p1/dim0 = 1`, or `p1/dim1 = 1` "
                    f"under `dim0 = %` for 4-D data); or, if this file really is one frame per EVENT, "
                    f"pass event_axis=True / --event-axis event.")
            groups = {0: panels}       # one slab, or a one-panel geometry: slab 0 IS the frame (as _load_image)
        else:
            # every panel must address the data, checked up front as integrate_spots_stack does: a
            # mismatch is a geometry/data disagreement, not a frame with no peaks on that panel
            for p, sl in zip(panels, panel_slabs):
                if not 0 <= sl < nslab:
                    raise ValueError(
                        f"panel {p['name']}: slab {sl} does not address the {nslab}-slab stack "
                        f"{cxi_path}:{data_key} {data.shape} (glint#148)")
                if not (0 <= p["min_fs"] <= p["max_fs"] < W and 0 <= p["min_ss"] <= p["max_ss"] < H):
                    raise ValueError(
                        f"panel {p['name']}: window fs {p['min_fs']}..{p['max_fs']} x ss "
                        f"{p['min_ss']}..{p['max_ss']} does not address the slab shape {(H, W)}; in a "
                        f"slab-mapped .geom (integer dimN) min/max fs/ss lie WITHIN the panel's slab "
                        f"(glint#148)")
            groups = {sl: [p for p, ps in zip(panels, panel_slabs) if ps == sl]
                      for sl in sorted(set(panel_slabs))}
    else:
        groups = {None: panels}
    if n:
        nfr = min(n, nfr)
    good = None
    if mask_key and mask_key in f:
        m = f[mask_key]
        if stack:      # per-slab (slab, ss, fs) or one (ss, fs) for every slab; a per-event stack -> event 0's
            m = np.asarray(m[0] if m.ndim >= 4 else m)
        else:
            m = np.asarray(m[0] if m.ndim >= 3 else m)
        good = (m == int(str(glob.get("mask_good", "0")), 0))     # True = good pixel
    masks = {k: (None if good is None else good[k] if (stack and good.ndim == 3) else good) for k in groups}
    if ring_focus is not None:                                     # KNOWN-CELL: search only the powder-ring annuli
        cell6, qlow = ring_focus
        c0 = _meta(clen_spec, f, 0, 0.1); sc = clen_scale if clen_scale is not None else (0.001 if abs(c0) > 10 else 1.0)
        e0 = _meta(en_spec, f, 0, None); wl0 = wavelength_A or (lambda_from_eV(e0) if e0 else None)
        from glint.ring_mask import ring_qmask
        for k, grp in groups.items():                              # one canvas per slab: windows overlap across slabs
            rmask = ring_qmask(grp, c0 * sc + coff, wl0, cell6, (H, W), qlow=qlow)
            masks[k] = rmask if masks[k] is None else (masks[k] & rmask)
    finder = _get_finder(peakfinder)
    # per_panel on a multi-panel slab: one finder per panel rectangle, so no background ring, local-max
    # window or component reaches across a panel seam (peakfinder_v4.PerPanelFinder). One panel: unchanged.
    rects = slab_rects(panels, (H, W)) if per_panel and not stack else None
    if rects is not None:
        from glint.peakfinder_v4 import merge_panel_peaks
    for i in range(nfr):
        images.append({"image": cxi_path, "event": i})
        if not stack:
            img = np.asarray(data[i] if data.ndim >= 3 else data, np.float32)
            m = masks[None]
            if rects is None:
                pk = finder(img, mask=m, **pf_kw)
            else:
                pk = merge_panel_peaks([finder(img[a:b, c:d], mask=None if m is None else m[a:b, c:d],
                                               **pf_kw) for (a, b, c, d) in rects], rects)
            x, y = np.asarray(pk["x"]), np.asarray(pk["y"])
            if top_n and len(x) > top_n:                           # keep the strongest (guards over-finding)
                s = np.asarray(pk.get("intensity", pk.get("snr", np.zeros(len(x)))))
                keep = np.argsort(s)[::-1][:top_n]; x, y = x[keep], y[keep]
            q = _q(x, y, i) if len(x) else np.empty((0, 3))
            frames.append(q if len(q) >= min_peaks else np.empty((0, 3)))     # min_peaks counts usable rows
            continue
        # PANEL STACK: peak-find each slab on its own, map its peaks through ITS panels only, and join
        # the slabs into one frame for this event.
        stk = data[i] if data.ndim >= 4 else data
        xs, ys, ws, ks = [], [], [], []
        for k, grp in groups.items():
            pk = finder(np.asarray(stk[k], np.float32), mask=masks[k], **pf_kw)
            xk = np.asarray(pk["x"], float)
            xs.append(xk); ys.append(np.asarray(pk["y"], float))
            ws.append(np.asarray(pk.get("intensity", pk.get("snr", np.zeros(len(xk)))), float))
            ks.append(np.full(len(xk), k))
        x, y, w, kk = (np.concatenate(a) for a in (xs, ys, ws, ks))
        if top_n and len(x) > top_n:                               # strongest over the whole event, not per slab
            keep = np.argsort(w)[::-1][:top_n]; x, y, kk = x[keep], y[keep], kk[keep]
        if len(x) < min_peaks:
            frames.append(np.empty((0, 3)))
            continue
        q = np.empty((len(x), 3))
        for k, grp in groups.items():
            r = kk == k
            if r.any():
                q[r] = _q(x[r], y[r], i, grp, clean=False)   # raw, so rows stay aligned with x, y
        q = _clean(q)                                       # then drop unusable rows once per event
        frames.append(q if len(q) >= min_peaks else np.empty((0, 3)))
    _report_dropped()
    return frames, images
