"""Spot prediction + box integration: the bridge from a GLINT orientation to real I/sigma.

GLINT indexes (orientation M -> reciprocal rows a*,b*,c* = inv(M), q = hkl @ inv(M)); to MERGE we
need integrated intensities at PREDICTED reflection positions, not just the observed peaks. This is
the inverse of lute_bridge.peaks_to_q: enumerate hkl, gate by the Ewald excitation error (stills =
partials within a tolerance of the sphere), project each surviving rlp to a detector (panel, fs, ss),
then box-sum the image minus a local background -> (I, sigma). The result is a CrystFEL .stream whose
reflection rows carry REAL I/sigma/fs/ss, mergeable by partialator/process_hkl.

Conventions match the rest of fftindex exactly (so a predicted q inverts an observed one):
  reciprocal rows R (3x3, [a*;b*;c*], 1/A);  q = hkl @ R;  q = (s_hat - z_hat)/lambda;
  |q| = 2 sin(theta)/lambda = 1/d.  Flat-panel detector model identical to peaks_to_q
  (z = clen + coffset constant; fs/ss z-components ignored, as the forward bridge does).
"""
from __future__ import annotations
import os
import numpy as np

Z_HAT = np.array([0.0, 0.0, 1.0])


def recip_from_M(M):
    """GLINT orientation M (columns = real-space axes, A) -> reciprocal rows [a*;b*;c*] (1/A)."""
    return np.linalg.inv(np.asarray(M, float))


def _hkl_grid(R, qmax):
    """All integer hkl with |hkl @ R| <= qmax, bounded per-axis by qmax / |row|."""
    norms = np.linalg.norm(R, axis=1)
    H, K, L = (int(np.ceil(qmax / n)) + 1 for n in norms)
    h = np.arange(-H, H + 1); k = np.arange(-K, K + 1); l = np.arange(-L, L + 1)
    g = np.stack(np.meshgrid(h, k, l, indexing="ij"), -1).reshape(-1, 3)
    g = g[np.any(g != 0, axis=1)]                       # drop (0,0,0)
    q = g @ R
    keep = np.einsum("ij,ij->i", q, q) <= qmax * qmax
    return g[keep], q[keep]


def project_q(q, panels, clen_m, wavelength_A):
    """Project reciprocal vectors q (n,3, 1/A) to detector (fs, ss, panel_index).

    Inverse of peaks_to_q under the same flat-panel model. Returns (fs, ss, panel) arrays;
    entries that miss every panel (or back-scatter) are NaN / -1. Vectorised over q.
    """
    q = np.asarray(q, float)
    s_hat = wavelength_A * q + Z_HAT                     # scattered direction (|s|~1 near sphere)
    s_hat = s_hat / np.linalg.norm(s_hat, axis=1, keepdims=True)
    n = len(q)
    fs = np.full(n, np.nan); ss = np.full(n, np.nan); pan = np.full(n, -1, int)
    fwd = s_hat[:, 2] > 1e-6                             # forward-scattered only
    for pi, p in enumerate(panels):
        Zp = clen_m + p["coffset"]
        t = np.where(fwd, Zp / np.where(fwd, s_hat[:, 2], 1.0), np.nan)
        X = s_hat * t[:, None]                           # lab hit point (m)
        # res * X_xy - corner = [fs_vec_xy ss_vec_xy] @ [lf; ls]
        A = np.array([[p["fs"][0], p["ss"][0]], [p["fs"][1], p["ss"][1]]])
        rhs = p["res"] * X[:, :2] - np.array([p["cx"], p["cy"]])
        lfls = rhs @ np.linalg.inv(A).T                  # (n,2): local fs,ss offsets
        f = p["min_fs"] + lfls[:, 0]; s = p["min_ss"] + lfls[:, 1]
        on = fwd & (f >= p["min_fs"]) & (f <= p["max_fs"]) & (s >= p["min_ss"]) & (s <= p["max_ss"])
        take = on & (pan < 0)                             # first panel that catches it
        fs[take] = f[take]; ss[take] = s[take]; pan[take] = pi
    return fs, ss, pan


def predict_spots(M_or_R, panels, clen_m, wavelength_A, dmin=2.0, tol=0.006, is_recip=False):
    """Predict on-detector reflections for orientation M (or reciprocal rows R).

    dmin [A]: resolution limit (qmax = 1/dmin). tol [1/A]: half-width of the Ewald excitation-error
    gate (stills partiality window; ~mosaicity+bandwidth+1/size). Returns a structured array with
    fields h,k,l,fs,ss,panel,exc,res (res = 1/|q|, the d-spacing).
    """
    R = np.asarray(M_or_R, float) if is_recip else recip_from_M(M_or_R)
    qmax = 1.0 / dmin
    hkl, q = _hkl_grid(R, qmax)
    qn2 = np.einsum("ij,ij->i", q, q)
    exc = q[:, 2] + 0.5 * wavelength_A * qn2             # 0 on the Ewald sphere
    near = np.abs(exc) < tol
    hkl, q, exc, qn2 = hkl[near], q[near], exc[near], qn2[near]
    fs, ss, pan = project_q(q, panels, clen_m, wavelength_A)
    on = pan >= 0
    out = np.zeros(int(on.sum()), dtype=[("h", int), ("k", int), ("l", int),
                                         ("fs", float), ("ss", float), ("panel", int),
                                         ("exc", float), ("res", float)])
    out["h"], out["k"], out["l"] = hkl[on, 0], hkl[on, 1], hkl[on, 2]
    out["fs"], out["ss"], out["panel"] = fs[on], ss[on], pan[on]
    out["exc"] = exc[on]; out["res"] = 1.0 / np.sqrt(qn2[on])
    return out


BG_NSIG = 5.0                # MAD-clip half-width of the robust background mean; see _bg_clipmean
BG_MODES = ("clipmean", "median", "mean")


def _bg_clipmean(annpx, nsig=BG_NSIG):
    """Per-row MAD-clipped MEAN of the annulus -- the robust mean a box SUM needs (glint#131).

    The estimator: centre on the row median, scale by 1.4826*MAD, keep everything within
    ``nsig`` scales of the centre, average what is kept. Outliers (a neighbouring spot, a hot or
    saturated pixel) fall outside the window and are dropped; the surviving sample is the background
    itself, so its MEAN is what gets multiplied by nbox.

    THE FLOOR IS LOAD-BEARING. On counting data the MAD is quantised and is exactly 0 whenever more
    than half the annulus shares one value -- routine at low counts, universal below ~1 count/px.
    With scale 0 the window collapses onto the median and the estimator becomes the median again,
    which is the bug. So the scale is floored at the Poisson sigma of the median level,
    sqrt(max(med,1)), matching the counting model `sigma` already assumes one line below.

    Measured on 2209 pure-background boxes (Poisson, 49-px box, 168-px annulus), mean I where the
    true answer is 0 -- the plain mean is the unbiased reference:

        lambda      6.0      2.0      0.3     0.05
        median    +3.79    +0.62   +14.73   +2.49      <- shipped before this
        mean      -0.07    +0.15    -0.02   +0.05
        nsig=3    +0.93    +1.39    +0.04   +0.05      <- clips real background at high counts
        nsig=4    +0.05    +0.51    -0.01   +0.05
        nsig=5    -0.06    +0.17    -0.02   +0.05      <- chosen
        nsig=6    -0.07    +0.16    -0.02   +0.05

    and with the SAME background contaminated (box untouched, so the truth is still 0):

        contaminant                      median      mean     nsig=5
        9-px neighbouring spot, 800/px    -1.91  -2785.21      -0.10
        one saturated pixel (65535)       +2.89 -19112.70      -0.07

    i.e. nsig=5 is unbiased like the mean and more robust than the median. A symmetric 10% trimmed
    mean was also measured and is NOT a substitute (+4.72 at lambda=6, +4.98 at lambda=0.3):
    trimming both tails of a right-skewed count distribution biases the background low and I high.
    """
    med = np.median(annpx, axis=1)
    dev = np.abs(annpx - med[:, None])
    mad = np.median(dev, axis=1)
    scale = np.maximum(1.4826 * mad, np.sqrt(np.maximum(med, 1.0)))
    inlier = dev <= nsig * scale[:, None]        # NOT a `keep`: this selects PIXELS, never reflections
    cnt = inlier.sum(1)
    tot = np.where(inlier, annpx, 0.0).sum(1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), med)      # cnt==0 cannot happen; guard anyway


def integrate_spots(data, pred, half=3, gap=2, ring=3, bg_mode="clipmean"):
    """Box integrate a predicted reflection list against an assembled detector array `data` (ss,fs).

    Signal = sum over a (2*half+1)^2 box; background = robust mean of a surrounding annulus
    (gap..gap+ring) scaled to the box; I = signal - nbox*bg; sigma = sqrt(signal + nbox*bg)  (Poisson,
    gain=1). Returns (I, sigma, peak, bg_per_px) arrays aligned with `pred`. Out-of-frame -> 0.

    ``bg_mode`` picks the annulus estimator:
      ``"clipmean"``  MAD-clipped mean (default) -- the robust mean this docstring has always
                      claimed. See ``_bg_clipmean``.
      ``"median"``    the plain median, which is what shipped up to glint#131. It is a robust
                      LOCATION, but the quantity a box SUM must subtract is a MEAN, and on discrete
                      counts the median sits below it, so I comes out high by nbox*(mean - median)
                      on every reflection. Measured on pure Poisson background at lambda=6, 49-px
                      box: mean I = +3.79 +/- 0.46 where the truth is 0 (8.3 SE), against -0.07 for
                      the mean; the gap grows as the data get sparser (+14.7 at lambda=0.3). Kept
                      because every intensity GLINT produced before this change used it, so it is
                      the only way to reproduce those numbers.

                      ON REAL DATA THE OFFSET IS 13x THE SYNTHETIC ESTIMATE, and it games R_split
                      (the glint#129 A/B: cxil1015922 r0033, 2,547,616 reflection pairs, job
                      35836938). Cheetah-corrected frames have a near-zero float annulus with a
                      long right tail -- nothing like Poisson counts -- so the median under-
                      subtracts by +49.7 counts/reflection there, not +3.79. That offset pads
                      R_split's denominator without touching its numerator, so raw R_split REWARDS
                      the bias: ~90% of the median arm's 6.3-point R_split "advantage" vanished
                      when the measured offset was added back onto every merged clipmean I (33.0%
                      -> 27.2%), while CC* did not move under the shift. Two rules follow: never
                      judge a background estimator by raw R_split, and quote ``bg_mode`` next to
                      the CrystFEL version and flags with any merge number this route produced. At
                      matched intensity scale the real residual on that data was -0.005 CC* under
                      unity (concentrated in the weak high-resolution shells; inside repeat spread
                      under partiality) and +0.5-0.9 points R_split -- measured, single-dataset,
                      and NOT evidence that the median is the better estimator, since its raw
                      advantage is exactly the defect glint#131 identified.
      ``"mean"``      the plain mean: unbiased, and destroyed by one hot pixel in the annulus.
                      Diagnostic only -- it is the reference the other two are measured against.
    """
    if bg_mode not in BG_MODES:
        raise ValueError(f"bg_mode must be one of {BG_MODES}, got {bg_mode!r}")
    data = np.asarray(data)                              # NOT upcast: see the gather below
    H, W = data.shape
    n = len(pred)
    I = np.zeros(n); sig = np.zeros(n); peak = np.zeros(n); bgpp = np.zeros(n)
    R = half + gap + ring
    cf = np.rint(pred["fs"]).astype(int); cs = np.rint(pred["ss"]).astype(int)
    valid = (cs - R >= 0) & (cs + R < H) & (cf - R >= 0) & (cf + R < W)   # in-frame boxes only
    vi = np.where(valid)[0]
    if len(vi):                                          # vectorized gather (constant masks hoisted): ~7x the loop, bit-exact
        dy, dx = np.mgrid[-R:R + 1, -R:R + 1]
        rad = np.maximum(np.abs(dy), np.abs(dx))          # Chebyshev (square rings)
        box = rad <= half
        ann = (rad > half + gap) & (rad <= half + gap + ring)
        nbox = int(box.sum())
        # Gather in the detector's NATIVE dtype, then upcast only the patches. Upcasting the whole
        # image first is O(H*W) per call for O(n*(2R+1)^2) of work -- on a 16 Mpix frame that single
        # cast measured ~546 ms, i.e. ~99% of this function; doing it here instead is bit-identical
        # and ~105x faster (4 Mpix: 2.5x). Accumulation stays float64, so results are unchanged.
        patch = data[cs[vi, None, None] + dy[None], cf[vi, None, None] + dx[None]].astype(float, copy=False)
        boxpx = patch[:, box]; annpx = patch[:, ann]
        if not ann.any():
            bg = np.zeros(len(vi))
        elif bg_mode == "clipmean":
            bg = _bg_clipmean(annpx)
        elif bg_mode == "median":
            bg = np.median(annpx, axis=1)
        else:
            bg = annpx.mean(1)
        sig_sum = boxpx.sum(1)
        I[vi] = sig_sum - nbox * bg
        sig[vi] = np.sqrt(np.maximum(sig_sum + nbox * np.maximum(bg, 0.0), 1.0))
        peak[vi] = boxpx.max(1); bgpp[vi] = bg
    return I, sig, peak, bgpp


def integrate_spots_stack(stack, pred, panels, half=3, gap=2, ring=3, bg_mode="clipmean"):
    """Box-integrate against an UN-ASSEMBLED ``(panel, ss, fs)`` stack, slab-locally (glint#148).

    Each reflection is integrated on ITS OWN panel's slab (``panels[i]['slab']``, from the .geom's
    integer ``dimN`` key) at its ``fs``/``ss`` as-is: ``project_q`` emits data-array coordinates
    (panel window + local offset), and in a slab-mapped .geom the window addresses WITHIN the
    slab, so no coordinate shift is needed. Panels sharing a slab (asics tiling one module) each
    read their own window of it.

    No assembled canvas is ever built, so no pixel is invented and the gap question dissolves.
    Each panel is CROPPED to its own window before integrating, so the in-frame gate is the panel
    edge -- a reflection whose box would leave its panel is edge-gated (sigma=0, dropped by the
    caller's keep filter) exactly like a frame-edge reflection on an assembled frame, and that
    includes the boundary between two asics packed into one slab: array adjacency there is
    packing, not geometry, so no box ever reads a neighbouring panel's pixels. Returns
    (I, sigma, peak, bg) aligned with ``pred``; raises on a panel whose slab or window does not
    address the stack, because that is a geometry/data mismatch, not a measurement of zero.
    """
    stack = np.asarray(stack)
    # Validate EVERY panel against the stack up front, reflections or not: a slab or window that
    # does not address the data is a geometry/data mismatch, and it must not depend on where this
    # frame's reflections happened to land whether it is caught.
    for pi, p in enumerate(panels):
        s = p.get("slab")
        if s is None or not 0 <= s < stack.shape[0]:
            raise ValueError(
                f"panel {p.get('name', pi)}: slab {s} does not address the {stack.shape[0]}-slab "
                f"stack -- every panel needs an integer dimN key matching the data layout "
                f"(glint#148)")
        if not (0 <= p["min_fs"] <= p["max_fs"] < stack.shape[2]
                and 0 <= p["min_ss"] <= p["max_ss"] < stack.shape[1]):
            # the COMPLETE ordered window, not just the upper endpoints: a negative or inverted
            # bound does not address the slab either, and letting it through would surface as
            # silently dropped reflections instead of the promised mismatch error
            raise ValueError(
                f"panel {p.get('name', pi)}: window fs {p['min_fs']}..{p['max_fs']} x "
                f"ss {p['min_ss']}..{p['max_ss']} does not address the slab shape "
                f"{stack.shape[1:]} -- in a slab-mapped .geom (integer dimN) min/max fs/ss must "
                f"be an ordered range WITHIN the panel's slab, not a virtual assembled plane "
                f"(glint#148)")
    n = len(pred)
    I = np.zeros(n); sig = np.zeros(n); peak = np.zeros(n); bgpp = np.zeros(n)
    for pi, p in enumerate(panels):
        m = pred["panel"] == pi
        if not m.any():
            continue
        # Crop to THIS panel's window and translate the predictions into it, so integrate_spots'
        # in-frame gate is the PANEL edge -- not the slab edge. Handing it the whole slab let a
        # box near an intra-slab asic boundary gather signal/background from the neighbouring
        # panel's pixels, which may be a physically separated or differently oriented ASIC
        # (Copilot review of #157); array adjacency within a slab is packing, not geometry.
        win = stack[p["slab"], p["min_ss"]:p["max_ss"] + 1, p["min_fs"]:p["max_fs"] + 1]
        loc = pred[m].copy()
        loc["fs"] = loc["fs"] - p["min_fs"]; loc["ss"] = loc["ss"] - p["min_ss"]
        I[m], sig[m], peak[m], bgpp[m] = integrate_spots(win, loc, half=half,
                                                         gap=gap, ring=ring, bg_mode=bg_mode)
    return I, sig, peak, bgpp


_RCOL = "   h    k    l          I   sigma(I)   peak  background  fs/px  ss/px panel\n"


def _header(geom_text):
    """CrystFEL stream header -- ONE definition, shared with the orientation-only writer.

    It lived here while `stream.py` had no header block at all, which is exactly how the two
    drifted: this writer's output was readable and that one's was not."""
    from glint.stream import _header as _shared      # stream.py imports nothing from here
    return _shared(geom_text, generator="GLINT (fftindex predict+integrate)")


def _cell_line(M):
    M = np.asarray(M, float)
    a, b, c = (np.linalg.norm(M[:, i]) for i in range(3))
    def ang(i, j):
        u, v = M[:, i], M[:, j]
        return np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
    Minv = np.linalg.inv(M)
    lines = [f"Cell parameters {a/10:.5f} {b/10:.5f} {c/10:.5f} nm, "
             f"{ang(1,2):.3f} {ang(0,2):.3f} {ang(0,1):.3f} deg"]
    for nm, v in zip(("astar", "bstar", "cstar"), Minv):
        lines.append(f"{nm} = {v[0]*10:+.7f} {v[1]*10:+.7f} {v[2]*10:+.7f} nm^-1")
    return "\n".join(lines)


def _fmt_scalar(v):
    """Serialise a provenance value for a `key = value` stream line.

    Must survive NUMPY scalars, which is what these values actually are in practice: `np.bool_` is NOT
    a subclass of `bool` (so a plain isinstance(v, bool) test misses it and emits 'True'/'False'
    instead of 1/0), and a comparison like `python_float < np.float64` returns exactly that type.
    np.float64 IS a float subclass, but np.float32 is not, and formatting it as a plain value prints
    its full binary expansion. Checked bool-first because np.bool_ is an integer-like otherwise."""
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def _write_chunk(f, serial, r, panel_name="p0", photon_eV=9392.7, clen_m=0.15, panel_names=None,
                 pk_bounds=None):
    """Write ONE chunk of a CrystFEL .stream (format 2.3). Shared by the batch writer
    (write_stream_integrated) and the incremental one (StreamWriter) so the two cannot drift apart.

    PER-FRAME DRIFT / PROVENANCE. A streaming run's detector geometry and cell are not constants: the
    live GeomRefiner keeps re-solving (clen, dfs, dss) as frames arrive, and an adaptive-relock driver
    can switch cells mid-run. Since merging happens OFFLINE, the merger has to learn the state that was
    in effect for EACH frame -- a run-level summary would apply a late-run correction to early-run data.
    So every field below may be supplied per-result, and each falls back to the previous constant when
    absent (so existing callers are byte-identical):
      det_shift_mm  (x, y) mm  -> predict_refine/det_shift   [was hardcoded 0.000/0.000]
      clen_m        m          -> average_camera_length      [was the run-level kwarg]
      lattice_type/centering/unique_axis                     [were hardcoded triclinic/P/*]
    GLINT-specific provenance that has no CrystFEL field is emitted under a ``glint/`` prefix, the same
    convention CrystFEL uses for its own namespaced keys (``predict_refine/...``); readers skip
    unrecognised ``key = value`` lines, so this stays parseable while a GLINT-aware merger can use it.
    Returns True if the chunk carried an indexed crystal."""
    M = r.get("M")
    valid = M is not None and abs(np.linalg.det(np.asarray(M, float))) >= 1.0
    f.write("----- Begin chunk -----\n")
    f.write(f"Image filename: {r.get('image', 'glint.cxi')}\n")
    f.write(f"Event: //{r.get('event', 0)}\n")
    f.write(f"Image serial number: {serial}\n")
    f.write("hit = 1\n")
    f.write(f"indexed_by = {'file' if valid else 'none'}\n")   # 'file' = externally-supplied orientation
    f.write(f"photon_energy_eV = {photon_eV:.2f}\n")
    f.write("beam_divergence = 0.00e+00 rad\nbeam_bandwidth = 1.00e-08 %\n")
    f.write(f"average_camera_length = {float(r.get('clen_m', clen_m)):.6f} m\n")
    # OBSERVED peaks. Optional, but it is what makes a chunk re-indexable downstream: the reflection
    # rows below are PREDICTED under the orientation the indexer chose, so they cannot rescue a frame
    # whose orientation was itself the problem -- only the observed peaks can. r["peaks"] is (n,3) of
    # fs, ss, intensity; r["peaks_invd"] the matching |q| in A^-1 (written as nm^-1, x10).
    pks = r.get("peaks")
    npk = 0 if pks is None else len(pks)
    f.write(f"num_peaks = {npk}\nnum_saturated_peaks = 0\n")
    f.write("Peaks from peak search\n  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel\n")
    if npk:
        invd = r.get("peaks_invd")
        invd = np.zeros(npk) if invd is None else np.asarray(invd, float) * 10.0
        pk_fs, pk_ss, pk_I = (np.asarray(pks, float)[:, j].tolist() for j in range(3))
        il = invd.tolist()
        prow = "%7.2f %7.2f %10.5f %10.2f %s\n"
        # Observed peaks have real fs/ss, so they get the panel they actually LAND on. The old
        # fixed `panel_name` put every peak on 'p0', which on a 64-panel Jungfrau geometry is both
        # a panel CrystFEL warns does not exist and, where it does, the wrong one.
        from glint.stream import panel_at
        f.write("".join([prow % (pk_fs[j], pk_ss[j], il[j], pk_I[j],
                                 panel_at(pk_bounds, pk_fs[j], pk_ss[j], panel_name)
                                 if pk_bounds else panel_name)
                         for j in range(npk)]))
    f.write("End of peak list\n")
    if valid:
        pred, I, sg = r.get("pred"), r.get("I"), r.get("sigma")
        pk, bg = r.get("peak"), r.get("bg")
        nref = len(pred) if pred is not None else 0
        dres = float(pred["res"].min()) if pred is not None and nref else 2.0   # A
        dx, dy = r.get("det_shift_mm", (0.0, 0.0))
        f.write("--- Begin crystal\n" + _cell_line(M) + "\n")
        f.write(f"lattice_type = {r.get('lattice_type', 'triclinic')}\n"
                f"centering = {r.get('centering', 'P')}\n"
                f"unique_axis = {r.get('unique_axis', '*')}\n")
        f.write("profile_radius = 0.00200 nm^-1\n")
        f.write(f"predict_refine/det_shift x = {float(dx):.3f} y = {float(dy):.3f} mm\n")
        for key in ("cell_id", "lock_generation", "matched_frac", "low_confidence",
                    "dclen_m", "geom_n_solves", "frame_no"):
            if key in r and r[key] is not None:
                f.write(f"glint/{key} = {_fmt_scalar(r[key])}\n")
        f.write(f"diffraction_resolution_limit = {10.0/dres:.2f} nm^-1 or {dres:.2f} A\n")
        f.write(f"num_reflections = {nref}\n")
        f.write("num_saturated_reflections = 0\nnum_implausible_reflections = 0\n")
        f.write("Reflections measured after indexing\n" + _RCOL)
        if pred is not None and nref:
            # predict_spots tags each reflection with the panel it landed on. With panel_names given,
            # name each row by ITS OWN panel -- on a real multi-panel detector a single scalar name
            # mislabels every reflection that is not on panel 0, and partialator keys per-panel
            # geometry off that name. Falls back to the scalar when no map is supplied.
            names = None
            if panel_names is not None and "panel" in (pred.dtype.names or ()):
                names = [panel_names[p] if 0 <= p < len(panel_names) else panel_name
                         for p in pred["panel"].tolist()]
            # Formatted via %-interpolation over .tolist() into ONE joined write, not an f-string plus
            # f.write per reflection. This is on the live integrate path, so it is not free: measured
            # 8.98 -> 2.60 ms for a dense 2212-reflection chunk (3.5x). At the driver's default
            # tol=0.002 chunks are far smaller, but on a dense run this still costs milliseconds per
            # frame against a ~0.21 ms/frame known-cell path (driver default B=64) -- if that
            # matters, the next step is to
            # hand formatting to a writer thread rather than to micro-optimise further.
            h, k, l = pred["h"].tolist(), pred["k"].tolist(), pred["l"].tolist()
            fsl, ssl = pred["fs"].tolist(), pred["ss"].tolist()
            Il, sl = np.asarray(I).tolist(), np.asarray(sg).tolist()
            pl, bl = np.asarray(pk).tolist(), np.asarray(bg).tolist()
            row = "%4d %4d %4d %10.2f %10.2f %6.1f %10.2f %6.1f %6.1f %s\n"
            f.write("".join([row % (h[j], k[j], l[j], Il[j], sl[j], pl[j], bl[j], fsl[j], ssl[j],
                                    names[j] if names is not None else panel_name)
                             for j in range(nref)]))
        f.write("End of reflections\n--- End crystal\n")
    f.write("----- End chunk -----\n")
    return valid


def write_stream_integrated(results, path, panel_name="p0", geom_text=None,
                            photon_eV=9392.7, clen_m=0.15, panel_names=None):
    """results: list of {image,event,M, pred (structured), I, sigma, peak, bg}. Writes a CrystFEL
    .stream (format 2.3) with REAL integrated reflection rows, mergeable by partialator/process_hkl.
    Chunk layout mirrors CrystFEL's own writer (peak-list block + crystal metadata) -- the reader
    rejects an under-specified chunk as 'incomplete'. Per-result drift/provenance fields are optional;
    see _write_chunk."""
    from glint.stream import panel_bounds
    pk_bounds = panel_bounds(geom_text)          # parsed ONCE, not per peak
    n_idx = 0
    with open(path, "w") as f:
        f.write(_header(geom_text))
        for serial, r in enumerate(results, 1):
            n_idx += bool(_write_chunk(f, serial, r, panel_name, photon_eV, clen_m, panel_names,
                                       pk_bounds))
    return n_idx


class StreamWriter:
    """Append-as-you-go CrystFEL .stream writer -- the offline-merge handoff for a LIVE run.

    write_stream_integrated needs every result in memory at once, which a streaming driver cannot
    supply: it sees each frame exactly once, integrates it, and drops the pixels. This writes the
    header on open and one chunk per frame thereafter, so memory is O(1) in run length. Flushed every
    ``flush_every`` chunks so a downstream/monitoring reader sees data before the run ends.

    ON A KILLED RUN the file is NOT guaranteed to end at a chunk boundary: the underlying buffer
    flushes when it fills, which is mid-chunk most of the time (measured 194/200 at the default
    flush_every=32). Every individual f.write is one whole line, so the tail is a truncated chunk
    rather than a truncated line -- a reader that tolerates an unterminated final chunk is fine, one
    that requires the End-of-chunk marker is not. ``flush_every=1`` makes each chunk atomic at the
    cost of a syscall per frame.

        w = StreamWriter(path, geom_text=..., clen_m=..., photon_eV=...)
        w.write(record)          # per frame; same dict shape write_stream_integrated takes
        w.close()                # or use as a context manager
    """

    def __init__(self, path, geom_text=None, panel_name="p0", photon_eV=9392.7, clen_m=0.15,
                 flush_every=32, panel_names=None):
        self.path = str(path)
        self.panel_name, self.photon_eV, self.clen_m = panel_name, float(photon_eV), float(clen_m)
        self.panel_names = list(panel_names) if panel_names is not None else None
        self.flush_every = int(flush_every)
        self.n_chunks = 0
        self.n_indexed = 0
        from glint.stream import panel_bounds
        self.pk_bounds = panel_bounds(geom_text)     # parsed once at open, not per frame
        self._f = open(self.path, "w")
        self._f.write(_header(geom_text))

    def write(self, r):
        if self._f is None:
            raise ValueError("StreamWriter is closed")
        self.n_chunks += 1
        if _write_chunk(self._f, self.n_chunks, r, self.panel_name, self.photon_eV, self.clen_m,
                        self.panel_names, self.pk_bounds):
            self.n_indexed += 1
        if self.flush_every and self.n_chunks % self.flush_every == 0:
            self._f.flush()
        return self.n_chunks

    def close(self):
        if self._f is not None:
            self._f.flush(); self._f.close(); self._f = None
        return self.n_indexed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _canonical_axes(M):
    """Reorder real-space axes (columns of ``M``) to (long, long, short) -- a per-frame-consistent cell
    setting so tetragonal/orthorhombic reflections co-merge under the point group (c = the unique short
    axis, which a 422/mmm merge does NOT absorb, unlike a<->b and the Friedel/handedness ambiguity).
    Idempotent. Both ``write_fromfile`` (the CrystFEL handoff) and ``integrate_cxi`` (native merge) use it,
    so the two merge paths share one setting."""
    Ar = np.asarray(M, float)
    o = np.argsort(np.linalg.norm(Ar, axis=0))                 # shortest axis first
    return Ar[:, [o[1], o[2], o[0]]]                           # -> (long, long, short)


def write_fromfile(results, path, lattice_code="aP"):
    """Emit a CrystFEL ``--indexing=file`` solution file -- the refined-merge handoff. GLINT supplies
    the orientation; CrystFEL's own prediction-refinement imposes the lattice symmetry (run with a loose
    ``--tolerance``), which on real data merges better than either freezing GLINT's raw orientation
    (--no-refine) or pre-symmetrising the cell. One line per indexed frame:

        <image> //<event> a*x a*y a*z b*x b*y b*z c*x c*y c*z shift_x shift_y <lattice_code>

    with the reciprocal cell in nm^-1 and axes reordered to (long, long, short) so the standard setting
    matches the lattice code (e.g. ``tPc`` for tetragonal lysozyme; ``--tolerance=10,10,10,3`` recommended).
    """
    rows = []
    for r in results:
        M = r.get("M")
        if M is None:
            continue
        Are = _canonical_axes(M)                                   # (long, long, short) canonical setting
        Br = np.linalg.inv(Are).T * 10.0                           # reciprocal a*,b*,c* in nm^-1 (1/A -> 1/nm)
        v = Br[:, 0].tolist() + Br[:, 1].tolist() + Br[:, 2].tolist()
        ev = r.get("event", "")
        rows.append("%s //%s %s 0.0 0.0 %s"
                    % (r.get("image", "glint.cxi"), ev, " ".join("%.7f" % x for x in v), lattice_code))
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")
    return len(rows)


def _panel_slab(p):
    """The panel's data-array SLAB index from its CrystFEL ``dimN`` keys, or None.

    In a 3-D data layout -- ``(panel, ss, fs)``, or 4-D with an event axis in front -- each panel
    names its place along the extra leading axis with an INTEGER dim entry (``p1/dim0 = 1``);
    ``%`` marks the event axis and ``ss``/``fs`` the in-slab axes, and ``parse_geom`` already
    keeps all of them on the panel dict (integers as floats, the rest as strings). Exactly one
    integer entry is the slab; none means a plain 2-D layout; more than one describes a layout
    this flat-panel model does not (refused downstream by returning None). glint#148."""
    ints = [int(v) for k in ("dim0", "dim1", "dim2", "dim3")
            for v in (p.get(k),)
            if v is not None and not isinstance(v, str) and float(v).is_integer()]
    return ints[0] if len(ints) == 1 else None


def panels_from_geom(geom):
    """Convert an ``fftindex.geom.parse_geom()`` result into the panel dicts predict_spots/project_q
    want (same flat-panel model, different key names). Returns (panels, clen_m).

    ``slab`` is the panel's index along a 3-D dataset's leading axis, from its integer ``dimN``
    key, or None for a plain 2-D layout -- what lets ``integrate_spots_stack`` integrate an
    un-assembled panel stack slab-locally instead of refusing it (glint#148)."""
    g = geom.get("global", {})
    clen = float(g.get("clen", 0.1)); coff = float(g.get("coffset", 0.0)); res_g = float(g.get("res", 1.0))
    panels = []
    for nm, p in geom["panels"].items():
        panels.append(dict(name=nm, fs=np.array([p["fsx"], p["fsy"]]), ss=np.array([p["ssx"], p["ssy"]]),
                           res=float(p.get("res", res_g)), cx=float(p["corner_x"]), cy=float(p["corner_y"]),
                           coffset=float(p.get("coffset", coff)),
                           min_fs=int(p["min_fs"]), max_fs=int(p["max_fs"]),
                           min_ss=int(p["min_ss"]), max_ss=int(p["max_ss"]),
                           slab=_panel_slab(p)))
    return panels, clen


def _event_index(event):
    """CrystFEL ``Event://N`` value -> integer stack index.

    ``geom.read_crystfel_peaks`` int()s what it can and keeps anything else as the raw STRING, so a
    multi-dimensional event id (``entry_1//7``) arrives here as text. CrystFEL's own form puts the
    frame number last, so take the trailing field; refuse rather than guess if there is no number in
    it, because the alternative is silently integrating the wrong frame -- which is glint#136."""
    if event is None or event == "":
        return 0
    try:
        return int(event)
    except (TypeError, ValueError):
        pass
    tail = str(event).strip().strip("/").rsplit("/", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        raise ValueError(f"cannot read a frame index out of Event {event!r}") from None


# PER-EVENT DATASETS. A stacked .cxi carries one row per EVENT in these alongside the images, so
# their length is a positive statement of how many events the file holds -- unlike the image array's
# leading axis, which is what we are trying to interpret. psocake/btx write `nPeaks` and the LCLS
# block; `experiment_identifier` is the CXI standard's own per-entry field.
_EVENT_COUNT_PATHS = ("/entry_1/result_1/nPeaks", "/entry_1/result_1/peakXPosRaw",
                      "/LCLS/eventNumber", "/LCLS/fiducial", "/LCLS/timestamp",
                      "/entry_1/experiment_identifier")


def _event_count(f):
    """Events in an open HDF5 file according to its own per-event metadata, or None if it says
    nothing. Cheap: membership tests plus a shape read, no data."""
    for p in _EVENT_COUNT_PATHS:
        try:
            d = f[p]
        except (KeyError, OSError):
            continue
        shape = getattr(d, "shape", None)
        if shape:                                    # a dataset with at least one axis
            return int(shape[0])
    return None


def _leading_axis_is_events(f, d, path, data_path, n_panels=1, event_axis=None, panel_slabs=None):
    """What the LEADING axis of a stacked (>= 3-D, ``shape[0] > 1``) image dataset means: True for
    ``(event, ...)``, False for ``(panel, ss, fs)``.

    THE one layout decision, shared by ``_load_image`` (the --peaks route) and ``integrate_cxi`` (the
    --images route) so the two cannot read the same file differently. Rules, in order -- the history
    behind each is in ``_load_image``'s docstring:

      1. ``event_axis=True/False``  explicit, wins over everything.
      2. 4-D and higher             EVENTS: the leading axis is the geometry's ``%`` dimension; the
                                    slab-count inference below is reserved for 3-D data, where the
                                    leading axis is the ambiguous one.
      3. slab-mapped geometry       ``panel_slabs`` (integer dimN keys, glint#148) DECLARE the 3-D
                                    layout as PANELS, full stop.
      4. per-event metadata         the file's own event count: equal to the leading axis -> EVENTS,
                                    different -> PANELS.
      5. one-panel geometry, or leading axis != n_panels -> EVENTS: it cannot be a panel stack.
      6. otherwise                  genuinely ambiguous -> ValueError naming both readings and the
                                    override. Never silently pick (glint#136).
    """
    is_event = event_axis
    n_slabs = (max(panel_slabs) + 1) if panel_slabs else None
    if is_event is None and getattr(d, "ndim", 0) >= 4:
        # 4-D and higher: the leading axis IS the event axis (the geometry's '%' dimension);
        # slab-count inference below is reserved for 3-D data, where the leading axis is the
        # ambiguous one. Without this, a 4-D file whose EVENT count happens to equal the slab
        # count would be classified as a panel stack and returned whole to the 2-D integrator
        # (Copilot review of #157).
        is_event = True
    if is_event is None and n_slabs is not None:
        # The .geom's integer dimN keys DECLARE the 3-D layout: the leading axis is the panel
        # axis, full stop. (A 3-D EVENT stack cannot coexist with a slab-mapped multi-panel
        # geometry -- its per-slab windows overlap, so a single 2-D frame per event describes
        # nothing; a real event series under this geometry is 4-D and is claimed above.) This
        # must come before the metadata step and before rule 5: with asics sharing modules the
        # slab count differs from the panel count, so "leading axis != n_panels -> events"
        # read a 2-slab/4-panel stack as events and integrated slab 0's pixels for every panel
        # (glint#148), and a per-event array that happens to match the leading axis is
        # circumstance, while the dims are a statement. A leading axis that does not match the
        # mapping is a geometry/data MISMATCH, and integrate_spots_stack raises it by name --
        # guessing events there instead would integrate wrong pixels silently.
        is_event = False
    if is_event is None:
        n_ev = _event_count(f)
        if n_ev is not None:
            is_event = (n_ev == d.shape[0])          # the file's own per-event metadata decides
        elif n_panels <= 1 or d.shape[0] != n_panels:
            is_event = True                          # cannot be a panel stack
        else:
            raise ValueError(
                f"{path}:{data_path} has a leading axis of {d.shape[0]}, which equals the "
                f"geometry's panel count, and the file carries no per-event metadata "
                f"({', '.join(_EVENT_COUNT_PATHS)}) to settle it. It is either {d.shape[0]} "
                f"EVENTS of an assembled frame or {n_panels} PANELS of one event, and picking "
                f"wrong integrates the wrong pixels silently -- which is glint#136. Pass "
                f"event_axis=True (events) or event_axis=False (panels), or --event-axis "
                f"event|panel on the CLI, to say which it is.")
    return bool(is_event)


def _load_image(path, data_path, event=0, n_panels=1, event_axis=None, panel_slabs=None):
    """Read the frame for THIS event out of an image file at the geom ``data`` path.

    The layout decision itself is ``_leading_axis_is_events`` (shared with ``integrate_cxi``); what
    follows is the history that shaped its rules and how each reading is then honoured.

    EVENT-AWARE since glint#136. This used to end ``a = a[0] if a.shape[0] > 1 else a[0]`` -- both
    branches index 0 -- so on a stacked multi-event ``.cxi`` every result in the run was handed
    event 0's pixels, silently, and ``integrate_frames`` then produced real-looking I/sigma from the
    wrong image.

    WHAT THE LEADING AXIS OF A 3-D DATASET MEANS is not decidable from the shape: the old docstring
    called it ``(event|panel, ss, fs)`` and both readings occur. Nor is it decidable by comparing
    that axis to the geometry's panel count, which was this function's first attempt and is wrong by
    COINCIDENCE -- a 4-event .cxi under a 4-panel geometry matches, and every frame in the run would
    silently load slab 0 again, which is glint#136 exactly. So the file is ASKED, and where it does
    not answer the ambiguity is refused rather than guessed:

      1. ``event_axis=True/False``  explicit, wins over everything.
      2. per-event metadata          if the file carries a per-event dataset (``nPeaks``,
                                     ``LCLS/eventNumber``, ``experiment_identifier``, ...) its length
                                     is the event count. Equal to the leading axis -> EVENTS.
                                     Different -> the file knows its event count and this axis is
                                     not it, so PANELS.
      3. one-panel geometry          -> EVENTS. A one-panel geometry cannot describe a panel stack.
      4. leading axis != n_panels    -> EVENTS. It cannot be a panel stack either.

    A PANEL reading is honoured two ways: a ONE-PANEL geometry takes slab 0 (the whole detector),
    and a slab-mapped multi-panel geometry (``panel_slabs`` from integer ``dimN`` keys, glint#148)
    gets the WHOLE 3-D stack back, which ``integrate_frames`` integrates slab-locally through
    ``integrate_spots_stack``. A multi-panel stack WITHOUT the slab mapping is still refused --
    there is no mapping from slab index to panel, and guessing one integrates the wrong pixels
    silently -- see the comment at the raise.
      5. otherwise                   ``n_panels > 1`` and the axis matches it and the file said
                                     nothing -> genuinely ambiguous, so RAISE, naming both readings
                                     and the override. Never silently pick.

    An event past the end of an event stack RAISES instead of quietly falling back to frame 0 --
    that silence is what made the original defect invisible. Reads one frame, not the whole stack."""
    import h5py
    with h5py.File(path, "r") as f:
        try:
            d = f[data_path]
        except KeyError:
            # The bare h5py KeyError names neither the file nor where the path came from -- and
            # the path is usually not the user's own choice: it is the .geom's `data =` key
            # (parse_geom forwards it since glint#143), or the /data/data fallback when the .geom
            # has none. Name all of that, plus what the file actually holds.
            cands = []
            def _c(name, obj):
                if isinstance(obj, h5py.Dataset) and getattr(obj, "ndim", 0) >= 2:
                    cands.append("/" + name)
                    if len(cands) >= 8:
                        return True                  # any non-None return stops the visit
            f.visititems(_c)
            raise KeyError(
                f"{path} has no dataset at '{data_path}'. That path comes from the .geom's "
                f"`data = <path>` key when it has one, --data-path on the CLI, or the /data/data "
                f"default -- set whichever applies to where this file keeps its frames. "
                f"Image-like datasets found here: {', '.join(cands) if cands else 'none'}.") from None
        stacked = getattr(d, "ndim", 0) >= 3 and d.shape[0] > 1
        is_event = (_leading_axis_is_events(f, d, path, data_path, n_panels, event_axis, panel_slabs)
                    if stacked else None)             # the decision is only consumed when stacked
        if stacked and is_event:
            ev = _event_index(event)
            if not 0 <= ev < d.shape[0]:
                raise IndexError(
                    f"{path}:{data_path} is a stack of {d.shape[0]} frames but this result asks for "
                    f"event {event!r}; for stacked multi-event .cxi use integrate_cxi (the --images "
                    f"route), which reads per-event clen/energy as well (glint#136)")
            a = np.asarray(d[ev], np.float32)        # ONE frame, not the whole stack
        elif stacked:
            # PANEL STACK. Handing back slab 0 is what this function did before glint#136, and it
            # was WRONG for any multi-panel geometry -- not a behaviour worth preserving. Measured
            # on a 2-panel tiled geometry: a real 4500-count spot on panel 1 comes back
            # I=0.0 sigma=0.0 peak=0.0 from a slab-0 frame -- silently zero, not missing. With the
            # .geom's integer dimN keys mapping each panel to its slab (panel_slabs), the whole
            # stack goes back to integrate_frames for slab-local integration instead
            # (integrate_spots_stack, glint#148). WITHOUT that mapping there is no way to tell the
            # slabs apart, and guessing one -- by declaration order, say -- integrates the wrong
            # pixels silently, which is glint#136 in a different coat. So: refuse, naming the fix.
            if n_panels > 1:
                if panel_slabs is not None and getattr(d, "ndim", 0) == 3:
                    return np.asarray(d[()], np.float32)     # (panel, ss, fs), integrated per slab
                raise NotImplementedError(
                    f"{path}:{data_path} is being read as a stack of {d.shape[0]} PANELS, but the "
                    f"geometry has {n_panels} panels and none of them maps to a slab: every "
                    f"reflection outside panel 0 would integrate to exactly 0 (measured) or read "
                    f"another panel's pixels (glint#148). Declare each panel's slab in the .geom "
                    f"with an integer dimN key (e.g. `p1/dim0 = 1`) and GLINT integrates each "
                    f"panel on its own slab; or use the --images route (integrate_cxi) for "
                    f"assembled stacked .cxi; or, if this file is really one frame per EVENT, "
                    f"pass event_axis=True / --event-axis event.")
            a = np.asarray(d[0], np.float32)         # single-panel geometry: slab 0 IS the frame
        else:
            a = np.asarray(d[()], np.float32)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]        # a ONE-event 4-D file is not "stacked" (leading axis 1), but its single
                        # event is still a panel stack -- hand it to the 3-D rules below rather
                        # than returning 4 axes to a 2-D integrator (Copilot review of #157)
    if a.ndim == 3:
        if n_panels > 1 and panel_slabs is not None:
            # A mapped multi-panel stack goes back WHOLE at any slab count -- including a
            # singleton. Squeezing (1, ss, fs) to 2-D here skipped integrate_spots_stack's slab
            # validation, so a geometry mapping panels onto slabs {0, 1} silently integrated
            # slab 0 for every panel (Copilot review of #157); now that mismatch raises by name.
            return a
        if a.shape[0] > 1:
            # one EVENT of a 4-D (event, panel, ss, fs) file: the same un-assembled shape, so the
            # same rules as the 3-D branch above -- slab-local with a mapping, refused without.
            # (This line used to take a[0] unconditionally, which was the #148 defect on the 4-D
            # path.)
            if n_panels > 1 and panel_slabs is not None:
                return a
            if n_panels > 1:
                raise NotImplementedError(
                    f"{path}:{data_path} event {event!r} is a stack of {a.shape[0]} PANELS under "
                    f"a {n_panels}-panel geometry with no slab mapping: reflections outside "
                    f"panel 0 would integrate to exactly 0 (glint#148). Declare integer dimN keys "
                    f"in the .geom (e.g. `p1/dim1 = 1`).")
        elif event_axis is False and n_panels > 1:
            # An EXPLICIT panel reading of a (1, ss, fs) singleton under a multi-panel geometry with
            # no slab mapping. Auto (event_axis=None) keeps the legacy per-file behaviour below --
            # the singleton IS the frame -- but the caller has just said it is a panel stack, and
            # taking a[0] would integrate one slab against every panel's predictions, which is the
            # glint#148 defect. The override wins on singletons exactly as it does on stacks
            # (Copilot review of #183 found the --images route skipping it; this is the same hole).
            raise NotImplementedError(
                f"{path}:{data_path} is a (1, ss, fs) singleton read as a 1-slab PANEL stack "
                f"(event_axis=False / --event-axis panel) under a {n_panels}-panel geometry with "
                f"no slab mapping: every reflection outside that slab would integrate to exactly 0 "
                f"(glint#148). Declare integer dimN keys in the .geom, or drop the override to read "
                f"it as the single assembled frame it is treated as by default.")
        a = a[0]                                     # (1, ss, fs) -> single assembled 2D frame
    return a


def integrate_frames(results, geom, image_dir=".", data_path=None, dmin=2.0, tol=0.006,
                     bg_mode="clipmean", event_axis=None):
    """Native predict + box-integrate (the fast, self-contained QC path; for the best MERGE use
    ``glint --fromfile`` -> CrystFEL refine). For each result carrying an orientation ``M``: load the
    frame image (``image_dir/<basename(image)>`` at the geom ``data`` path), predict on-detector spots,
    integrate. Attaches pred/I/sigma/peak/bg to each result in place. Returns (n_integrated, tot_refl).

    This variant reads one image FILE per result (legacy per-file detectors) and takes the frame at
    that result's own ``event`` when the file is an EVENT stack -- see ``_load_image`` for how that
    is decided from the geometry's panel count, and ``event_axis`` to override it. It does NOT read
    per-event clen/energy: for a modern STACKED .cxi (the ``--images`` front end, many events in one
    file) ``integrate_cxi`` is still the better route. Until glint#136 this function ignored
    ``event`` entirely and integrated every result against event 0 of its file."""
    panels, clen = panels_from_geom(geom)
    # Slab mapping from the .geom's integer dimN keys, required on EVERY panel to count: with it,
    # an un-assembled (panel, ss, fs) stack comes back whole from _load_image and is integrated
    # slab-locally; without it such a stack is refused there rather than guessed at (glint#148).
    slabs = [p["slab"] for p in panels]
    panel_slabs = slabs if len(panels) > 1 and all(s is not None for s in slabs) else None
    if data_path is None:
        data_path = geom.get("global", {}).get("data", "/data/data")
    lam = geom.get("wavelength_A")
    n = tot = 0
    for r in results:
        M = r.get("M")
        if M is None:
            continue
        img = _load_image(os.path.join(image_dir, os.path.basename(str(r.get("image", "")))),
                          data_path, r.get("event", 0),       # THIS result's event, not frame 0 (glint#136)
                          n_panels=len(panels), event_axis=event_axis, panel_slabs=panel_slabs)
        pred = predict_spots(M, panels, clen, lam, dmin=dmin, tol=tol)
        if img.ndim == 3:                             # un-assembled panel stack -> slab-local (glint#148)
            I, sig, peak, bg = integrate_spots_stack(img, pred, panels, bg_mode=bg_mode)
        else:
            I, sig, peak, bg = integrate_spots(img, pred, bg_mode=bg_mode)
        # Keep NON-POSITIVE intensities. Dropping them is a selection on the measured value of
        # the quantity being measured: a reflection whose true I is ~0 measures negative about
        # half the time, so discarding exactly those while keeping their positive counterparts
        # biases the retained mean upward, worst where the data are weakest. partialator takes
        # negatives, and the streaming path (stream_driver) already kept them and cut by SNR at
        # merge time instead -- this line was the only place the two paths disagreed. glint#130.
        keep = np.isfinite(I) & np.isfinite(sig) & (sig > 0)
        r.update(pred=pred[keep], I=I[keep], sigma=sig[keep], peak=peak[keep], bg=bg[keep])
        n += 1; tot += int(keep.sum())
    return n, tot


def integrate_cxi(results, geom_path, wavelength_A=None, dmin=2.0, tol=0.006, half=3, clen_scale=None,
                  sym_refine=None, sym_refine_tol=0.02, bg_mode="clipmean", data_key=None,
                  event_axis=None):
    """Self-contained native integrate for a STACKED .cxi -- the ``--images`` merge path, no CrystFEL.

    LAYOUT. The frame for event ``ev`` is ``data[ev]`` of a 3-D ``(event, ss, fs)`` stack -- the
    reading ``frames_from_cxi`` takes by definition of this front end -- but the file is put through
    the SAME decision the --peaks route uses (``_leading_axis_is_events``) before that index is
    trusted: a dataset whose leading axis is the geometry's PANEL count and is not the file's own
    event count is an un-assembled ``(panel, ss, fs)`` stack, and ``data[ev]`` of it is one panel's
    pixels, not one event's. That is refused by name (glint#148) rather than integrated silently, as
    is any per-event frame that is not a single 2-D image (a 4-D ``(event, panel, ss, fs)`` file):
    this integrator works on ONE assembled frame. ``event_axis`` is the explicit override
    (``--event-axis`` on the CLI, both routes); the ambiguous case -- leading axis == panel count and
    no per-event metadata -- raises and names it, exactly as ``_load_image`` does.

    sym_refine (default None -> OFF, nothing changes): if set to a Bravais system name (e.g.
    ``"tetragonal"``) AND a result carries its observed reciprocal peaks under key ``"q"`` (the (N,3)
    1/A vectors it was indexed from), refine that frame's orientation with the Bravais-CONSTRAINED
    refine (``glint.refine_sym.refine_bravais``, cell locked to the manifold to machine precision)
    before predicting.  This imposes the lattice symmetry on the per-frame cell the way CrystFEL's
    prediction-refinement does, instead of GLINT's unconstrained (triclinic-drifting) fit.  Results
    without a ``"q"`` are left untouched, so callers that do not attach observed peaks are unaffected.
    ``sym_refine_tol`` is the absolute inlier gate (1/A) handed to the refine.

    Predicts + box-integrates each GLINT-indexed frame, reading its image straight from
    ``results[i]['image']`` at event ``results[i]['event']`` (``data[event]`` in the .cxi), using the SAME
    ``lute_bridge`` geom panels + per-event clen/energy that ``frames_from_cxi``/``peaks_to_q`` used to make
    the q it indexed -- so the predicted (fs,ss) invert the observed peaks by construction (the round trip is
    self-consistent; a wrong orientation would look off the real spots and merge worse, not better). Attaches
    pred/I/sigma/peak/bg to each result carrying an orientation ``M`` in place; returns (n_integrated,
    tot_refl). Frame data are read once per file (h5 handles cached, closed on return).

    ``bg_mode`` is handed to ``integrate_spots``; the default changed from the annulus median to a
    MAD-clipped mean in glint#131, so the published merge numbers below were measured with
    ``bg_mode="median"``, which is still available for reproducing them. The glint#129 A/B measured
    both under the paper protocol: clipmean/partiality CC*=0.9122 / Rsplit=34.58% vs median/partiality
    0.9146 / 31.60%.

    On real lysozyme stills (Jungfrau-4M, 1482 frames) this self-merges to CC*=0.915 / Rsplit=31.6% at
    2.1 A (partialator, default 10 cycles; the older 0.90/39% figure was a --iterations=1 under-converged
    merge, glint#129) -- on par with a CrystFEL/xgandalf run on the same frames (CC*=0.930 / Rsplit=34.9%)
    -- with peak search, indexing AND integration all in GLINT. For the best (prediction-refined) merge,
    hand orientations to CrystFEL via ``write_fromfile``."""
    import h5py
    from glint.lute_bridge import parse_geom as _parse_geom, lambda_from_eV, _meta
    panels, glob = _parse_geom(geom_path)
    clen_spec, en_spec = glob.get("clen"), glob.get("photon_energy")
    coff = float(glob.get("coffset", 0.0))
    data_key = data_key or glob.get("data", "/entry_1/data_1/data")   # explicit arg > .geom `data` key > default
    handles = {}
    def _h5(p):
        h = handles.get(p)
        if h is None:
            h = handles[p] = h5py.File(p, "r")
        return h
    n = tot = 0
    try:
        for r in results:
            if r.get("M") is None:
                continue
            M_raw = r["M"]
            if sym_refine and r.get("q") is not None:             # Bravais-constrained per-frame orientation refine (default OFF)
                from glint.refine_sym import refine_bravais
                qobs = np.asarray(r["q"], float)
                qobs = qobs[np.isfinite(qobs).all(1)]
                if len(qobs) >= 6:
                    M_raw, _, _, _ = refine_bravais(qobs, M_raw, sym_refine, sym_refine_tol)
            M = _canonical_axes(M_raw)                             # (long,long,short): cross-frame-consistent hkl for the merge
            f = _h5(str(r.get("image")))
            ev = int(r.get("event", 0))
            clen = _meta(clen_spec, f, ev, 0.1)
            scale = clen_scale if clen_scale is not None else (0.001 if abs(clen) > 10 else 1.0)
            clen_m = clen * scale + coff
            wl = wavelength_A
            if wl is None:
                eV = _meta(en_spec, f, ev, None)
                wl = lambda_from_eV(eV) if eV else None
            if wl is None:
                continue
            pred = predict_spots(M, panels, clen_m, wl, dmin=dmin, tol=tol)
            dset = f[data_key]
            if getattr(dset, "ndim", 0) >= 3:
                # (event, ss, fs) is the fast path; the decision below only refuses what is NOT that.
                # A (1, ss, fs) singleton is auto-read as one assembled frame (the legacy per-file
                # layout), but an EXPLICIT override still wins there too: event_axis=False declares
                # a 1-slab panel stack, which this integrator cannot serve, so it is refused rather
                # than quietly read as event 0 (Copilot review of #183).
                if (dset.shape[0] > 1 or event_axis is not None) and not _leading_axis_is_events(
                        f, dset, str(r.get("image")), data_key, n_panels=len(panels),
                        event_axis=event_axis):
                    raise NotImplementedError(
                        f"{r.get('image')}:{data_key} reads as a stack of {dset.shape[0]} PANELS "
                        f"under a {len(panels)}-panel geometry, not as events, so data[{ev}] would "
                        f"be one panel's pixels integrated as event {ev}'s assembled frame "
                        f"(glint#148). integrate_cxi (the --images route) integrates ASSEMBLED "
                        f"(event, ss, fs) stacks only: for an un-assembled panel stack use the "
                        f"--peaks route with integer dimN keys in the .geom (slab-local "
                        f"integration), or pass event_axis=True / --event-axis event if this file "
                        f"really is one frame per event.")
                frame = np.asarray(dset[ev], np.float32)
            else:
                frame = np.asarray(dset, np.float32)
            if frame.ndim != 2:
                raise NotImplementedError(
                    f"{r.get('image')}:{data_key} event {ev} is a {frame.ndim}-D array of shape "
                    f"{frame.shape} -- an un-assembled per-event panel stack -- and integrate_cxi "
                    f"integrates one assembled 2-D frame per event (glint#148). Use the --peaks "
                    f"route with integer dimN keys in the .geom for slab-local integration.")
            I, sig, peak, bg = integrate_spots(frame, pred, half=half, bg_mode=bg_mode)
            keep = np.isfinite(I) & np.isfinite(sig) & (sig > 0)   # non-positive I kept: glint#130
            r.update(M=M, pred=pred[keep], I=I[keep], sigma=sig[keep], peak=peak[keep], bg=bg[keep])  # store canonical M so the stream cell matches the hkl
            n += 1; tot += int(keep.sum())
    finally:
        for h in handles.values():
            h.close()
    return n, tot
