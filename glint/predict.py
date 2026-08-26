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
            # frame against a ~0.26 ms/frame known-cell path -- if that matters, the next step is to
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


def panels_from_geom(geom):
    """Convert an ``fftindex.geom.parse_geom()`` result into the panel dicts predict_spots/project_q
    want (same flat-panel model, different key names). Returns (panels, clen_m)."""
    g = geom.get("global", {})
    clen = float(g.get("clen", 0.1)); coff = float(g.get("coffset", 0.0)); res_g = float(g.get("res", 1.0))
    panels = []
    for nm, p in geom["panels"].items():
        panels.append(dict(name=nm, fs=np.array([p["fsx"], p["fsy"]]), ss=np.array([p["ssx"], p["ssy"]]),
                           res=float(p.get("res", res_g)), cx=float(p["corner_x"]), cy=float(p["corner_y"]),
                           coffset=float(p.get("coffset", coff)),
                           min_fs=int(p["min_fs"]), max_fs=int(p["max_fs"]),
                           min_ss=int(p["min_ss"]), max_ss=int(p["max_ss"])))
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


def _load_image(path, data_path, event=0, n_panels=1, event_axis=None):
    """Read the frame for THIS event out of an image file at the geom ``data`` path.

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

    A PANEL reading is only honoured for a ONE-PANEL geometry, where slab 0 is the whole detector.
    Un-assembled multi-panel input is refused, because integration runs on one assembled frame and
    a single slab is not one -- see the comment at the raise, and glint#148.
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
        is_event = event_axis
        if stacked and is_event is None:
            n_ev = _event_count(f)
            if n_ev is not None:
                is_event = (n_ev == d.shape[0])      # the file's own per-event metadata decides
            elif n_panels <= 1 or d.shape[0] != n_panels:
                is_event = True                      # cannot be a panel stack
            else:
                raise ValueError(
                    f"{path}:{data_path} has a leading axis of {d.shape[0]}, which equals the "
                    f"geometry's panel count, and the file carries no per-event metadata "
                    f"({', '.join(_EVENT_COUNT_PATHS)}) to settle it. It is either {d.shape[0]} "
                    f"EVENTS of an assembled frame or {n_panels} PANELS of one event, and picking "
                    f"wrong integrates the wrong pixels silently -- which is glint#136. Pass "
                    f"event_axis=True (events) or event_axis=False (panels), or --event-axis "
                    f"event|panel on the CLI, to say which it is.")
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
            # on a 2-panel tiled geometry: predict_spots emits ASSEMBLED coordinates (project_q
            # adds each panel's min_fs/min_ss), so a real 4500-count spot on panel 1 comes back
            # I=0.0 sigma=0.0 peak=0.0 from a slab-0 frame -- silently zero, not missing. On a
            # CrystFEL 3-D layout (dim0 selects the slab, every panel min_fs/min_ss = 0..N) it is
            # worse: all panels occupy the SAME assembled window, so panel-1 reflections read
            # panel-0 pixels. Assembling correctly needs a multi-panel data model in project_q and
            # integrate_spots, not a reshape here -- glint#148. So: refuse.
            if n_panels > 1:
                raise NotImplementedError(
                    f"{path}:{data_path} is being read as a stack of {d.shape[0]} PANELS, but the "
                    f"geometry has {n_panels} panels and integration runs on ONE assembled frame: "
                    f"predict_spots emits assembled fs/ss across all panels, so every reflection "
                    f"outside panel 0 would integrate to exactly 0 (measured) or read another "
                    f"panel's pixels. Un-assembled multi-panel input is not supported (glint#148). "
                    f"Use the --images route (integrate_cxi) for stacked .cxi, or supply assembled "
                    f"frames; if this file is really one frame per EVENT, pass event_axis=True / "
                    f"--event-axis event.")
            a = np.asarray(d[0], np.float32)         # single-panel geometry: slab 0 IS the frame
        else:
            a = np.asarray(d[()], np.float32)
    if a.ndim == 3:                                  # (1, ss, fs) -> single assembled 2D frame
        a = a[0]
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
                          n_panels=len(panels), event_axis=event_axis)
        pred = predict_spots(M, panels, clen, lam, dmin=dmin, tol=tol)
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
                  sym_refine=None, sym_refine_tol=0.02, bg_mode="clipmean", data_key=None):
    """Self-contained native integrate for a STACKED .cxi -- the ``--images`` merge path, no CrystFEL.

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
    MAD-clipped mean in glint#131, so the merge numbers below -- and every intensity GLINT produced
    before that -- were measured with ``bg_mode="median"``, which is still available for reproducing
    them. They have NOT been re-measured under the new default.

    On real lysozyme stills (Jungfrau-4M, 1476 frames) this self-merges to CC*=0.90 / Rsplit=39% at 2.1 A --
    on par with a CrystFEL/xgandalf run on the same frames -- with peak search, indexing AND integration all
    in GLINT. For the best (prediction-refined) merge, hand orientations to CrystFEL via ``write_fromfile``."""
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
            frame = np.asarray(dset[ev] if getattr(dset, "ndim", 0) >= 3 else dset, np.float32)
            I, sig, peak, bg = integrate_spots(frame, pred, half=half, bg_mode=bg_mode)
            keep = np.isfinite(I) & np.isfinite(sig) & (sig > 0)   # non-positive I kept: glint#130
            r.update(M=M, pred=pred[keep], I=I[keep], sigma=sig[keep], peak=peak[keep], bg=bg[keep])  # store canonical M so the stream cell matches the hkl
            n += 1; tot += int(keep.sum())
    finally:
        for h in handles.values():
            h.close()
    return n, tot
