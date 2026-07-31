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


def integrate_spots(data, pred, half=3, gap=2, ring=3):
    """Box integrate a predicted reflection list against an assembled detector array `data` (ss,fs).

    Signal = sum over a (2*half+1)^2 box; background = robust mean of a surrounding annulus
    (gap..gap+ring) scaled to the box; I = signal - nbox*bg; sigma = sqrt(signal + nbox*bg)  (Poisson,
    gain=1). Returns (I, sigma, peak, bg_per_px) arrays aligned with `pred`. Out-of-frame -> 0.
    """
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
        bg = np.median(annpx, axis=1) if ann.any() else np.zeros(len(vi))
        sig_sum = boxpx.sum(1)
        I[vi] = sig_sum - nbox * bg
        sig[vi] = np.sqrt(np.maximum(sig_sum + nbox * np.maximum(bg, 0.0), 1.0))
        peak[vi] = boxpx.max(1); bgpp[vi] = bg
    return I, sig, peak, bgpp


_RCOL = "   h    k    l          I   sigma(I)   peak  background  fs/px  ss/px panel\n"


def _header(geom_text):
    """CrystFEL stream header. The reader needs a geometry-file block right after the format line
    (else it consumes the whole file as audit info and aborts with 'Too much audit information')."""
    g = geom_text if geom_text is not None else (
        "clen = 0.1\nres = 5000\nadu_per_photon = 1\nphoton_energy = 9400\n"
        "p0/min_fs = 0\np0/max_fs = 1023\np0/min_ss = 0\np0/max_ss = 1023\n"
        "p0/corner_x = -512.5\np0/corner_y = -512.5\np0/fs = x\np0/ss = y\n")
    return ("CrystFEL stream format 2.3\nGenerated by GLINT (fftindex predict+integrate)\n"
            "----- Begin geometry file -----\n" + g.rstrip("\n") + "\n----- End geometry file -----\n")


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


def _write_chunk(f, serial, r, panel_name="p0", photon_eV=9392.7, clen_m=0.15, panel_names=None):
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
    f.write("num_peaks = 0\nnum_saturated_peaks = 0\n")
    f.write("Peaks from peak search\n  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel\n")
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
    n_idx = 0
    with open(path, "w") as f:
        f.write(_header(geom_text))
        for serial, r in enumerate(results, 1):
            n_idx += bool(_write_chunk(f, serial, r, panel_name, photon_eV, clen_m, panel_names))
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
        self._f = open(self.path, "w")
        self._f.write(_header(geom_text))

    def write(self, r):
        if self._f is None:
            raise ValueError("StreamWriter is closed")
        self.n_chunks += 1
        if _write_chunk(self._f, self.n_chunks, r, self.panel_name, self.photon_eV, self.clen_m,
                        self.panel_names):
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


def _load_image(path, data_path):
    import h5py
    with h5py.File(path, "r") as f:
        a = np.asarray(f[data_path][()], np.float32)
    if a.ndim == 3:                                  # (event|panel, ss, fs) -> single assembled 2D frame
        a = a[0] if a.shape[0] > 1 else a[0]
    return a


def integrate_frames(results, geom, image_dir=".", data_path=None, dmin=2.0, tol=0.006):
    """Native predict + box-integrate (the fast, self-contained QC path; for the best MERGE use
    ``glint --fromfile`` -> CrystFEL refine). For each result carrying an orientation ``M``: load the
    frame image (``image_dir/<basename(image)>`` at the geom ``data`` path), predict on-detector spots,
    integrate. Attaches pred/I/sigma/peak/bg to each result in place. Returns (n_integrated, tot_refl).

    This variant reads one image FILE per result (legacy per-file detectors). For a modern STACKED .cxi
    (the ``--images`` front end, many events in one file) use ``integrate_cxi`` instead."""
    panels, clen = panels_from_geom(geom)
    if data_path is None:
        data_path = geom.get("global", {}).get("data", "/data/data")
    lam = geom.get("wavelength_A")
    n = tot = 0
    for r in results:
        M = r.get("M")
        if M is None:
            continue
        img = _load_image(os.path.join(image_dir, os.path.basename(str(r.get("image", "")))), data_path)
        pred = predict_spots(M, panels, clen, lam, dmin=dmin, tol=tol)
        I, sig, peak, bg = integrate_spots(img, pred)
        keep = (I > 0) & np.isfinite(sig) & (sig > 0)
        r.update(pred=pred[keep], I=I[keep], sigma=sig[keep], peak=peak[keep], bg=bg[keep])
        n += 1; tot += int(keep.sum())
    return n, tot


def integrate_cxi(results, geom_path, wavelength_A=None, dmin=2.0, tol=0.006, half=3, clen_scale=None,
                  sym_refine=None, sym_refine_tol=0.02):
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

    On real lysozyme stills (Jungfrau-4M, 1476 frames) this self-merges to CC*=0.90 / Rsplit=39% at 2.1 A --
    on par with a CrystFEL/xgandalf run on the same frames -- with peak search, indexing AND integration all
    in GLINT. For the best (prediction-refined) merge, hand orientations to CrystFEL via ``write_fromfile``."""
    import h5py
    from glint.lute_bridge import parse_geom as _parse_geom, lambda_from_eV, _meta
    panels, glob = _parse_geom(geom_path)
    clen_spec, en_spec = glob.get("clen"), glob.get("photon_energy")
    coff = float(glob.get("coffset", 0.0))
    data_key = glob.get("data", "/entry_1/data_1/data")
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
            I, sig, peak, bg = integrate_spots(frame, pred, half=half)
            keep = (I > 0) & np.isfinite(sig) & (sig > 0)
            r.update(M=M, pred=pred[keep], I=I[keep], sigma=sig[keep], peak=peak[keep], bg=bg[keep])  # store canonical M so the stream cell matches the hkl
            n += 1; tot += int(keep.sum())
    finally:
        for h in handles.values():
            h.close()
    return n, tot
