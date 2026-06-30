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
    data = np.asarray(data, float)
    H, W = data.shape
    n = len(pred)
    I = np.zeros(n); sig = np.zeros(n); peak = np.zeros(n); bgpp = np.zeros(n)
    R = half + gap + ring
    for i in range(n):
        cf = int(round(pred["fs"][i])); cs = int(round(pred["ss"][i]))
        if cs - R < 0 or cs + R >= H or cf - R < 0 or cf + R >= W:
            continue
        patch = data[cs - R:cs + R + 1, cf - R:cf + R + 1]
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        rad = np.maximum(np.abs(yy), np.abs(xx))         # Chebyshev (square rings)
        box = rad <= half
        ann = (rad > half + gap) & (rad <= half + gap + ring)
        bg = np.median(patch[ann]) if ann.any() else 0.0
        nbox = int(box.sum())
        sig_sum = patch[box].sum()
        I[i] = sig_sum - nbox * bg
        sig[i] = np.sqrt(max(sig_sum + nbox * max(bg, 0.0), 1.0))
        peak[i] = patch[box].max()
        bgpp[i] = bg
    return I, sig, peak, bgpp


_HDR = "CrystFEL stream format 2.3\nGenerated by GLINT (fftindex predict+integrate)\n"
_RCOL = "   h    k    l          I   sigma(I)   peak  background  fs/px  ss/px panel\n"


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


def write_stream_integrated(results, path, panel_name="p0"):
    """results: list of {image,event,M, pred (structured), I, sigma, peak, bg}. Writes a CrystFEL
    .stream with REAL integrated reflection rows (mergeable by partialator/process_hkl)."""
    n_idx = 0
    with open(path, "w") as f:
        f.write(_HDR)
        for r in results:
            f.write("----- Begin chunk -----\n")
            f.write(f"Image filename: {r.get('image', 'glint.cxi')}\n")
            f.write(f"Event: //{r.get('event', 0)}\n")
            M = r.get("M")
            valid = M is not None and abs(np.linalg.det(np.asarray(M, float))) >= 1.0
            f.write(f"indexed_by = {'glint' if valid else 'none'}\n")
            if valid:
                n_idx += 1
                f.write("Begin crystal\n" + _cell_line(M) + "\n")
                f.write("lattice_type = triclinic\ncentering = P\nunique_axis = *\n")
                f.write("Reflections measured after indexing\n" + _RCOL)
                pred, I, sg = r.get("pred"), r.get("I"), r.get("sigma")
                pk, bg = r.get("peak"), r.get("bg")
                if pred is not None:
                    for j in range(len(pred)):
                        f.write(f"{pred['h'][j]:4d} {pred['k'][j]:4d} {pred['l'][j]:4d} "
                                f"{I[j]:10.2f} {sg[j]:10.2f} {pk[j]:6.1f} {bg[j]:10.2f} "
                                f"{pred['fs'][j]:6.1f} {pred['ss'][j]:6.1f} {panel_name}\n")
                f.write("End of reflections\nEnd crystal\n")
            f.write("----- End chunk -----\n")
    return n_idx
