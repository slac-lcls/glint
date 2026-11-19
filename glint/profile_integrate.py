"""Profile-fitting integration, ported from laue-dials to torch and batched over frames.

The algorithm is ``laue_dials.algorithms.integration.Integrator`` (rs-station/laue-dials, main at
d57f96e; MIT licence, reproduced in THIRD_PARTY_NOTICES.md) together with the per-image driver around it (``laue_dials.command_line.integrate.
integrate_image``): the mask-dilation drop, the minimum-pixel drop and the skip-the-image-on-failure
rule. Each reflection gets a 2-D Gaussian profile on a disk window of radius R. Over ``maxiter``
rounds it re-estimates

  * the background, a profile-distance-weighted mean of (counts - I*p) over the window;
  * the profile, a weighted mean and covariance of the background-subtracted pixels of its k
    nearest STRONG neighbours (never itself), so weak spots inherit a determined shape;
  * the intensity, the Poisson profile fit I = sum((c-b) w), w = (p/v) / sum(p^2/v), v = I p + b;
  * the strong set, I >= isigi_cutoff * sigma.

What this port adds is the batch. laue-dials fits one image at a time in numpy; here a batch of F
frames is padded to (F, N reflections, M window pixels) and every step runs as one torch op on the
device. Each frame keeps its own radius, k, strong set, convergence test and failure, so a batch
gives the per-image answer frame by frame. ``experiments/test_profile_integrate.py`` checks that
against the laue-dials numpy code itself.

PIXEL CONVENTION. GLINT puts pixel centres at integers: the box integrator centres its box on
``rint(fs)``. laue-dials says pixel i spans [i, i+1) and centres the window on ``floor(centroid)``.
The port maps one onto the other with centroid = fs + 0.5, so both integrators centre on the same
pixel.

PANELS. A reflection's window is clipped to its own panel's rectangle of the data array, which is
how laue-dials treats a window that overhangs a panel edge. Pixels past the edge are invalid, not
clamped copies. Nearest neighbours and the automatic radius use lab-frame pixel coordinates from
the .geom corner and fs/ss vectors. Panel-local coordinates would pile every panel onto (0, 0); the
laue-dials docstrings explain why that breaks both.

DEPARTURES, none of which changes a number:
  * the Poisson score of a pixel whose count is not a non-negative integer is -inf, as in
    ``scipy.stats.poisson.logpmf``. The score is then inf, the relative improvement is nan, and
    the early stop never fires, as in laue-dials. With the default maxiter=2 the stop cannot
    change anything anyway: it is tested after the updates and only skips later iterations;
  * a failed frame (<= 1 strong spot, or none left after a pass) comes back with sigma = 0 on every
    reflection, where laue-dials returns an empty table for the image. GLINT's ``keep`` filter
    (sigma > 0) drops both the same way.
"""
from __future__ import annotations

import math

import numpy as np

MIN_WINDOW_PIXELS = 6                 # laue_dials.command_line.integrate.MIN_WINDOW_PIXELS


def _torch():
    import torch                                     # deferred, as everywhere torch is optional
    return torch


def lab_pixel_coords(pred, panels):
    """Lab-frame (X, Y) of each predicted reflection, in pixels, from the .geom corner and fs/ss
    vectors (``geom.py``: X = corner_x + lfs*fsx + lss*ssx). This is the frame shared by every panel,
    used for nearest neighbours and the automatic radius."""
    n = len(pred)
    out = np.zeros((n, 2))
    pan = np.asarray(pred["panel"], int)
    for pi, p in enumerate(panels):
        m = pan == pi
        if not m.any():
            continue
        lfs = pred["fs"][m] - p["min_fs"]; lss = pred["ss"][m] - p["min_ss"]
        out[m, 0] = p["cx"] + lfs * p["fs"][0] + lss * p["ss"][0]
        out[m, 1] = p["cy"] + lfs * p["fs"][1] + lss * p["ss"][1]
    return out


def _panel_rects(pred, panels, shape):
    """(n, 5) int: slab, row lo, row hi, col lo, col hi (inclusive) of each reflection's panel."""
    n = len(pred)
    if panels is None:
        S, H, W = shape
        return np.tile(np.array([0, 0, H - 1, 0, W - 1]), (n, 1))
    tab = np.array([[p.get("slab") or 0, p["min_ss"], p["max_ss"], p["min_fs"], p["max_fs"]]
                    for p in panels], dtype=np.int64)
    return tab[np.asarray(pred["panel"], int)]


def _disk(R):
    """Window offsets (row, col) of radius R, in the order laue-dials builds them."""
    g = np.mgrid[-R:R + 1, -R:R + 1].reshape(2, -1).T
    r = np.sqrt((g ** 2).sum(1))
    return g[r <= R], r[r <= R]


def estimate_radius(nb, rv):
    """Per-frame ``estimate_integration_radius``: round(0.5 * 20th percentile of nearest-neighbour
    distances), from every prediction on the frame. nb (F,N,d), rv (F,N) bool. Returns (F,) long,
    -1 where a frame has fewer than two predictions (laue-dials would fail there too)."""
    torch = _torch()
    D = torch.cdist(nb, nb)
    big = torch.finfo(D.dtype).max
    pair = rv[:, :, None] & rv[:, None, :]
    pair &= ~torch.eye(nb.shape[1], dtype=torch.bool, device=nb.device)[None]
    D = torch.where(pair, D, torch.full_like(D, big))
    nn = D.min(-1).values
    nn = torch.where(rv, nn, torch.full_like(nn, float("nan")))
    q = torch.nanquantile(nn, 0.2, dim=1)                    # linear interpolation, as numpy
    ok = (rv.sum(1) >= 2) & torch.isfinite(q)
    R = torch.round(0.5 * torch.where(ok, q, torch.zeros_like(q)))   # half-to-even, as np.round
    return torch.where(ok, R.long(), torch.full_like(R, -1).long())


def _mvn(xy, loc, S):
    """Batched 2-D ``mvn_log_pdf`` with the z-score. xy (...,M,2), loc (...,2), S (...,2,2)."""
    torch = _torch()
    d = xy - loc[..., None, :]
    a, b, c, e = S[..., 0, 0], S[..., 0, 1], S[..., 1, 0], S[..., 1, 1]
    det = a * e - b * c
    i00, i01, i10, i11 = e / det, -b / det, -c / det, a / det
    dx, dy = d[..., 0], d[..., 1]
    z = (dx * (i00[..., None] * dx + i01[..., None] * dy)
         + dy * (i10[..., None] * dx + i11[..., None] * dy))
    logZ = -math.log(2 * math.pi) - 0.5 * torch.log(torch.abs(det))      # d = 2; slogdet's log|det|
    return logZ[..., None] - 0.5 * z, z


def _poisson_nll(c, v):
    """-scipy.stats.poisson.logpmf(c, v), +inf where c is not a non-negative integer."""
    torch = _torch()
    is_int = (c >= 0) & (c == torch.floor(c))
    ll = torch.xlogy(c, v) - v - torch.lgamma(c + 1)
    return torch.where(is_int, -ll, torch.full_like(ll, float("inf")))


class _Batch:
    """One padded batch of frames, the state laue-dials' Integrator keeps per image."""

    def __init__(self, img, mask, cen, rect, nb, rv, radius, k, isigi_cutoff, epsilon, drop_masked):
        torch = _torch()
        dev, dt = img.device, cen.dtype
        F, N = rv.shape
        self.k, self.cut, self.eps = k, isigi_cutoff, epsilon
        self.ok = rv.any(1)                                   # frame still integrable
        if radius is None:
            R = estimate_radius(nb, rv)
            self.ok &= R >= 0
            R = R.clamp(min=0)
        else:
            R = torch.full((F,), int(radius), dtype=torch.long, device=dev)
        self.R = R
        Rmax = int(R.max()) if F else 0
        off, offr = _disk(Rmax)
        off = torch.as_tensor(off, device=dev); offr = torch.as_tensor(offr, device=dev, dtype=dt)
        disk = offr[None, :] <= R[:, None].to(dt)                       # (F,M)
        self.m = disk.sum(1)                                             # laue-dials' self.m
        base = torch.floor(cen).long()                                   # centre pixel: floor, not round
        idx = base[:, :, None, :] + off[None, None]                      # (F,N,M,2) row, col
        sl, r0, r1, c0, c1 = (rect[..., i] for i in range(5))
        inb = ((idx[..., 0] >= r0[..., None]) & (idx[..., 0] <= r1[..., None])
               & (idx[..., 1] >= c0[..., None]) & (idx[..., 1] <= c1[..., None]))
        rr = torch.minimum(torch.maximum(idx[..., 0], r0[..., None]), r1[..., None])
        cc = torch.minimum(torch.maximum(idx[..., 1], c0[..., None]), c1[..., None])
        fi = torch.arange(F, device=dev)[:, None, None].expand_as(rr)
        si = sl[..., None].expand_as(rr)
        good = mask[fi, si, rr, cc]
        inwin = inb & disk[:, None, :]
        if drop_masked:
            # laue.integrate's unmasked_selection_per_panel: drop a prediction whose radius-R disk
            # holds a masked pixel of its own panel (isotropic dilation of the bad pixels by R, then
            # the centre pixel looked up, which is the same test).
            rv = rv & ~(inwin & ~good).any(-1)
        valid = inwin & good & rv[..., None]
        nvalid = valid.sum(-1)
        # ... then the MIN_WINDOW_PIXELS rule: too few usable pixels for an ellipse, so drop it.
        enough = nvalid >= torch.clamp(0.25 * self.m.to(dt), min=MIN_WINDOW_PIXELS)[:, None]
        rv = rv & enough
        valid &= rv[..., None]
        self.ok &= rv.any(1)
        self.rv, self.valid = rv, valid
        self.nb = nb
        self.c = img[fi, si, rr, cc].to(dt)                              # (F,N,M) the windows
        self.cen = cen
        self.xy = idx.to(dt) + 0.5                                       # laue-dials' self.xy
        self.xyrel = self.xy - cen[:, :, None, :]
        nv = valid.sum(-1).clamp(min=1).to(dt)
        self.I = (self.c * valid).sum(-1) / nv
        self.U = torch.sqrt(self.I.clamp(min=0))
        eye = torch.eye(2, dtype=dt, device=dev)
        self.loc = cen.clone()
        self.S = (eye[None, None] * (R.to(dt) / 2.0)[:, None, None, None]).expand(F, N, 2, 2).clone()
        self.bg = torch.ones((F, N), dtype=dt, device=dev)
        self.strong = rv.clone()
        self._eye = eye
        # softmax needs at least one finite entry per row; padded rows get every pixel, then zeroed
        self._smask = valid | ~rv[..., None]

    def _logp(self):
        return _mvn(self.xy, self.loc, self.S)

    def _p(self, logp):
        torch = _torch()
        p = torch.softmax(torch.where(self._smask, logp, torch.full_like(logp, -float("inf"))), -1)
        return p * self.valid

    def _knn(self, act):
        """``assign_knn``: the k nearest STRONG spots of every reflection, never itself."""
        torch = _torch()
        F, N = self.rv.shape
        strong = self.strong & self.rv
        ns = strong.sum(1)
        fail = act & (ns <= 1)
        kact = torch.clamp(ns - 1, max=self.k)
        D = torch.cdist(self.nb, self.nb)
        cand = strong[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=D.device)[None]
        D = torch.where(cand, D, torch.full_like(D, float("inf")))
        kk = min(self.k, N)
        dist, nn = torch.topk(D, kk, dim=-1, largest=False)
        use = (torch.arange(kk, device=D.device)[None, None, :] < kact[:, None, None]) & torch.isfinite(dist)
        return nn, use, fail

    def step(self, act):
        """One laue-dials iteration on the frames in ``act``. Returns (score, failed)."""
        torch = _torch()
        F, N, M = self.c.shape
        eps = self.eps
        A = act[:, None]
        nn, use, fail = self._knn(act)

        # estimate_background
        logp, z = self._logp()
        p = self._p(logp)
        w = torch.where(self.valid, z + eps, torch.zeros_like(z))
        bg = (w * (self.c - self.I[..., None] * p)).sum(-1) / w.sum(-1).clamp(min=1e-300)
        bg = bg.clamp(min=eps)
        self.bg = torch.where(A, bg, self.bg)

        # estimate_profiles: pool k neighbours' background-subtracted pixels, offsets taken from
        # each neighbour's own centroid
        pf = torch.where(self.valid, torch.exp(logp), torch.zeros_like(logp))
        wp = (self.c - self.bg[..., None]).clamp(min=0) / self.I.clamp(min=eps)[..., None] * pf
        fi = torch.arange(F, device=wp.device)[:, None, None]
        wk = (wp[fi, nn] * use[..., None]).reshape(F, N, -1)                  # (F,N,k*M)
        xk = self.xyrel[fi, nn].reshape(F, N, -1, 2)
        ws = wk.sum(-1)
        sig = (ws > 0) & A & self.rv
        wsafe = torch.where(ws > 0, ws, torch.ones_like(ws))
        mu = (wk[..., None] * xk).sum(-2) / wsafe[..., None]
        X = xk - mu[..., None, :]
        S = torch.einsum("fnm,fnmi,fnmj->fnij", wk, X, X) / wsafe[..., None, None]
        self.loc = torch.where(sig[..., None], mu + self.cen, self.loc)
        self.S = torch.where(sig[..., None, None], S + eps * self._eye, self.S)

        # integrate: Poisson profile fit, v from the PREVIOUS intensity and the new profile
        logp, _ = self._logp()
        p = self._p(logp)
        v = self.I.clamp(min=0)[..., None] * p + self.bg[..., None]
        wi = p / v
        wi = wi / (p * p / v).sum(-1, keepdim=True)
        I = ((self.c - self.bg[..., None]) * wi).sum(-1)
        U = torch.sqrt((v * wi).sum(-1))
        self.I = torch.where(A, I, self.I)
        self.U = torch.where(A, U, self.U)

        # set_strong, then the no-strong-spots failure
        strong = (self.I >= self.cut * self.U) & self.rv
        self.strong = torch.where(A, strong, self.strong)
        fail = fail | (act & ~(self.strong & self.rv).any(1))

        # score (after this iteration's updates, as laue-dials evaluates it)
        v = self.I.clamp(min=0)[..., None] * p + self.bg[..., None]
        nll = torch.where(self.valid, _poisson_nll(self.c, v), torch.zeros_like(v))
        return nll.sum((1, 2)), fail


def profile_integrate_frames(frames, preds, panels=None, masks=None, *, radius=None, k=5,
                             isigi_cutoff=3.0, maxiter=2, tol=1e-3, epsilon=1e-6, drop_masked=True,
                             batch=8, device=None, dtype=None):
    """Profile-fit integrate many frames at once. Returns one ``(I, sigma, peak, bg)`` per frame,
    aligned with its ``pred``, the same contract as ``predict.integrate_spots``.

    frames  sequence of 2-D (ss, fs) arrays, or 3-D (slab, ss, fs) panel stacks with the panel's
            ``slab`` from the .geom (glint#148), all the same shape.
    preds   one ``predict_spots`` structured array per frame (fs, ss, panel).
    panels  ``panels_from_geom`` panels; None treats each frame as a single panel.
    masks   a boolean good-pixel array shaped like one frame (shared), or one per frame, or None.
    radius  window radius (px); None estimates it per frame as laue-dials does.
    k, isigi_cutoff, maxiter, tol, epsilon   laue-dials' parameters, same defaults.
    drop_masked  apply laue.integrate's mask-dilation and minimum-pixel drops.
    batch   frames per device batch. 8 is past the knee on an A6000 (~600 predictions a frame):
            larger batches are no faster and peak memory grows linearly, ~0.27 GB a frame in float64.
    dtype   torch dtype for the fit. Default float64, which is what the agreement test checks against
            laue-dials.

    Dropped reflections and every reflection of a failed frame come back with I = sigma = 0, which
    GLINT's ``keep`` filter (sigma > 0) removes. ``bg`` is the per-pixel background and ``peak`` the
    window maximum, as the box integrator reports them.
    """
    torch = _torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = dtype or torch.float64
    out = []
    shared_mask = masks is not None and not isinstance(masks, (list, tuple))
    for b0 in range(0, len(frames), batch):
        fr = frames[b0:b0 + batch]
        pr = preds[b0:b0 + batch]
        stack = [np.asarray(f) for f in fr]
        stack = [f[None] if f.ndim == 2 else f for f in stack]
        shape = stack[0].shape
        img = torch.as_tensor(np.stack(stack), device=device)
        if masks is None:
            mk = torch.ones((len(fr),) + shape, dtype=torch.bool, device=device)
        else:
            ms = [masks] * len(fr) if shared_mask else masks[b0:b0 + batch]
            ms = [np.asarray(m, bool) for m in ms]
            mk = torch.as_tensor(np.stack([m[None] if m.ndim == 2 else m for m in ms]), device=device)
        F = len(fr)
        N = max(1, max(len(p) for p in pr))
        cen = np.zeros((F, N, 2)); rect = np.zeros((F, N, 5), np.int64); nb = np.zeros((F, N, 2))
        rv = np.zeros((F, N), bool)
        for i, p in enumerate(pr):
            n = len(p)
            if n == 0:
                continue
            cen[i, :n, 0] = np.asarray(p["ss"], float) + 0.5         # GLINT centre-at-integer -> [i, i+1)
            cen[i, :n, 1] = np.asarray(p["fs"], float) + 0.5
            rect[i, :n] = _panel_rects(p, panels, shape)
            nb[i, :n] = (lab_pixel_coords(p, panels) if panels is not None
                         else np.stack([p["fs"], p["ss"]], 1))
            rv[i, :n] = True
        T = lambda a, d=dt: torch.as_tensor(a, device=device, dtype=d)
        st = _Batch(img, mk, T(cen), T(rect, torch.long), T(nb), T(rv, torch.bool), radius, k,
                    isigi_cutoff, epsilon, drop_masked)
        act = st.ok.clone()
        prev = None
        for _ in range(maxiter):
            if not bool(act.any()):
                break
            score, fail = st.step(act)
            st.ok &= ~fail
            act &= ~fail
            if prev is not None:
                imp = (prev - score) / torch.clamp(prev.abs(), min=epsilon)
                act &= ~(imp < tol)                        # nan (inf - inf) never stops, as in numpy
            prev = score
        good = st.ok[:, None] & st.rv
        I = torch.where(good, st.I, torch.zeros_like(st.I))
        U = torch.where(good, st.U, torch.zeros_like(st.U))
        peak = torch.where(st.valid, st.c, torch.full_like(st.c, -float("inf"))).amax(-1)
        peak = torch.where(good, peak, torch.zeros_like(peak))
        bgo = torch.where(good, st.bg, torch.zeros_like(st.bg))
        I, U, peak, bgo = (t.cpu().numpy() for t in (I, U, peak, bgo))
        for i, p in enumerate(pr):
            n = len(p)
            out.append((I[i, :n].astype(float), U[i, :n].astype(float),
                        peak[i, :n].astype(float), bgo[i, :n].astype(float)))
    return out


def profile_integrate_spots(data, pred, panels=None, mask=None, **kw):
    """Single-frame ``profile_integrate_frames``, a drop-in for ``predict.integrate_spots``."""
    return profile_integrate_frames([data], [pred], panels, mask, **kw)[0]
