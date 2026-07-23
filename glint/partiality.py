"""Partiality + per-crystal scale model for the GLINT stills merge (item 3).

In serial femtosecond crystallography every still slices a reciprocal-lattice
node at a FRACTIONAL Ewald crossing: the measured intensity is a partial
``I_obs = G_c * p_hkl * I_full`` where ``p in (0,1]`` is the partiality (how
much of the full reflection the rocking curve captures) and ``G_c`` is a
per-crystal scale.  GLINT's current merge GATES partials (the
``predict_spots`` excitation-error window) but does NOT model the fraction, so
the merged intensities are partial and unscaled.  partialator (CrystFEL) models
``p`` and fits ``G_c`` by alternating scaling rounds and un-scales
``I_full,est = I_obs / (G_c * p)`` -- a large part of why its merge is better.

This module supplies exactly that, as NEW code behind a flag (default OFF):

  * ``partiality(exc, q_mag, eta, dloverl, Dinv)`` -- a Gaussian rocking-curve
    partiality from the excitation error ``exc`` already stored per reflection
    by ``predict_spots`` (field "exc", 1/A) and the reflection's ``|q|`` (=1/d).
    No new geometry: ``exc`` and ``res=1/|q|`` come straight from prediction.

  * ``PartialityScaler`` -- a partialator-style alternating per-crystal scale
    <-> merge fit.  Given per-measurement (asu-key, I, sigma, exc, |q|, crystal,
    frame) it fits ``G_c`` by weighted least squares against a running merged
    reference and returns per-measurement corrected values ``v = I/(G_c p)`` and
    weights ``w = p^2/sigma^2`` -- the ONLY two quantities the merge fold needs.
    Feed those to ``MergeAccumulator.add_frame(..., values=v, weights=w)``.

The rocking-curve width is resolution dependent (three physical broadeners in
quadrature): a finite-size floor ``1/D`` (const), mosaicity ``eta*|q|`` (linear
in |q|) and bandwidth ``0.5*lambda*|q|^2*(dl/l)`` (quadratic in |q|), so high-|q|
reflections have wider curves and lower average partiality -- the real behaviour.

Design note: the scale<->I_full global scale is degenerate (only ratios are
constrained); the reference is renormalised each round (geometric-mean of G_c
fixed to 1) and any absolute scale is recovered once, globally, at scoring time.
"""
from __future__ import annotations

import numpy as np

P_MIN = 0.02          # clamp: never divide by a partiality below this
_LN = np.log


def sigma_p(q_mag, eta, dloverl, Dinv=0.0, wavelength_A=1.0):
    """Rocking-curve width sigma_p(|q|) [1/A] from three broadeners in quadrature.

    term1 (finite size)  : Dinv                          (constant in |q|)
    term2 (mosaicity)    : eta * |q| / 2                 (linear in |q|)
    term3 (bandwidth)    : 0.5 * lambda * |q|^2 * (dl/l) (quadratic in |q|)

    eta is the mosaic full width in RADIANS. Scalar or broadcastable arrays.
    """
    q = np.asarray(q_mag, float)
    t1 = Dinv
    t2 = 0.5 * eta * q
    t3 = 0.5 * wavelength_A * q * q * dloverl
    return np.sqrt(t1 * t1 + t2 * t2 + t3 * t3)


def partiality(exc, q_mag, eta, dloverl, Dinv=0.0, wavelength_A=1.0, p_min=P_MIN):
    """Gaussian rocking-curve partiality p = exp(-exc^2 / (2 sigma_p^2)), clamped >= p_min.

    exc [1/A] : excitation error (predict_spots field "exc"; 0 on the Ewald sphere).
    q_mag[1/A]: reflection |q| = 1/d (from predict_spots field "res" = 1/|q|).
    Returns p in [p_min, 1].
    """
    sp = sigma_p(q_mag, eta, dloverl, Dinv, wavelength_A)
    sp = np.maximum(sp, 1e-9)
    p = np.exp(-np.asarray(exc, float) ** 2 / (2.0 * sp * sp))
    return np.clip(p, p_min, 1.0)


class PartialityScaler:
    """partialator-style alternating per-crystal scale <-> merge fit.

    Parameters (the partiality model kernel; kept GLOBAL by default -- per-still
    identifiability of eta is weak, so a global eta/bandwidth is the robust
    variant.  Per-crystal G_c is always fit):

        eta          mosaic full width [rad]           (global model value)
        dloverl      relative bandwidth dl/l           (global model value)
        Dinv         finite-size floor [1/A]           (global model value)
        wavelength_A
        n_iter       alternating scale<->merge rounds
        min_refl     crystals with fewer usable refl keep G_c = 1 (unstable fit)
        fit_B        also fit a per-crystal B_c (resolution falloff); default False

    Call ``fit(key, I, sigma, exc, q_mag, crystal, frame)`` (1-D arrays over ALL
    measurements).  After fitting, ``self.values`` and ``self.weights`` are the
    per-measurement ``v = I/(G_c p) [* exp(2 B_c s^2)]`` and ``w = p^2/sigma^2``
    ready for the merge fold.  ``self.G`` is the per-crystal scale, ``self.p``
    the predicted partiality, ``self.I_ref`` the final merged reference (dict).
    """

    def __init__(self, eta, dloverl, Dinv=0.0, wavelength_A=1.0, n_iter=5,
                 min_refl=8, fit_B=False, tol=1e-3, verbose=False, B_clip=15.0):
        self.B_clip = float(B_clip)
        self.eta = float(eta)
        self.dloverl = float(dloverl)
        self.Dinv = float(Dinv)
        self.wavelength_A = float(wavelength_A)
        self.n_iter = int(n_iter)
        self.min_refl = int(min_refl)
        self.fit_B = bool(fit_B)
        self.tol = float(tol)
        self.verbose = bool(verbose)

    def fit(self, key, I, sigma, exc, q_mag, crystal, frame):
        key = np.asarray(key, np.int64)
        I = np.asarray(I, float)
        sigma = np.maximum(np.asarray(sigma, float), 1e-3)
        exc = np.asarray(exc, float)
        q_mag = np.asarray(q_mag, float)
        crystal = np.asarray(crystal, np.int64)
        frame = np.asarray(frame, np.int64)

        # predicted partiality from the (global) model kernel
        p = partiality(exc, q_mag, self.eta, self.dloverl, self.Dinv, self.wavelength_A)
        self.p = p
        s2 = (q_mag / 2.0) ** 2                      # s^2 = (|q|/2)^2 = (sin th/lambda)^2
        w = p * p / (sigma * sigma)                  # merge weight: p^2 / sigma^2
        self.weights = w

        # compact crystal / key indices
        uc, cidx = np.unique(crystal, return_inverse=True)
        uk, kidx = np.unique(key, return_inverse=True)
        nC, nK = len(uc), len(uk)
        self._uk = uk

        Ioverp = I / p                               # I_obs / p  (== G_c * I_full in the noiseless model)

        # init G_c = weighted mean of I/p per crystal (reduces to 1/mean(I) when p~1)
        num = np.bincount(cidx, weights=w * Ioverp, minlength=nC)
        den = np.bincount(cidx, weights=w, minlength=nC)
        G = np.where(den > 0, num / den, 1.0)
        G = G / np.exp(np.mean(_LN(np.maximum(G, 1e-12))))   # geomean(G)=1
        B = np.zeros(nC)
        n_per_c = np.bincount(cidx, minlength=nC)

        prev_ref = None
        for it in range(self.n_iter):
            corr = np.exp(2.0 * B[cidx] * s2)         # exp(+2 B s^2)
            v = Ioverp / G[cidx] * corr               # I_full estimate per measurement
            # ---- merge reference: I_ref(k) = sum w v / sum w ----
            rnum = np.bincount(kidx, weights=w * v, minlength=nK)
            rden = np.bincount(kidx, weights=w, minlength=nK)
            I_ref = np.where(rden > 0, rnum / rden, 0.0)
            if prev_ref is not None:
                d = np.abs(I_ref - prev_ref).sum() / max(np.abs(I_ref).sum(), 1e-12)
                if self.verbose:
                    print(f"    [scaler] iter {it} dRef={d:.2e}")
                if d < self.tol:
                    prev_ref = I_ref
                    break
            prev_ref = I_ref
            # ---- per-crystal WLS refit of (ln G_c [, B_c]) against I_ref ----
            ref_m = I_ref[kidx]
            valid = ref_m > 0
            if not self.fit_B:
                # minimise sum w (I/p - G * ref)^2  ->  G = sum w (I/p) ref / sum w ref^2
                gn = np.bincount(cidx, weights=np.where(valid, w * Ioverp * ref_m, 0.0), minlength=nC)
                gd = np.bincount(cidx, weights=np.where(valid, w * ref_m * ref_m, 0.0), minlength=nC)
                Gnew = np.where(gd > 0, gn / gd, G)
            else:
                # 2-param log-space WLS: ln(I/p) = ln G - 2 B s^2 + ln ref
                y = np.where(valid, _LN(np.maximum(Ioverp, 1e-9)) - _LN(np.maximum(ref_m, 1e-9)), 0.0)
                x = -2.0 * s2                          # coefficient of B
                ww = np.where(valid, w, 0.0)
                Swx = np.bincount(cidx, weights=ww * x, minlength=nC)
                Sww = np.bincount(cidx, weights=ww, minlength=nC)
                Swxx = np.bincount(cidx, weights=ww * x * x, minlength=nC)
                Swy = np.bincount(cidx, weights=ww * y, minlength=nC)
                Swxy = np.bincount(cidx, weights=ww * x * y, minlength=nC)
                det = Sww * Swxx - Swx * Swx
                lnG = np.where(np.abs(det) > 1e-12, (Swxx * Swy - Swx * Swxy) / det, 0.0)
                Bnew = np.where(np.abs(det) > 1e-12, (Sww * Swxy - Swx * Swy) / det, 0.0)
                Gnew = np.exp(np.clip(lnG, -10.0, 10.0))
                # per-still B_c is weakly constrained and diverges without restraint (partialator
                # uses B-factor restraints for the same reason); clamp to a physical window and drop
                # non-finite fits. Even so fit_B is default OFF -- the G-only variant is the robust win.
                Bnew = np.clip(np.where(np.isfinite(Bnew), Bnew, 0.0), -self.B_clip, self.B_clip)
                B = np.where(n_per_c >= self.min_refl, Bnew, 0.0)
            # crystals with too few reflections keep G=1 (unstable scale)
            G = np.where(n_per_c >= self.min_refl, Gnew, 1.0)
            G = np.where(np.isfinite(G) & (G > 0), G, 1.0)
            G = G / np.exp(np.mean(_LN(np.maximum(G, 1e-12))))   # renormalise: geomean(G)=1

        # final per-measurement corrected values + weights
        corr = np.exp(2.0 * B[cidx] * s2)
        self.G = G
        self.B = B
        self.values = Ioverp / G[cidx] * corr
        self.I_ref = {int(k): float(v) for k, v in zip(uk, prev_ref)}
        return self
