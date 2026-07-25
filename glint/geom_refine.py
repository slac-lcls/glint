"""Live detector-geometry refinement -- a running accumulator, in the shape of MergeAccumulator.

The detector geometry (distance ``clen``; beam-center shift ``dfs``,``dss``) is a
FRAME-INVARIANT global, exactly like the unit cell. So the same "pool weak per-frame evidence
into a consensus" move GLINT uses to recover the cell refines the geometry: each indexed frame
contributes predicted-vs-observed peak residuals, and pooled across frames they over-determine
a small correction. The Jacobian w.r.t. ``clen`` is taken NUMERICALLY through
``predict.project_q``, so the exact multi-panel ``.geom`` model is used with no hand-derived
derivatives; the beam-center columns are analytic (a global shift of the predicted pattern).

Two signatures make the three parameters identifiable and separable:
  * a beam-center error shifts every peak UNIFORMLY   (dU/d(dfs) = 1) ;
  * a distance error BREATHES them radially           (dU/d(clen) via project_q).
They separate as soon as peaks span a range of radii -- i.e. given orientation diversity, which
serial data supplies for free.

``gamma`` < 1 turns on EXPONENTIAL FORGETTING so the estimate TRACKS a slowly drifting detector
(thermal creep) instead of averaging it away; ``gamma`` = 1 is a static calibration (the normal
equations reset after each solve, an ICP-style outer loop).

This PR is DIAGNOSTIC-ONLY: ``correction()`` reports the running (clen, dfs, dss). Feeding the
correction back into indexing/prediction (closing the live-recalibration loop) is the flagged
next step. NOTES: per-shot wavelength and per-panel metrology are more columns, same machinery.
"""
import numpy as np
from glint.predict import project_q, predict_spots


class GeomRefiner:
    """Pool per-frame residuals into a 3-parameter geometry correction (clen, dfs, dss).

    add_frame() one indexed frame at a time (like MergeAccumulator.add_frame); every
    ``update_every`` frames it solves the pooled 3x3 and applies the step. Read the running
    estimate with correction().
    """

    def __init__(self, panels, clen_m, wavelength_A, *, gamma=1.0, update_every=32,
                 gate=8.0, huber=3.0, fd_clen=1e-4, min_frames=8, dmin=2.0, tol=0.006):
        self.panels = panels
        self.clen0 = float(clen_m)
        self.lam = float(wavelength_A)
        self.gamma = float(gamma)
        self.update_every = int(update_every)
        self.gate = float(gate)
        self.huber = float(huber)
        self.fd_clen = float(fd_clen)                 # finite-difference step for d/dclen (m)
        self.min_frames = int(min_frames)
        self.dmin, self.tol = float(dmin), float(tol)
        self.dclen = 0.0                              # correction to clen0 (m)
        self.dfs = 0.0                                # beam-center shift, fast axis (px)
        self.dss = 0.0                                # beam-center shift, slow axis (px)
        self.N = np.zeros((3, 3))
        self.b = np.zeros(3)
        self.n_frames = 0
        self.n_resid = 0
        self.n_solves = 0

    @property
    def clen_m(self):
        return self.clen0 + self.dclen

    def correction(self):
        return dict(clen_m=self.clen_m, dclen_m=self.dclen, dfs=self.dfs, dss=self.dss,
                    n_frames=self.n_frames, n_resid=self.n_resid, n_solves=self.n_solves)

    def _project(self, q, clen):
        fs, ss, pan = project_q(q, self.panels, clen, self.lam)
        return np.stack([fs, ss], 1), pan

    def _frame_normal_eqs(self, R, obs, pred):
        """(N, b) contribution of one frame at the current (clen, dfs, dss)."""
        if pred is None:
            pr = predict_spots(R, self.panels, self.clen_m, self.lam,
                               dmin=self.dmin, tol=self.tol, is_recip=True)
        else:
            pr = pred
        hkl = np.stack([pr["h"], pr["k"], pr["l"]], 1).astype(float)
        if len(hkl) == 0:
            return np.zeros((3, 3)), np.zeros(3), 0
        q = hkl @ R
        m0, pan = self._project(q, self.clen_m)
        m = m0 + np.array([self.dfs, self.dss])
        ok = (pan >= 0) & np.isfinite(m).all(1)
        q, m = q[ok], m[ok]
        if len(m) == 0:
            return np.zeros((3, 3)), np.zeros(3), 0
        d2 = ((m[:, None, :] - obs[None, :, :]) ** 2).sum(2)     # nearest observed peak per prediction
        j = d2.argmin(1)
        sel = d2[np.arange(len(m)), j] < self.gate * self.gate
        q, m, om = q[sel], m[sel], obs[j[sel]]
        k = len(m)
        if k == 0:
            return np.zeros((3, 3)), np.zeros(3), 0
        m1, _ = self._project(q, self.clen_m + self.fd_clen)
        dclen_col = (m1 + np.array([self.dfs, self.dss]) - m) / self.fd_clen   # d(fs,ss)/d(clen)
        N, b = np.zeros((3, 3)), np.zeros(3)
        for ax in (0, 1):                                        # fs row, then ss row
            r = om[:, ax] - m[:, ax]
            J = np.column_stack([dclen_col[:, ax],
                                 np.full(k, 1.0 if ax == 0 else 0.0),
                                 np.full(k, 0.0 if ax == 0 else 1.0)])
            good = np.isfinite(J).all(1) & np.isfinite(r)
            J, r = J[good], r[good]
            w = np.where(np.abs(r) <= self.huber, 1.0, self.huber / np.maximum(np.abs(r), 1e-9))
            N += (J * w[:, None]).T @ J
            b += (J * w[:, None]).T @ r
        return N, b, k

    def add_frame(self, R, obs, pred=None):
        """Pool one indexed frame. R: recip rows (3x3). obs: (n,2) observed (fs,ss) peaks.
        pred: optional predicted-reflection record (needs fields h,k,l) to reuse the driver's
        prediction; if None, predicts internally."""
        obs = np.asarray(obs, float)
        if obs.ndim != 2 or obs.shape[0] == 0:
            return
        dN, db, k = self._frame_normal_eqs(np.asarray(R, float), obs, pred)
        if k == 0:
            return
        self.N = self.gamma * self.N + dN
        self.b = self.gamma * self.b + db
        self.n_frames += 1
        self.n_resid += k
        if self.n_frames >= self.min_frames and self.n_frames % self.update_every == 0:
            self.solve()

    def solve(self):
        """Solve the pooled 3x3 and apply the step. Returns the applied delta (or None)."""
        try:
            d = np.linalg.solve(self.N + 1e-9 * np.eye(3), self.b)
        except np.linalg.LinAlgError:
            return None
        self.dclen += float(d[0])
        self.dfs += float(d[1])
        self.dss += float(d[2])
        self.n_solves += 1
        if self.gamma >= 1.0:                # static calibration: re-linearise at the new estimate
            self.N[:] = 0.0
            self.b[:] = 0.0
        return d


def refine_geometry(panels, clen_m, wavelength_A, frames, *, passes=6, gamma=1.0, **kw):
    """Offline batch driver: run `passes` sweeps over `frames` = [(R, obs_fsss), ...].

    Returns the fitted GeomRefiner. With gamma=1 (default) it recomputes residuals at the current
    estimate each pass (ICP-style); with gamma<1 it streams with forgetting."""
    gr = GeomRefiner(panels, clen_m, wavelength_A, gamma=gamma, update_every=len(frames) or 1,
                     min_frames=1, **kw)
    for _ in range(passes):
        for R, obs in frames:
            gr.add_frame(R, obs)
    return gr
