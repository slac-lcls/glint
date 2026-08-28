"""project_q rewrite: hoist frame-invariant panel geometry, cut numpy op count. Round 2.

Round 1 lesson (kept here as the reason for each choice):
  * np.errstate(...) as a divide guard costs ~10 us of PURE fixed overhead -- more than the ops it
    saved. Replaced by a one-off np.where on the rare non-forward path.
  * ((fss>=lo)&(fss<=hi)).all(1) on an (n,2) bool is a 2-long inner reduction: slower than two
    column ANDs despite being one fewer "op".
  * np.linalg.norm is an __array_function__ dispatch wrapping conj/mul/add.reduce/sqrt; the explicit
    sqrt(add.reduce(s*s,1)) is the SAME arithmetic (numpy's own implementation) with less dispatch.

Variants:
  V1a  packed (n,2) compare      V1b  unpacked f/s (reference shape, contiguous)
  V2   broadcast over all panels (op count independent of P)
  V3   V1b minus the s_hat normalisation (NOT bit-identical -- reported separately)
"""
import os, sys, time, math
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for c in ("/sdf/home/s/smarches/glint_streamfix_wt", os.path.dirname(HERE)):
    if os.path.isdir(c):
        sys.path.insert(0, c)

Z_HAT = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------ REFERENCE (verbatim copy) ---
def project_q_ref(q, panels, clen_m, wavelength_A):
    q = np.asarray(q, float)
    s_hat = wavelength_A * q + Z_HAT
    s_hat = s_hat / np.linalg.norm(s_hat, axis=1, keepdims=True)
    n = len(q)
    fs = np.full(n, np.nan); ss = np.full(n, np.nan); pan = np.full(n, -1, int)
    fwd = s_hat[:, 2] > 1e-6
    for pi, p in enumerate(panels):
        Zp = clen_m + p["coffset"]
        t = np.where(fwd, Zp / np.where(fwd, s_hat[:, 2], 1.0), np.nan)
        X = s_hat * t[:, None]
        A = np.array([[p["fs"][0], p["ss"][0]], [p["fs"][1], p["ss"][1]]])
        rhs = p["res"] * X[:, :2] - np.array([p["cx"], p["cy"]])
        lfls = rhs @ np.linalg.inv(A).T
        f = p["min_fs"] + lfls[:, 0]; s = p["min_ss"] + lfls[:, 1]
        on = fwd & (f >= p["min_fs"]) & (f <= p["max_fs"]) & (s >= p["min_ss"]) & (s <= p["max_ss"])
        take = on & (pan < 0)
        fs[take] = f[take]; ss[take] = s[take]; pan[take] = pi
    return fs, ss, pan


# --------------------------------------------------------------- frame-invariant panel consts ---
class PanelConst:
    """Every quantity project_q recomputes each call that depends ONLY on panel geometry."""
    __slots__ = ("P", "W", "corner", "mins", "lo", "hi", "res", "coff", "single",
                 "gid", "gcoff", "gres", "ngroup",
                 "minf", "mins_", "maxf", "maxs")

    def __init__(self, panels):
        P = self.P = len(panels)
        self.W = np.empty((P, 2, 2)); self.corner = np.empty((P, 2))
        self.mins = np.empty((P, 2)); self.lo = np.empty((P, 2)); self.hi = np.empty((P, 2))
        self.res = np.empty(P); self.coff = np.empty(P)
        self.minf = np.empty(P); self.mins_ = np.empty(P)
        self.maxf = np.empty(P); self.maxs = np.empty(P)
        for i, p in enumerate(panels):
            A = np.array([[p["fs"][0], p["ss"][0]], [p["fs"][1], p["ss"][1]]])
            self.W[i] = np.linalg.inv(A).T                       # <- one np.linalg.inv per panel per frame, gone
            self.corner[i] = (p["cx"], p["cy"])                  # <- one np.array per panel per frame, gone
            self.mins[i] = self.lo[i] = (p["min_fs"], p["min_ss"])
            self.hi[i] = (p["max_fs"], p["max_ss"])
            self.res[i] = p["res"]; self.coff[i] = p["coffset"]
            self.minf[i] = p["min_fs"]; self.mins_[i] = p["min_ss"]
            self.maxf[i] = p["max_fs"]; self.maxs[i] = p["max_ss"]
        key = {}; gid = np.empty(P, np.int64)
        for i in range(P):
            k = (float(self.coff[i]), float(self.res[i]))
            key.setdefault(k, len(key)); gid[i] = key[k]
        self.gid = gid.tolist(); self.ngroup = len(key)
        self.gcoff = [k[0] for k in key]; self.gres = [k[1] for k in key]
        self.single = (P == 1)


_AR = np.arange(1 << 15)


def _arange(n):
    global _AR
    if n > len(_AR):
        _AR = np.arange(max(n, 2 * len(_AR)))
    return _AR[:n]


def _unit_s(q, wavelength_A):
    """s_hat, bit-identical to  (wl*q + Z_HAT) / np.linalg.norm(..., axis=1, keepdims=True)."""
    s = wavelength_A * q
    s += Z_HAT
    nrm = np.sqrt(np.add.reduce(s * s, axis=1, keepdims=True))   # exactly what np.linalg.norm does
    s /= nrm
    return s


# ------------------------------------------------------------------------- V1a: packed compare --
def project_q_v1a(q, C, clen_m, wavelength_A):
    q = np.asarray(q, float)
    n = len(q)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, int)
    s_hat = _unit_s(q, wavelength_A)
    sz = s_hat[:, 2]
    fwd = None
    if not (sz.min() > 1e-6):                       # the ONLY case the reference's two np.where guard
        fwd = sz > 1e-6
        sz = np.where(fwd, sz, 1.0)
    rq = []
    for g in range(C.ngroup):                       # once per (coffset,res) GROUP, not per panel
        rq.append(C.gres[g] * (s_hat[:, :2] * ((clen_m + C.gcoff[g]) / sz)[:, None]))
    if C.single:
        fss = C.mins[0] + (rq[0] - C.corner[0]) @ C.W[0]
        ge = fss >= C.lo[0]; ge &= fss <= C.hi[0]
        on = ge[:, 0] & ge[:, 1]
        if fwd is not None:
            on &= fwd
        return (np.where(on, fss[:, 0], np.nan), np.where(on, fss[:, 1], np.nan),
                np.where(on, 0, -1))
    fs = np.full(n, np.nan); ss = np.full(n, np.nan); pan = np.full(n, -1, int)
    free = np.ones(n, bool)
    for pi in range(C.P):
        fss = C.mins[pi] + (rq[C.gid[pi]] - C.corner[pi]) @ C.W[pi]
        ge = fss >= C.lo[pi]; ge &= fss <= C.hi[pi]
        on = ge[:, 0] & ge[:, 1]
        if fwd is not None:
            on &= fwd
        take = on & free
        np.copyto(fs, fss[:, 0], where=take)
        np.copyto(ss, fss[:, 1], where=take)
        np.copyto(pan, pi, where=take)
        if pi + 1 < C.P:
            np.copyto(free, False, where=on)
    return fs, ss, pan


# ------------------------------------------------------------- V1b: unpacked f/s (contiguous) ---
def project_q_v1b(q, C, clen_m, wavelength_A):
    q = np.asarray(q, float)
    n = len(q)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, int)
    s_hat = _unit_s(q, wavelength_A)
    sz = s_hat[:, 2]
    fwd = None
    if not (sz.min() > 1e-6):
        fwd = sz > 1e-6
        sz = np.where(fwd, sz, 1.0)
    rq = []
    for g in range(C.ngroup):
        rq.append(C.gres[g] * (s_hat[:, :2] * ((clen_m + C.gcoff[g]) / sz)[:, None]))
    if C.single:
        lfls = (rq[0] - C.corner[0]) @ C.W[0]
        f = C.minf[0] + lfls[:, 0]; s = C.mins_[0] + lfls[:, 1]
        on = f >= C.minf[0]; on &= f <= C.maxf[0]; on &= s >= C.mins_[0]; on &= s <= C.maxs[0]
        if fwd is not None:
            on &= fwd
        return np.where(on, f, np.nan), np.where(on, s, np.nan), np.where(on, 0, -1)
    fs = np.full(n, np.nan); ss = np.full(n, np.nan); pan = np.full(n, -1, int)
    free = np.ones(n, bool)
    for pi in range(C.P):
        lfls = (rq[C.gid[pi]] - C.corner[pi]) @ C.W[pi]
        f = C.minf[pi] + lfls[:, 0]; s = C.mins_[pi] + lfls[:, 1]
        on = f >= C.minf[pi]; on &= f <= C.maxf[pi]; on &= s >= C.mins_[pi]; on &= s <= C.maxs[pi]
        if fwd is not None:
            on &= fwd
        take = on & free
        np.copyto(fs, f, where=take); np.copyto(ss, s, where=take)
        np.copyto(pan, pi, where=take)
        if pi + 1 < C.P:
            np.copyto(free, False, where=on)
    return fs, ss, pan



# --------------------------- V1c: V1b + one contiguous copy of s_hat[:,:2] (strided reads hurt) --
def project_q_v1c(q, C, clen_m, wavelength_A):
    q = np.asarray(q, float)
    n = len(q)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, int)
    s_hat = _unit_s(q, wavelength_A)
    sz = s_hat[:, 2]
    fwd = None
    if not (sz.min() > 1e-6):
        fwd = sz > 1e-6
        sz = np.where(fwd, sz, 1.0)
    sxy = np.ascontiguousarray(s_hat[:, :2])
    rq = []
    for g in range(C.ngroup):
        rq.append(C.gres[g] * (sxy * ((clen_m + C.gcoff[g]) / sz)[:, None]))
    if C.single:
        lfls = (rq[0] - C.corner[0]) @ C.W[0]
        f = C.minf[0] + lfls[:, 0]; s = C.mins_[0] + lfls[:, 1]
        on = f >= C.minf[0]; on &= f <= C.maxf[0]; on &= s >= C.mins_[0]; on &= s <= C.maxs[0]
        if fwd is not None:
            on &= fwd
        return np.where(on, f, np.nan), np.where(on, s, np.nan), np.where(on, 0, -1)
    fs = np.full(n, np.nan); ss = np.full(n, np.nan); pan = np.full(n, -1, int)
    free = np.ones(n, bool)
    for pi in range(C.P):
        lfls = (rq[C.gid[pi]] - C.corner[pi]) @ C.W[pi]
        f = C.minf[pi] + lfls[:, 0]; s = C.mins_[pi] + lfls[:, 1]
        on = f >= C.minf[pi]; on &= f <= C.maxf[pi]; on &= s >= C.mins_[pi]; on &= s <= C.maxs[pi]
        if fwd is not None:
            on &= fwd
        take = on & free
        np.copyto(fs, f, where=take); np.copyto(ss, s, where=take)
        np.copyto(pan, pi, where=take)
        if pi + 1 < C.P:
            np.copyto(free, False, where=on)
    return fs, ss, pan


# ---------------------------------------------------------------- V2: broadcast over panels -----
def project_q_v2(q, C, clen_m, wavelength_A):
    q = np.asarray(q, float)
    n = len(q)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, int)
    s_hat = _unit_s(q, wavelength_A)
    sz = s_hat[:, 2]
    fwd = None
    if not (sz.min() > 1e-6):
        fwd = sz > 1e-6
        sz = np.where(fwd, sz, 1.0)
    t = (clen_m + C.coff)[:, None] / sz                        # (P,n)
    rhs = C.res[:, None, None] * (s_hat[None, :, :2] * t[:, :, None])
    rhs -= C.corner[:, None, :]
    fss = C.mins[:, None, :] + np.matmul(rhs, C.W)             # batched (P,n,2)@(P,2,2)
    ge = fss >= C.lo[:, None, :]; ge &= fss <= C.hi[:, None, :]
    on = ge[:, :, 0] & ge[:, :, 1]                             # (P,n)
    if fwd is not None:
        on &= fwd
    any_ = on.any(0)
    pi = on.argmax(0)                                          # FIRST panel that catches it
    sel = fss[pi, _arange(n)]
    return (np.where(any_, sel[:, 0], np.nan), np.where(any_, sel[:, 1], np.nan),
            np.where(any_, pi, -1))


# ------------------------------- V3: V1b without the s_hat normalisation (NOT bit-identical) -----
def project_q_v3(q, C, clen_m, wavelength_A):
    """(f,s) depend on s_hat only through the ratios sx/sz, sy/sz -- which normalisation does not
    change. Dropping it removes a 3n multiply-add-sqrt and a 3n divide. Rounding differs."""
    q = np.asarray(q, float)
    n = len(q)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, int)
    s_hat = wavelength_A * q
    s_hat += Z_HAT                                             # NOT normalised
    sz = s_hat[:, 2]
    fwd = None
    if not (sz.min() > 1e-6):
        fwd = sz > 1e-6
        sz = np.where(fwd, sz, 1.0)
    rq = []
    for g in range(C.ngroup):
        rq.append(C.gres[g] * (s_hat[:, :2] * ((clen_m + C.gcoff[g]) / sz)[:, None]))
    if C.single:
        lfls = (rq[0] - C.corner[0]) @ C.W[0]
        f = C.minf[0] + lfls[:, 0]; s = C.mins_[0] + lfls[:, 1]
        on = f >= C.minf[0]; on &= f <= C.maxf[0]; on &= s >= C.mins_[0]; on &= s <= C.maxs[0]
        if fwd is not None:
            on &= fwd
        return np.where(on, f, np.nan), np.where(on, s, np.nan), np.where(on, 0, -1)
    fs = np.full(n, np.nan); ss = np.full(n, np.nan); pan = np.full(n, -1, int)
    free = np.ones(n, bool)
    for pi in range(C.P):
        lfls = (rq[C.gid[pi]] - C.corner[pi]) @ C.W[pi]
        f = C.minf[pi] + lfls[:, 0]; s = C.mins_[pi] + lfls[:, 1]
        on = f >= C.minf[pi]; on &= f <= C.maxf[pi]; on &= s >= C.mins_[pi]; on &= s <= C.maxs[pi]
        if fwd is not None:
            on &= fwd
        take = on & free
        np.copyto(fs, f, where=take); np.copyto(ss, s, where=take)
        np.copyto(pan, pi, where=take)
        if pi + 1 < C.P:
            np.copyto(free, False, where=on)
    return fs, ss, pan


# ============================================================== numpy-op counter (auto) ==========
_NOPS = [0]
_NEST = [0]
_OAS = np.asarray          # captured BEFORE count_ops patches np.asarray


def _isarr(x):
    return isinstance(x, np.ndarray)


class _CA(np.ndarray):
    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        if not _NEST[0]:
            _NOPS[0] += 1
        ins = tuple(_OAS(x).view(np.ndarray) if _isarr(x) else x for x in inputs)
        if kw.get("out") is not None:
            o = kw["out"]
            o = o if isinstance(o, tuple) else (o,)
            kw["out"] = tuple(_OAS(x).view(np.ndarray) if _isarr(x) else x for x in o)
        r = getattr(ufunc, method)(*ins, **kw)
        return r.view(_CA) if _isarr(r) else r

    def __array_function__(self, func, types, args, kwargs):
        if not _NEST[0]:
            _NOPS[0] += 1

        def strip(x):
            if _isarr(x):
                return _OAS(x).view(np.ndarray)
            if isinstance(x, (list, tuple)):
                return type(x)(strip(y) for y in x)
            return x
        a = tuple(strip(x) for x in args)
        k = {kk: strip(vv) for kk, vv in kwargs.items()}
        _NEST[0] += 1
        try:
            r = func(*a, **k)
        finally:
            _NEST[0] -= 1
        return r.view(_CA) if _isarr(r) else r

    def __getitem__(self, k):
        if _isarr(k) or (isinstance(k, tuple) and any(_isarr(x) for x in k)):
            if not _NEST[0]:
                _NOPS[0] += 1
        return super().__getitem__(k)

    def __setitem__(self, k, v):
        if _isarr(k) or (isinstance(k, tuple) and any(_isarr(x) for x in k)):
            if not _NEST[0]:
                _NOPS[0] += 1
        return super().__setitem__(k, v)


_CTORS = ("full", "empty", "zeros", "ones", "array", "arange")


def count_ops(fn, q, geo, clen, lam):
    orig = {c: getattr(np, c) for c in _CTORS}
    o_asarray = np.asarray

    def wrap(f):
        def g(*a, **k):
            if not _NEST[0]:
                _NOPS[0] += 1
            r = f(*a, **k)
            return r.view(_CA) if _isarr(r) and not _NEST[0] else r
        return g

    def wrap_asarray(*a, **k):
        if not _NEST[0]:
            _NOPS[0] += 1
        r = o_asarray(*a, **k)
        if _NEST[0]:
            return r
        return r.view(_CA) if _isarr(r) else r
    for c in _CTORS:
        setattr(np, c, wrap(orig[c]))
    np.asarray = wrap_asarray
    _NOPS[0] = 0; _NEST[0] = 0
    try:
        fn(o_asarray(q, float).view(_CA), geo, clen, lam)
    finally:
        for c in _CTORS:
            setattr(np, c, orig[c])
        np.asarray = o_asarray
    return _NOPS[0]


# =============================================================== geometry / data construction ====
def tiled_panels(N, G, pix_mm, jitter=False, seed=0):
    rng = np.random.default_rng(seed)
    w = N // G
    res = 1.0 / (pix_mm / 1000.0)
    c0 = -(N / 2.0 - 0.5)
    ps = []
    for a in range(G):
        for b in range(G):
            fsv = np.array([1.0, 0.0, 0.0]); ssv = np.array([0.0, 1.0, 0.0])
            r = res; co = 0.0
            if jitter:
                th = rng.normal(0, 0.02)
                fsv = np.array([math.cos(th), math.sin(th), 0.0])
                ssv = np.array([-math.sin(th), math.cos(th), 0.0])
                co = float(rng.choice([0.0, 1e-4, -1.5e-4]))
                r = res * float(rng.choice([1.0, 1.0, 1.0002]))
            ps.append(dict(name=f"p{a}_{b}", fs=fsv, ss=ssv, res=r,
                           cx=c0 + b * w, cy=c0 + a * w, coffset=co,
                           min_fs=b * w, max_fs=b * w + w - 1,
                           min_ss=a * w, max_ss=a * w + w - 1))
    return ps


def rand_rot(rng):
    Qm, R = np.linalg.qr(rng.normal(size=(3, 3)))
    return Qm * np.sign(np.diag(R))


def make_q(R, wave, qmax, n_want):
    from glint.predict import _hkl_grid
    g, qv = _hkl_grid(R, qmax)
    qn2 = np.einsum("ij,ij->i", qv, qv)
    exc = np.abs(qv[:, 2] + 0.5 * wave * qn2)
    return qv[np.argsort(exc)[:n_want]]


def q_from_fs_ss(f, s, p, clen_m, wave):
    A = np.array([[p["fs"][0], p["ss"][0]], [p["fs"][1], p["ss"][1]]])
    rhs = np.array([f - p["min_fs"], s - p["min_ss"]]) @ A.T
    Xxy = (rhs + np.array([p["cx"], p["cy"]])) / p["res"]
    d = np.array([Xxy[0], Xxy[1], clen_m + p["coffset"]])
    return (d / np.linalg.norm(d) - Z_HAT) / wave


def bits(a):
    return np.ascontiguousarray(a).tobytes()


def identical(r1, r2):
    return all(bits(x) == bits(y) for x, y in zip(r1, r2))


# ===================================================================================== main ======
def main():
    SIM = os.environ.get("GLINT_SIM", "/sdf/home/s/smarches/glint_sim")
    t = np.load(f"{SIM}/truth.npz")
    N = int(t["det_n"]); pix_mm = float(t["pix_mm"]); dist_mm = float(t["dist_mm"])
    wave = float(t["wave_A"]); cell = t["cell"]
    clen = dist_mm / 1000.0
    from glint_fast import cell_to_Ar
    from glint.predict import recip_from_M
    Mc = cell_to_Ar(float(cell[0]), float(cell[1]), float(cell[2]), 90, 90, 90)

    P1 = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
               res=1.0 / (pix_mm / 1000.0), cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5),
               coffset=0.0, min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
    GEOMS = {1: P1, 4: tiled_panels(N, 2, pix_mm), 16: tiled_panels(N, 4, pix_mm),
             64: tiled_panels(N, 8, pix_mm), "16j": tiled_panels(N, 4, pix_mm, jitter=True, seed=7)}
    CONST = {k: PanelConst(v) for k, v in GEOMS.items()}
    VAR = {"V1a": project_q_v1a, "V1b": project_q_v1b, "V1c": project_q_v1c, "V2": project_q_v2}
    qmax = 0.5
    rng = np.random.default_rng(0)

    print(f"host numpy {np.__version__}  det {N}x{N}  pix {pix_mm} mm  clen {clen:.4f} m  wave {wave:.4f} A")

    # ---------------------------------------------------------------- 1. CORRECTNESS -----------
    print("\n=== CORRECTNESS: bit-identity vs current project_q (raw bytes of fs, ss, panel) ===")
    fails = 0; ncase = 0; v3dev = 0.0; v3pan = 0
    for key in GEOMS:
        pan_ls = GEOMS[key]; C = CONST[key]
        onp = tot = nback = 0
        for it in range(25):
            R = recip_from_M(Mc) @ rand_rot(np.random.default_rng(100 + it)).T
            q = make_q(R, wave, qmax, 900)
            far = rng.normal(size=(60, 3)) * 0.4                      # miss every panel
            back = np.array([[0.0, 0.0, -2.0 / wave]] * 5) + rng.normal(size=(5, 3)) * 1e-3
            q = np.vstack([q, far, back])
            r0 = project_q_ref(q, pan_ls, clen, wave)
            for nm, fn in VAR.items():
                if not identical(r0, fn(q, C, clen, wave)):
                    fails += 1; print(f"  {nm} MISMATCH geom={key} it={it}")
            r3 = project_q_v3(q, C, clen, wave)
            m = r0[2] >= 0
            v3dev = max(v3dev, float(np.nanmax(np.abs(r0[0][m] - r3[0][m]))),
                        float(np.nanmax(np.abs(r0[1][m] - r3[1][m]))))
            v3pan += int((r0[2] != r3[2]).sum())
            ncase += 1
            onp += int(m.sum()); tot += len(q)
            nback += int((wave * q[:, 2] + 1.0 <= 1e-6).sum())
        print(f"  geom {key!r:>5} (P={len(pan_ls):2d}): 25 orientations x {tot//25} q,  on-panel "
              f"{100*onp/tot:5.1f}%,  back-scattered {nback}  ->  {'OK' if fails==0 else 'FAIL'}")

    print("\n  boundary stress: rays aimed AT the <=/>= decision points (panel edges +- 0, 1e-12, 1e-9, 1e-6 px)")
    for key in (1, 16, "16j"):
        pan_ls = GEOMS[key]; C = CONST[key]
        qs = []
        for p in pan_ls:
            for eps in (0.0, 1e-12, -1e-12, 1e-9, -1e-9, 1e-6, -1e-6):
                for (f, s) in ((p["min_fs"], p["min_ss"]), (p["max_fs"], p["max_ss"]),
                               (p["min_fs"], p["max_ss"]), (p["max_fs"], p["min_ss"]),
                               (0.5 * (p["min_fs"] + p["max_fs"]), p["min_ss"]),
                               (0.5 * (p["min_fs"] + p["max_fs"]), p["max_ss"])):
                    qs.append(q_from_fs_ss(f + eps, s + eps, p, clen, wave))
                    qs.append(q_from_fs_ss(f + eps, s - eps, p, clen, wave))
        q = np.array(qs)
        r0 = project_q_ref(q, pan_ls, clen, wave)
        oks = {nm: identical(r0, fn(q, C, clen, wave)) for nm, fn in VAR.items()}
        r3 = project_q_v3(q, C, clen, wave)
        fails += sum(0 if v else 1 for v in oks.values())
        npan = len(np.unique(r0[2][r0[2] >= 0]))
        verd = "  ".join("%s %s" % (k, "OK" if v else "FAIL") for k, v in oks.items())
        print(f"    geom {key!r:>5}: {len(q):5d} rays, {int((r0[2]>=0).sum())} on-panel over {npan} panels"
              f"   {verd}   | V3 panel-flips {int((r0[2]!=r3[2]).sum())}")

    print(f"\n  BIT-IDENTICAL: {'PASS' if fails == 0 else 'FAIL (%d)' % fails}   "
          f"({ncase} orientation x geometry cases + 3 boundary sets)")
    print(f"  V3 (no normalisation): max |dfs|,|dss| = {v3dev:.3e} px, panel reassignments = {v3pan}"
          f"  -> NOT bit-identical")

    # ------------------------------------------------- 2. where the reference's time actually is --
    Rp = recip_from_M(Mc) @ rand_rot(np.random.default_rng(5)).T
    q740 = make_q(Rp, wave, qmax, 740)

    def bench(fn, reps=400, warm=80):
        for _ in range(warm):
            fn()
        ts = np.empty(reps)
        for i in range(reps):
            t0 = time.perf_counter(); fn(); ts[i] = time.perf_counter() - t0
        return 1e3 * ts.min(), 1e3 * np.median(ts)

    print("\n=== reference project_q, per-statement cost at n=740, P=1 (min of 400, ms) ===")
    p = P1[0]
    _q = np.asarray(q740, float)
    _s0 = wave * _q + Z_HAT
    _s = _s0 / np.linalg.norm(_s0, axis=1, keepdims=True)
    _fwd = _s[:, 2] > 1e-6
    _Zp = clen + p["coffset"]
    _t = np.where(_fwd, _Zp / np.where(_fwd, _s[:, 2], 1.0), np.nan)
    _X = _s * _t[:, None]
    _A = np.array([[p["fs"][0], p["ss"][0]], [p["fs"][1], p["ss"][1]]])
    _rhs = p["res"] * _X[:, :2] - np.array([p["cx"], p["cy"]])
    _W = np.linalg.inv(_A).T
    _lf = _rhs @ _W
    _f = p["min_fs"] + _lf[:, 0]; _ss_ = p["min_ss"] + _lf[:, 1]
    _on = _fwd & (_f >= 0) & (_f <= 1023) & (_ss_ >= 0) & (_ss_ <= 1023)
    _pan = np.full(740, -1, int); _fsarr = np.full(740, np.nan)
    steps = [
        ("np.asarray(q,float)", lambda: np.asarray(q740, float)),
        ("wl*q + Z_HAT", lambda: wave * _q + Z_HAT),
        ("np.linalg.norm(axis=1)", lambda: np.linalg.norm(_s0, axis=1, keepdims=True)),
        ("  = sqrt(add.reduce(s*s))", lambda: np.sqrt(np.add.reduce(_s0 * _s0, axis=1, keepdims=True))),
        ("s_hat / norm", lambda: _s0 / np.linalg.norm(_s0, axis=1, keepdims=True)),
        ("3x np.full(n)", lambda: (np.full(740, np.nan), np.full(740, np.nan), np.full(740, -1, int))),
        ("fwd = s[:,2] > 1e-6", lambda: _s[:, 2] > 1e-6),
        ("  sz.min() > 1e-6", lambda: _s[:, 2].min() > 1e-6),
        ("2x np.where -> t", lambda: np.where(_fwd, _Zp / np.where(_fwd, _s[:, 2], 1.0), np.nan)),
        ("  bare Zp / sz", lambda: _Zp / _s[:, 2]),
        ("X = s_hat * t[:,None]", lambda: _s * _t[:, None]),
        ("  s[:,:2] * t[:,None]", lambda: _s[:, :2] * _t[:, None]),
        ("np.array(A 2x2)", lambda: np.array([[p["fs"][0], p["ss"][0]], [p["fs"][1], p["ss"][1]]])),
        ("np.linalg.inv(A).T", lambda: np.linalg.inv(_A).T),
        ("np.array([cx,cy])", lambda: np.array([p["cx"], p["cy"]])),
        ("res*X[:,:2] - corner", lambda: p["res"] * _X[:, :2] - np.array([p["cx"], p["cy"]])),
        ("rhs @ W  (n,2)@(2,2)", lambda: _rhs @ _W),
        ("f,s = min + lfls[:,i]", lambda: (p["min_fs"] + _lf[:, 0], p["min_ss"] + _lf[:, 1])),
        ("on: 4 cmp + 4 and", lambda: _fwd & (_f >= 0) & (_f <= 1023) & (_ss_ >= 0) & (_ss_ <= 1023)),
        ("  on: 4 cmp + 3 &=", lambda: _v1b_mask(_f, _ss_)),
        ("take = on & (pan<0)", lambda: _on & (_pan < 0)),
        ("gather+scatter x3", lambda: _gs(_fsarr, _f, _on, _pan)),
        ("  3x np.where", lambda: (np.where(_on, _f, np.nan), np.where(_on, _ss_, np.nan),
                                   np.where(_on, 0, -1))),
        ("np.errstate(...) ctx", lambda: _errctx()),
    ]
    tot = 0.0
    for nm, fn in steps:
        a, _ = bench(fn, reps=400, warm=80)
        if not nm.startswith("  "):
            tot += a
        print(f"  {nm:<28} {a*1e3:8.2f} us")
    print(f"  {'SUM of reference statements':<28} {tot*1e3:8.2f} us")

    # ---------------------------------------------------------------- 3. OP COUNT --------------
    print("\n=== numpy ops issued per call (auto-traced: ufuncs, array-functions, constructors, fancy index) ===")
    print(f"  {'P':>5} {'groups':>7} | {'ref':>6} {'V1b':>6} {'V1c':>6} {'V2':>6} | {'V1c cut':>9} {'V2 cut':>9}")
    for key in (1, 4, 16, 64, "16j"):
        c0 = count_ops(lambda a, g, c, l: project_q_ref(a, GEOMS[key], c, l), q740, None, clen, wave)
        cb = count_ops(project_q_v1b, q740, CONST[key], clen, wave)
        cc = count_ops(project_q_v1c, q740, CONST[key], clen, wave)
        c2 = count_ops(project_q_v2, q740, CONST[key], clen, wave)
        print(f"  {len(GEOMS[key]):>5} {CONST[key].ngroup:>7} | {c0:>6} {cb:>6} {cc:>6} {c2:>6} | "
              f"{c0/cc:>8.2f}x {c0/c2:>8.2f}x")

    # ---------------------------------------------------------------- 4. TIMING ----------------
    print("\n=== wall clock, P=1, ms/call  [min | median of 400 reps] ===")
    NS = (100, 740, 2237, 7444)
    qs = {n: make_q(Rp, wave, qmax, n) for n in NS}
    hdr = f"  {'n':>6} |" + "".join(f"{k+' min':>10}{k+' med':>10}" for k in ("ref", "V1b", "V1c", "V2", "V3"))
    print(hdr)
    T = {k: {} for k in ("ref", "V1b", "V1c", "V2", "V3")}
    for n in NS:
        q = qs[n]
        row = f"  {n:>6} |"
        for nm, call in (("ref", lambda: project_q_ref(q, P1, clen, wave)),
                         ("V1b", lambda: project_q_v1b(q, CONST[1], clen, wave)),
                         ("V1c", lambda: project_q_v1c(q, CONST[1], clen, wave)),
                         ("V2", lambda: project_q_v2(q, CONST[1], clen, wave)),
                         ("V3", lambda: project_q_v3(q, CONST[1], clen, wave))):
            a, am = bench(call)
            T[nm][n] = a
            row += f"{a:>10.4f}{am:>10.4f}"
        print(row)

    print(f"\n  {'variant':>8} {'fit t(ms) = a + b n':>34} {'fixed% @740':>12} {'t(740)':>9} {'vs ref':>9}")
    A = np.stack([np.ones(len(NS)), np.array(NS, float)], 1)
    base = None
    for nm in ("ref", "V1b", "V1c", "V2", "V3"):
        y = np.array([T[nm][n] for n in NS])
        co, *_ = np.linalg.lstsq(A, y, rcond=None)
        pn = co[0] + co[1] * 740
        base = pn if base is None else base
        print(f"  {nm:>8} {f'{co[0]:.5f} + {co[1]:.4e} n':>34} {100*co[0]/pn:>11.1f}% {pn:>9.4f} "
              f"{base/pn:>8.2f}x")

    b1 = min(T["V1c"][740], T["V1b"][740])
    print(f"\n  PRODUCTION n=740, P=1:  ref {T['ref'][740]:.4f} ms -> best bit-identical "
          f"{b1:.4f} ms  (save {T['ref'][740]-b1:.4f} ms, {100*(1-b1/T['ref'][740]):.1f}%)"
          f"   [V3 non-identical {T['V3'][740]:.4f} ms, save "
          f"{100*(1-T['V3'][740]/T['ref'][740]):.1f}%]")

    print("\n=== wall clock vs PANEL COUNT at n=740, ms/call (min of 200) ===")
    print(f"  {'P':>5} {'grp':>4} | {'ref':>9} {'V1b':>9} {'V1c':>9} {'V2':>9} | {'best':>5} {'speedup':>8}")
    q = qs[740]
    for key in (1, 4, 16, 64, "16j"):
        pl = GEOMS[key]; C = CONST[key]
        a, _ = bench(lambda: project_q_ref(q, pl, clen, wave), reps=200, warm=50)
        va, _ = bench(lambda: project_q_v1b(q, C, clen, wave), reps=200, warm=50)
        vb, _ = bench(lambda: project_q_v1c(q, C, clen, wave), reps=200, warm=50)
        v2, _ = bench(lambda: project_q_v2(q, C, clen, wave), reps=200, warm=50)
        best = min(va, vb, v2)
        nm = "V1b" if best == va else ("V1c" if best == vb else "V2")
        print(f"  {len(pl):>5} {C.ngroup:>4} | {a:>9.4f} {va:>9.4f} {vb:>9.4f} {v2:>9.4f} | "
              f"{nm:>5} {a/best:>7.2f}x")

    # ---------------------------------------------------------------- 5. END TO END ------------
    print("\n=== end-to-end predict() on the driver path (GPU gate + host project_q), tol=0.002 ===")
    try:
        import cupy as cp
        import glint.stream_driver as sd
        TOL = float(os.environ.get("TOL", "0.002"))
        grid = sd.HKLGrid(Mc, dmin=2.0)
        Mrand = np.linalg.inv(recip_from_M(Mc) @ rand_rot(np.random.default_rng(11)).T)
        npred = len(grid.predict(Mrand, P1, clen, wave, tol=TOL))

        def tgpu(fn, reps=80, warm=25):
            for _ in range(warm):
                fn()
            cp.cuda.Stream.null.synchronize()
            ts = np.empty(reps)
            for i in range(reps):
                cp.cuda.Stream.null.synchronize()
                t0 = time.perf_counter(); fn(); cp.cuda.Stream.null.synchronize()
                ts[i] = time.perf_counter() - t0
            return 1e3 * ts.min(), 1e3 * np.median(ts)

        qsurv = _survivor_q(grid, Mrand, wave, TOL)
        pq, _ = bench(lambda: project_q_ref(qsurv, P1, clen, wave), reps=300, warm=60)
        pqn, _ = bench(lambda: project_q_v1b(qsurv, CONST[1], clen, wave), reps=300, warm=60)
        print(f"  survivors reaching project_q: n={len(qsurv)} q  ->  {npred} on-panel reflections (P=1)")
        print(f"  project_q(ref) alone on those n={len(qsurv)}: {pq:.4f} ms   ->  V1b {pqn:.4f} ms")
        print(f"  {'P':>5} | {'ref':>8}{'med':>8} | {'V1b':>8}{'med':>8} | {'V2':>8}{'med':>8} | "
              f"{'saved':>8} {'%':>6} {'bit-id':>7}")
        for key in (1, 4, 16, 64):
            pl = GEOMS[key]; C = CONST[key]
            sd.project_q = project_q_ref
            a, am = tgpu(lambda: grid.predict(Mrand, pl, clen, wave, tol=TOL))
            pa = grid.predict(Mrand, pl, clen, wave, tol=TOL)
            sd.project_q = lambda qq, pp, cc, ll, _C=C: project_q_v1b(qq, _C, cc, ll)
            b, bm = tgpu(lambda: grid.predict(Mrand, pl, clen, wave, tol=TOL))
            pb = grid.predict(Mrand, pl, clen, wave, tol=TOL)
            sd.project_q = lambda qq, pp, cc, ll, _C=C: project_q_v2(qq, _C, cc, ll)
            c, cm = tgpu(lambda: grid.predict(Mrand, pl, clen, wave, tol=TOL))
            pc = grid.predict(Mrand, pl, clen, wave, tol=TOL)
            sd.project_q = project_q_ref
            same = (bits(pa) == bits(pb)) and (bits(pa) == bits(pc))
            bb = min(b, c)
            print(f"  {len(pl):>5} | {a:>8.4f}{am:>8.4f} | {b:>8.4f}{bm:>8.4f} | {c:>8.4f}{cm:>8.4f} | "
                  f"{a-bb:>8.4f} {100*(1-bb/a):>5.1f}% {'OK' if same else 'DIFF':>7}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  end-to-end skipped: {e}")


def _survivor_q(grid, M, wave, tol):
    """The exact q array the driver's predict() hands to project_q."""
    from glint.predict import recip_from_M
    import cupy as cp
    R = recip_from_M(M)
    nhkl = grid.g.shape[0]; grid._gcnt[0] = 0
    import glint.stream_driver as sd
    Rf = cp.ascontiguousarray(cp.asarray(R).ravel())
    tpb = 256; blocks = (nhkl + tpb - 1) // tpb
    sd._GATE_KERNEL((blocks,), (tpb,), (grid._ggr, np.int32(nhkl), Rf, np.float64(wave),
                    np.float64(grid.qmax * grid.qmax), np.float64(tol), grid._gout.ravel(),
                    grid._gcnt, np.int32(nhkl)))
    nsv = int(grid._gcnt[0])
    C = cp.asnumpy(grid._gout[:nsv])
    C = C[np.argsort(C[:, 0], kind="stable")]
    return np.ascontiguousarray(C[:, 1:4])


def _v1b_mask(f, s):
    on = f >= 0.0; on &= f <= 1023.0; on &= s >= 0.0; on &= s <= 1023.0
    return on


def _gs(fs, f, take, pan):
    fs[take] = f[take]; pan[take] = 0
    return fs


def _errctx():
    with np.errstate(divide="ignore", invalid="ignore"):
        pass


if __name__ == "__main__":
    main()
