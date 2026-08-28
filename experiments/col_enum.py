"""Smarter hkl enumeration for HKLGrid.predict: per-(h,k) COLUMN quadratic solve
instead of testing every hkl in the ball.

Parts:
  A. algebra verification (exc via matvec vs via the alpha/beta/gamma parabola)
  B. numpy reference column enumerator; EXACT survivor-set match vs the production
     GPU gate kernel on >=20 real orientations
  C. operation counts (apples-to-apples) + host wall-clock
  D. GPU column-kernel prototype (one thread per (h,k) column), timed correctly
  E. where the time in predict() actually goes (kernel / sync / D2H / argsort / project_q)
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import cupy as cp
from glint_fast import cell_to_Ar
import glint.stream_driver as sd
from glint.predict import recip_from_M, _hkl_grid, project_q, _canonical_axes

SIM = os.environ.get("GLINT_SIM", "/sdf/home/s/smarches/glint_sim")
imgs = np.load(f"{SIM}/images.npy"); t = np.load(f"{SIM}/truth.npz")
N = int(t["det_n"]); pix_mm = float(t["pix_mm"]); dist_mm = float(t["dist_mm"]); wave = float(t["wave_A"])
cell = t["cell"]
panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
               res=1.0 / (pix_mm / 1000.0), cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5),
               coffset=0.0, min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
Mc = cell_to_Ar(float(cell[0]), float(cell[1]), float(cell[2]), 90, 90, 90)
clen = dist_mm / 1000.0
DMIN = 2.0
TOL = float(os.environ.get("TOL", "0.002"))
QMAX = 1.0 / DMIN
print(f"cell={np.round(np.asarray(cell,float),2)}  wave={wave:.4f}A  dmin={DMIN}  tol={TOL}  "
      f"det={N}x{N}  clen={clen:.4f}m")


# =================================================================== A. algebra ==================
def algebra_check():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        R = rng.normal(size=(3, 3)) * 0.02
        w = rng.uniform(0.5, 2.0)
        hkl = rng.integers(-40, 41, size=(500, 3)).astype(float)
        q = hkl @ R
        qn2 = np.einsum("ij,ij->i", q, q)
        exc_ref = q[:, 2] + 0.5 * w * qn2
        # parabola form
        Rh, Rk, Rv = R[0], R[1], R[2]
        h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
        u = h[:, None] * Rh + k[:, None] * Rk
        v = Rv
        alpha = 0.5 * w * (v @ v)
        beta = v[2] + w * (u @ v)
        gamma = u[:, 2] + 0.5 * w * np.einsum("ij,ij->i", u, u)
        exc_par = alpha * l * l + beta * l + gamma
        # |q|^2 form
        qn2_par = np.einsum("ij,ij->i", u, u) + 2 * l * (u @ v) + l * l * (v @ v)
        d = max(np.max(np.abs(exc_par - exc_ref) / np.maximum(np.abs(exc_ref), 1e-12)),
                np.max(np.abs(qn2_par - qn2) / np.maximum(qn2, 1e-12)))
        worst = max(worst, d)
    print(f"\n[A] algebra: exc = alpha*l^2+beta*l+gamma and |q|^2 = |u|^2+2l(u.v)+l^2|v|^2")
    print(f"    alpha = 0.5*wave*|v|^2 (h,k-independent), beta = v_z + wave*(u.v), "
          f"gamma = u_z + 0.5*wave*|u|^2")
    print(f"    max RELATIVE disagreement vs the matvec form over 200 random (R,wave), "
          f"500 hkl each: {worst:.3e}  -> identical to roundoff")


# ========================================== B. numpy column enumerator ===========================
def column_candidates(R, wave, qmax, tol, pad=0):
    """All integer (h,k,l) that CAN satisfy (|q|<=qmax and |exc|<tol), by solving quadratics
    per (h,k) column. `pad` widens each integer l-run by that many units on each side (a guard
    so the float interval arithmetic cannot lose a boundary reflection). Returns (cand, n_cols,
    n_cols_rect) where cand is (m,3) int."""
    Rh, Rk, v = R[0], R[1], R[2]
    v2 = v @ v
    alpha = 0.5 * wave * v2
    # ---- which (h,k) columns can reach inside the qmax ball at all?
    # min_l |u + l v| = |P u| with P = I - vv^T/|v|^2.  a = P Rh, b = P Rk.
    P = np.eye(3) - np.outer(v, v) / v2
    a = P @ Rh; b = P @ Rk
    aa = a @ a; bb = b @ b; ab = a @ b
    det2 = aa * bb - ab * ab                                  # Gram det of the projected 2-D lattice
    Hmax = int(np.floor(qmax * np.sqrt(bb / det2))) + 1
    Kmax = int(np.floor(qmax * np.sqrt(aa / det2))) + 1
    hh, kk = np.meshgrid(np.arange(-Hmax, Hmax + 1), np.arange(-Kmax, Kmax + 1), indexing="ij")
    hh = hh.ravel(); kk = kk.ravel()
    n_rect = hh.size
    # exact per-column reachability: h^2 aa + 2hk ab + k^2 bb <= qmax^2
    reach = hh * hh * aa + 2.0 * hh * kk * ab + kk * kk * bb <= qmax * qmax * (1.0 + 1e-9)
    hh = hh[reach]; kk = kk[reach]
    n_cols = hh.size
    # ---- per column: u, and the three quadratics in l
    u = hh[:, None] * Rh + kk[:, None] * Rk                   # (nc,3)
    uv = u @ v
    u2 = np.einsum("ij,ij->i", u, u)
    uz = u[:, 2]
    beta = v[2] + wave * uv
    gamma = uz + 0.5 * wave * u2
    # resolution: v2 l^2 + 2 uv l + (u2 - qmax^2) <= 0
    dr = uv * uv - v2 * (u2 - qmax * qmax)
    dr = np.maximum(dr, 0.0)
    sr = np.sqrt(dr)
    lr1 = (-uv - sr) / v2; lr2 = (-uv + sr) / v2
    # exc < tol : alpha l^2 + beta l + (gamma - tol) < 0  -> between roots
    d1 = beta * beta - 4.0 * alpha * (gamma - tol)
    ok1 = d1 > 0.0
    s1 = np.sqrt(np.maximum(d1, 0.0))
    p1 = (-beta - s1) / (2 * alpha); p2 = (-beta + s1) / (2 * alpha)
    lo = np.maximum(lr1, p1); hi = np.minimum(lr2, p2)
    lo = np.where(ok1, lo, 1.0); hi = np.where(ok1, hi, -1.0)  # empty when the parabola never dips
    # exc > -tol : alpha l^2 + beta l + (gamma + tol) > 0  -> OUTSIDE roots (all l if no roots).
    # With no roots the exclusion is empty: e1 must not clip seg0 and e2 must kill seg1, so both
    # are set to hi+1 (finite -- np.inf would blow up the integer cast below).
    d2 = beta * beta - 4.0 * alpha * (gamma + tol)
    ok2 = d2 > 0.0
    s2r = np.sqrt(np.maximum(d2, 0.0))
    e1 = np.where(ok2, (-beta - s2r) / (2 * alpha), hi + 1.0)
    e2 = np.where(ok2, (-beta + s2r) / (2 * alpha), hi + 1.0)
    segs = [(lo, np.minimum(hi, e1)), (np.maximum(lo, e2), hi)]
    hs, ks, ls = [], [], []
    for slo, shi in segs:
        li = np.ceil(np.clip(slo, -1e6, 1e6)).astype(np.int64) - pad
        lj = np.floor(np.clip(shi, -1e6, 1e6)).astype(np.int64) + pad
        cnt = np.maximum(lj - li + 1, 0)
        tot = int(cnt.sum())
        if tot == 0:
            continue
        off = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        j = np.arange(tot) - np.repeat(off, cnt)
        ls.append(np.repeat(li, cnt) + j)
        hs.append(np.repeat(hh, cnt)); ks.append(np.repeat(kk, cnt))
    if not hs:
        return np.zeros((0, 3), np.int64), n_cols, n_rect
    cand = np.stack([np.concatenate(hs), np.concatenate(ks), np.concatenate(ls)], 1)
    if pad:                                                    # padded segments can overlap
        key = (cand[:, 0] + 512) * (1 << 40) + (cand[:, 1] + 512) * (1 << 20) + (cand[:, 2] + 512)
        cand = cand[np.unique(key, return_index=True)[1]]
    return cand, n_cols, n_rect


def gate_host(cand, R, wave, qmax, tol):
    """The exact gate, same expressions as the full-grid host path."""
    q = cand.astype(float) @ R
    qn2 = np.einsum("ij,ij->i", q, q)
    exc = q[:, 2] + 0.5 * wave * qn2
    keep = (qn2 <= qmax * qmax) & (np.abs(exc) < tol) & np.any(cand != 0, axis=1)
    return cand[keep], q[keep], qn2[keep], exc[keep]


def key_of(hkl):
    h = hkl.astype(np.int64)
    return (h[:, 0] + 512) * (1 << 40) + (h[:, 1] + 512) * (1 << 20) + (h[:, 2] + 512)


# ================================================ get real orientations ==========================
def real_orientations(nwant=24):
    Ms = []
    orig = sd.HKLGrid.predict

    def spy(self, M_or_R, *a, **kw):
        Ms.append(np.asarray(M_or_R, float))
        return orig(self, M_or_R, *a, **kw)
    sd.HKLGrid.predict = spy
    try:
        drv = sd.StreamDriver(Mc, panels, clen, wave, (N, N), dtype=imgs.dtype,
                              B=32, dmin=DMIN, tol=TOL)
        for f in imgs[:min(len(imgs), 64)]:
            drv.push(f)
        drv.flush(); cp.cuda.Stream.null.synchronize()
    finally:
        sd.HKLGrid.predict = orig
    return Ms[:nwant]


# ========================================== GPU column kernel ====================================
_COL_SRC = r"""
extern "C" __global__ void col_gate(const double* R, double wave, double qmax2, double tol,
                                    int Hmax, int Kmax, double alpha,
                                    double aa, double bb, double ab,
                                    double* out, int* counter, int cap){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int nk = 2*Kmax+1, nh = 2*Hmax+1;
  if (i >= nh*nk) return;
  int h = i/nk - Hmax;
  int k = i%nk - Kmax;
  double dh=(double)h, dk=(double)k;
  // column reachable?
  // reachability of this column, with a relative slack: a column grazing the qmax sphere must not
  // be culled by roundoff, the exact gate at the bottom is what actually decides.
  if (dh*dh*aa + 2.0*dh*dk*ab + dk*dk*bb > qmax2*(1.0+1e-9)) return;
  double vx=R[6], vy=R[7], vz=R[8];
  double v2 = vx*vx+vy*vy+vz*vz;
  double ux = dh*R[0]+dk*R[3];
  double uy = dh*R[1]+dk*R[4];
  double uz = dh*R[2]+dk*R[5];
  double uv = ux*vx+uy*vy+uz*vz;
  double u2 = ux*ux+uy*uy+uz*uz;
  double beta  = vz + wave*uv;
  double gamma = uz + 0.5*wave*u2;
  double dr = uv*uv - v2*(u2 - qmax2);
  if (dr < 0.0) return;
  double sr = sqrt(dr);
  double lr1 = (-uv - sr)/v2, lr2 = (-uv + sr)/v2;
  double d1 = beta*beta - 4.0*alpha*(gamma - tol);
  if (d1 <= 0.0) return;                       // parabola never dips below +tol
  double s1 = sqrt(d1);
  double p1 = (-beta - s1)/(2.0*alpha), p2 = (-beta + s1)/(2.0*alpha);
  double lo = fmax(lr1, p1), hi = fmin(lr2, p2);
  double d2 = beta*beta - 4.0*alpha*(gamma + tol);
  // no roots => the exclusion is empty: e1 must not clip seg0 and e2 must kill seg1, so hi+1 for both
  double e1 = hi + 1.0, e2 = hi + 1.0;
  if (d2 > 0.0){ double s2 = sqrt(d2); e1 = (-beta - s2)/(2.0*alpha); e2 = (-beta + s2)/(2.0*alpha); }
  double segl[2], segh[2];
  segl[0]=lo; segh[0]=fmin(hi,e1);
  segl[1]=fmax(lo,e2); segh[1]=hi;
  long prev_hi = 0; int have_prev = 0;         // last l already emitted (dedup across the two segs)
  for (int s=0; s<2; ++s){
    // NOTE: do NOT skip on segl>segh. The +/-1 guard deliberately emits candidates around a
    // float-empty interval, and the exact gate below sorts them out. Skipping here lost 1
    // reflection on 4 of 24 real orientations.
    if (segh[s] - segl[s] < -4.0) continue;    // hopelessly empty; guard width is only 1
    long li = (long)ceil(segl[s]) - 1;         // +/-1 guard, then the EXACT gate below
    long lj = (long)floor(segh[s]) + 1;
    if (have_prev && li <= prev_hi) li = prev_hi + 1;   // guards can overlap; do not double-emit
    if (li > lj) continue;                     // nothing emitted -> must NOT arm prev_hi (see below)
    // prev_hi is armed only when this segment ACTUALLY emits. Arming it on an empty segment
    // truncated the second segment and silently lost 1 reflection on 4 of 24 real orientations.
    if (!have_prev || lj > prev_hi){ prev_hi = lj; have_prev = 1; }
    for (long l=li; l<=lj; ++l){
      if (h==0 && k==0 && l==0) continue;
      double dl=(double)l;
      double qx=dh*R[0]+dk*R[3]+dl*R[6];
      double qy=dh*R[1]+dk*R[4]+dl*R[7];
      double qz=dh*R[2]+dk*R[5]+dl*R[8];
      double qn2=qx*qx+qy*qy+qz*qz;
      double exc=qz+0.5*wave*qn2;
      if (qn2<=qmax2 && fabs(exc)<tol){
        int p=atomicAdd(counter,1);
        if (p<cap){ out[8*p]=dh; out[8*p+1]=dk; out[8*p+2]=dl;
                    out[8*p+3]=qx; out[8*p+4]=qy; out[8*p+5]=qz;
                    out[8*p+6]=qn2; out[8*p+7]=exc; }
      }
    }
  }
}"""
_COL_KERNEL = cp.RawKernel(_COL_SRC, "col_gate")


class ColGrid:
    def __init__(self, Mc, dmin, tol):
        self.qmax = 1.0 / float(dmin); self.tol = float(tol)
        self.cap = 65536
        self._out = cp.empty((self.cap, 8), cp.float64)
        self._cnt = cp.zeros(1, cp.int32)
        _, _, _, self.Hmax, self.Kmax, self.ncol = self._geom(recip_from_M(np.asarray(Mc, float)))

    def _geom(self, R):
        """Column-reachability ellipse constants. MUST be recomputed per frame: the per-frame R is a
        rotation AND a slightly refined cell, and a stale ellipse from the reference cell culls
        columns that graze the qmax sphere (measured: lost 1 reflection with l=0 and |q|/qmax>0.997
        on 4 of 24 real orientations)."""
        v = R[2]; v2 = v @ v
        P = np.eye(3) - np.outer(v, v) / v2
        a = P @ R[0]; b = P @ R[1]
        aa = a @ a; bb = b @ b; ab = a @ b
        det2 = aa * bb - ab * ab
        Hmax = int(np.floor(self.qmax * np.sqrt(bb / det2))) + 1
        Kmax = int(np.floor(self.qmax * np.sqrt(aa / det2))) + 1
        return aa, bb, ab, Hmax, Kmax, (2 * Hmax + 1) * (2 * Kmax + 1)

    def launch(self, R, wave, tpb=128):
        self._cnt[0] = 0
        Rf = cp.ascontiguousarray(cp.asarray(R).ravel())
        alpha = 0.5 * wave * float(R[2] @ R[2])
        aa, bb, ab, Hmax, Kmax, ncol = self._geom(R)
        blocks = (ncol + tpb - 1) // tpb
        _COL_KERNEL((blocks,), (tpb,),
                    (Rf, np.float64(wave), np.float64(self.qmax ** 2), np.float64(self.tol),
                     np.int32(Hmax), np.int32(Kmax), np.float64(alpha),
                     np.float64(aa), np.float64(bb), np.float64(ab),
                     self._out.ravel(), self._cnt, np.int32(self.cap)))

    def fetch(self):
        n = int(self._cnt[0])
        return cp.asnumpy(self._out[:n])


# ======================================================================== run ====================
algebra_check()

grid = sd.HKLGrid(Mc, DMIN, gpu=True)
NHKL = grid.g.shape[0]
R_ref = recip_from_M(Mc)
col = ColGrid(Mc, DMIN, TOL)
print(f"\n[B] full grid (margin 1.02, hoisted once): {NHKL} hkl point tests per frame")
print(f"    column rectangle: (2*{col.Hmax}+1) x (2*{col.Kmax}+1) = {col.ncol} columns")

Ms = real_orientations(24)
print(f"    real orientations captured from StreamDriver on the sim: {len(Ms)}")
if len(Ms) < 20:
    rng = np.random.default_rng(1)
    while len(Ms) < 24:
        A = rng.normal(size=(3, 3)); Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        Ms.append(Mc @ Q)
    print(f"    (topped up to {len(Ms)} with random rotations of the reference cell)")

print(f"\n{'#':>3} {'gate n':>7} {'raw n':>7} {'safe n':>7} {'gpucol':>7} "
      f"{'cols':>6} {'cand':>7} {'miss':>5} {'extra':>6}")
bad = 0
tot_cols = tot_cand = tot_surv = 0
for i, M in enumerate(Ms):
    Mcan = _canonical_axes(M)
    R = recip_from_M(Mcan)
    # ---- production path (GPU gate kernel over the full grid)
    pred = grid.predict(Mcan, panels, clen, wave, tol=TOL)
    # recompute the pre-projection survivor set the same way predict does, so we compare the GATE
    grid._gcnt[0] = 0
    Rf = cp.ascontiguousarray(cp.asarray(R).ravel())
    tpb = 256; blocks = (NHKL + tpb - 1) // tpb
    sd._GATE_KERNEL((blocks,), (tpb,),
                    (grid._ggr, np.int32(NHKL), Rf, np.float64(wave),
                     np.float64(grid.qmax ** 2), np.float64(TOL),
                     grid._gout.ravel(), grid._gcnt, np.int32(NHKL)))
    n_ref = int(grid._gcnt[0])
    C = cp.asnumpy(grid._gout[:n_ref])
    hkl_ref = grid.g[C[:, 0].astype(np.int64)]
    # ---- raw analytic (no guard, no re-check)
    cand0, ncols, nrect = column_candidates(R, wave, grid.qmax, TOL, pad=0)
    nz = np.any(cand0 != 0, axis=1)
    hkl_raw = cand0[nz]
    # ---- safe analytic (guard +-1, then the exact gate on candidates only)
    cand1, _, _ = column_candidates(R, wave, grid.qmax, TOL, pad=1)
    hkl_safe, _, _, _ = gate_host(cand1, R, wave, grid.qmax, TOL)
    # ---- GPU column kernel
    col.launch(R, wave); Cg = col.fetch()
    hkl_gpu = Cg[:, :3].astype(np.int64)

    kr = set(key_of(hkl_ref).tolist())
    ks = set(key_of(hkl_safe).tolist())
    kg = set(key_of(hkl_gpu).tolist())
    kw_ = set(key_of(hkl_raw).tolist())
    miss = len(kr - ks); extra = len(ks - kr)
    missg = len(kr - kg); extrag = len(kg - kr)
    missr = len(kr - kw_); extrar = len(kw_ - kr)
    if miss or extra or missg or extrag:
        bad += 1
        for kk_ in list(kr - kg)[:3] + list(kg - kr)[:3]:      # name the offending reflection
            j = int(np.where(key_of(hkl_ref) == kk_)[0][0]) if kk_ in kr else -1
            hh_ = hkl_ref[j] if j >= 0 else hkl_gpu[int(np.where(key_of(hkl_gpu) == kk_)[0][0])]
            qd = hh_.astype(float) @ R
            e_ = qd[2] + 0.5 * wave * (qd @ qd)
            print(f"      DIFF hkl={tuple(int(x) for x in hh_)} "
                  f"{'MISSED-by-gpucol' if j>=0 else 'EXTRA-in-gpucol'} "
                  f"exc={e_:+.6e} (tol={TOL})  |q|={np.sqrt(qd@qd):.6f} (qmax={grid.qmax:.6f})")
    tot_cols += ncols; tot_cand += len(cand1); tot_surv += n_ref
    print(f"{i:>3} {n_ref:>7} {len(hkl_raw):>7} {len(hkl_safe):>7} {len(hkl_gpu):>7} "
          f"{ncols:>6} {len(cand1):>7} {miss:>5} {extra:>6}"
          + ("" if not (missg or extrag) else f"   GPUCOL miss={missg} extra={extrag}")
          + ("" if not (missr or extrar) else f"   [raw-noguard miss={missr} extra={extrar}]"))
print(f"\n    frames with a safe/gpu mismatch vs the production gate: {bad} / {len(Ms)}")
nm = len(Ms)
print(f"    mean per frame: survivors={tot_surv/nm:.0f}  reachable columns={tot_cols/nm:.0f}  "
      f"padded l-candidates={tot_cand/nm:.0f}")

# --- edge case probes ------------------------------------------------------------------------
R = recip_from_M(_canonical_axes(Ms[0]))
c0, nc0, nr0 = column_candidates(R, wave, grid.qmax, TOL, pad=0)
Rh, Rk, v = R[0], R[1], R[2]
v2 = v @ v; alpha = 0.5 * wave * v2
P = np.eye(3) - np.outer(v, v) / v2
a = P @ Rh; b = P @ Rk
hh, kk = np.meshgrid(np.arange(-col.Hmax, col.Hmax + 1), np.arange(-col.Kmax, col.Kmax + 1), indexing="ij")
hh = hh.ravel(); kk = kk.ravel()
reach = hh * hh * (a @ a) + 2 * hh * kk * (a @ b) + kk * kk * (b @ b) <= grid.qmax ** 2
u = hh[reach][:, None] * Rh + kk[reach][:, None] * Rk
beta = v[2] + wave * (u @ v); gamma = u[:, 2] + 0.5 * wave * np.einsum("ij,ij->i", u, u)
d1 = beta * beta - 4 * alpha * (gamma - TOL)
d2 = beta * beta - 4 * alpha * (gamma + TOL)
print(f"\n[B-edge] on one real orientation, of {reach.sum()} reachable columns:")
print(f"    {int((d1<=0).sum())} have NO roots for exc=+tol (parabola never dips in -> column empty)")
print(f"    {int((d2>0).sum())} DO have roots for exc=-tol (two disjoint l-intervals, not one)")
# how many survivors are clipped by the qmax cut rather than by the exc cut
lo_e = np.where(d1 > 0, (-beta - np.sqrt(np.maximum(d1, 0))) / (2 * alpha), np.nan)
hi_e = np.where(d1 > 0, (-beta + np.sqrt(np.maximum(d1, 0))) / (2 * alpha), np.nan)
uv = u @ v; u2 = np.einsum("ij,ij->i", u, u)
dr = np.maximum(uv * uv - v2 * (u2 - grid.qmax ** 2), 0)
lr1 = (-uv - np.sqrt(dr)) / v2; lr2 = (-uv + np.sqrt(dr)) / v2
clip = np.nansum((lo_e < lr1) | (hi_e > lr2))
print(f"    {int(clip)} columns where the |q|<=qmax cut trims part of the exc interval")
# boundary tightness: closest |exc| to tol among all candidates of frame 0
cq = c0.astype(float) @ R
ce = cq[:, 2] + 0.5 * wave * np.einsum("ij,ij->i", cq, cq)
print(f"    closest approach of |exc| to tol among emitted candidates: "
      f"{np.min(np.abs(np.abs(ce) - TOL)):.3e} (tol={TOL})")


# ============================================================ C/D. cost ==========================
def tmin(fn, reps=30, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return ts.min() * 1e3, np.median(ts) * 1e3


def gtime(fn, reps=50, warm=10):
    """GPU-only elapsed via CUDA events; fn must only LAUNCH work on the null stream."""
    for _ in range(warm):
        fn()
    cp.cuda.Stream.null.synchronize()
    ts = []
    s = cp.cuda.Event(); e = cp.cuda.Event()
    for _ in range(reps):
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(cp.cuda.get_elapsed_time(s, e))
    ts = np.array(ts)
    return ts.min(), np.median(ts)


Mcan0 = _canonical_axes(Ms[0]); R0 = recip_from_M(Mcan0)
Rf0 = cp.ascontiguousarray(cp.asarray(R0).ravel())
tpb = 256; blocks = (NHKL + tpb - 1) // tpb


def launch_full():
    grid._gcnt[0] = 0
    sd._GATE_KERNEL((blocks,), (tpb,),
                    (grid._ggr, np.int32(NHKL), Rf0, np.float64(wave),
                     np.float64(grid.qmax ** 2), np.float64(TOL),
                     grid._gout.ravel(), grid._gcnt, np.int32(NHKL)))


def launch_col():
    col.launch(R0, wave)


print("\n[C] operation counts per frame (the apples-to-apples number)")
print(f"    full-grid gate : {NHKL} point tests (matvec + 2 gates each)")
print(f"    column solve   : {col.ncol} rectangle threads, {tot_cols//nm} reachable columns "
      f"(3 quadratics each), {tot_cand//nm} l-candidates re-gated")
red = NHKL / max(tot_cols // nm + tot_cand // nm, 1)
print(f"    candidate-evaluation reduction: {NHKL} -> {tot_cols//nm + tot_cand//nm}  ({red:.1f}x fewer)")

print("\n[D] wall clock, all inside one allocation (min / median over reps)")
a, b_ = gtime(launch_full)
print(f"    GPU full-grid gate kernel (CUDA events)      : {a*1000:8.1f} / {b_*1000:8.1f} us")
# Fair GPU-only timing: ALL host prep (cp.asarray of R, the ellipse constants) is hoisted out of
# the event window. Host numpy left inside a CUDA-event region shows up as GPU elapsed time,
# because the GPU sits idle between the two event records waiting for the launch.
_aa, _bb, _ab, _Hm, _Km, _nc = col._geom(R0)
_Rc = cp.ascontiguousarray(cp.asarray(R0).ravel())
_alpha = np.float64(0.5 * wave * float(R0[2] @ R0[2]))
_args = (_Rc, np.float64(wave), np.float64(col.qmax ** 2), np.float64(col.tol),
         np.int32(_Hm), np.int32(_Km), _alpha,
         np.float64(_aa), np.float64(_bb), np.float64(_ab),
         col._out.ravel(), col._cnt, np.int32(col.cap))
best_col = None
for tb in (32, 64, 128, 256):
    nb = (_nc + tb - 1) // tb

    def fire(tb=tb, nb=nb):
        col._cnt[0] = 0
        _COL_KERNEL((nb,), (tb,), _args)
    a, b_ = gtime(fire)
    print(f"    GPU column kernel  tpb={tb:<4d} ({nb:4d} blocks)     : {a*1000:8.1f} / {b_*1000:8.1f} us")
    best_col = a if best_col is None else min(best_col, a)
print(f"    -> occupancy: full grid launches {blocks} blocks of {tpb}; the column kernel has only "
      f"{col.ncol} threads total")


def launch_tiny():                                  # same kernel, 1 hkl: pure launch/latency floor
    grid._gcnt[0] = 0
    sd._GATE_KERNEL((1,), (32,),
                    (grid._ggr, np.int32(1), Rf0, np.float64(wave),
                     np.float64(grid.qmax ** 2), np.float64(TOL),
                     grid._gout.ravel(), grid._gcnt, np.int32(NHKL)))


a, b_ = gtime(launch_tiny)
print(f"    SAME kernel over nhkl=1 (latency floor)     : {a*1000:8.1f} / {b_*1000:8.1f} us")
gg = np.ascontiguousarray(grid.g.astype(float))


def host_full():
    q = gg @ R0
    qn2 = np.einsum("ij,ij->i", q, q)
    keep = qn2 <= grid.qmax ** 2
    q2 = q[keep]; qn22 = qn2[keep]
    exc = q2[:, 2] + 0.5 * wave * qn22
    near = np.abs(exc) < TOL
    return q2[near]


def host_col():
    c, _, _ = column_candidates(R0, wave, grid.qmax, TOL, pad=1)
    return gate_host(c, R0, wave, grid.qmax, TOL)


a, b_ = tmin(host_full)
print(f"    HOST numpy full-grid  (not a fair GPU rival): {a:8.3f} / {b_:8.3f} ms")
a, b_ = tmin(host_col)
print(f"    HOST numpy column     (not a fair GPU rival): {a:8.3f} / {b_:8.3f} ms")

# ============================================== E. where predict() time goes =====================
print("\n[E] breakdown of ONE HKLGrid.predict() call (host wall clock of each stage, ms)")


def stage_times(Mcan, reps=30):
    R = recip_from_M(Mcan)
    acc = np.zeros(6)
    for r in range(reps + 5):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        grid._gcnt[0] = 0
        Rf = cp.ascontiguousarray(cp.asarray(R).ravel())
        sd._GATE_KERNEL((blocks,), (tpb,),
                        (grid._ggr, np.int32(NHKL), Rf, np.float64(wave),
                         np.float64(grid.qmax ** 2), np.float64(TOL),
                         grid._gout.ravel(), grid._gcnt, np.int32(NHKL)))
        t1 = time.perf_counter()                       # launch only (async)
        n = int(grid._gcnt[0])                         # BLOCKING SYNC -- absorbs the kernel
        t2 = time.perf_counter()
        C = cp.asnumpy(grid._gout[:n])
        t3 = time.perf_counter()
        C = C[np.argsort(C[:, 0], kind="stable")]
        idx = C[:, 0].astype(np.int64); hkl = grid.g[idx]
        q = C[:, 1:4]; qn2 = C[:, 4]; exc = C[:, 5]
        t4 = time.perf_counter()
        fs, ss, pan = project_q(q, panels, clen, wave)
        t5 = time.perf_counter()
        on = pan >= 0
        out = np.zeros(int(on.sum()), dtype=[("h", int), ("k", int), ("l", int), ("fs", float),
                                             ("ss", float), ("panel", int), ("exc", float), ("res", float)])
        out["h"], out["k"], out["l"] = hkl[on, 0], hkl[on, 1], hkl[on, 2]
        out["fs"], out["ss"], out["panel"] = fs[on], ss[on], pan[on]
        out["exc"] = exc[on]; out["res"] = 1.0 / np.sqrt(qn2[on])
        t6 = time.perf_counter()
        if r >= 5:
            acc += np.array([t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5])
    return acc / reps * 1e3, n


st, nsurv = stage_times(Mcan0)
names = ["launch (async, host only)", "int(_gcnt[0])  BLOCKING SYNC", "cp.asnumpy D2H",
         "argsort + g[idx] gather", "project_q (host numpy loop)", "structured-array assembly"]
for nme, v in zip(names, st):
    print(f"    {nme:<32} {v:8.3f} ms   ({100*v/st.sum():5.1f}%)")
print(f"    {'TOTAL':<32} {st.sum():8.3f} ms   ({nsurv} survivors)")
print("    NOTE: the 'BLOCKING SYNC' row is the kernel's real cost showing up at the sync point;")
print(f"          the CUDA-event kernel time above says the kernel itself is what it is.")

# ================================ F. how the two scale with resolution ===========================
print("\n[F] scaling with dmin  (full grid is O(N^3) in the ball, columns are O(N^2))")
print(f"    {'dmin':>5} {'nhkl':>9} {'cols':>7} {'ratio':>7} | {'full us':>9} {'col us':>9} {'speedup':>8}")
for dm in (2.0, 1.6, 1.3, 1.0):
    g2 = sd.HKLGrid(Mc, dm, gpu=True)
    c2 = ColGrid(Mc, dm, TOL)
    n2 = g2.g.shape[0]
    Rf2 = cp.ascontiguousarray(cp.asarray(R0).ravel())
    bl2 = (n2 + 256 - 1) // 256

    def lf(g2=g2, n2=n2, Rf2=Rf2, bl2=bl2):
        g2._gcnt[0] = 0
        sd._GATE_KERNEL((bl2,), (256,),
                        (g2._ggr, np.int32(n2), Rf2, np.float64(wave),
                         np.float64(g2.qmax ** 2), np.float64(TOL),
                         g2._gout.ravel(), g2._gcnt, np.int32(n2)))
    aa2, bb2, ab2, Hm2, Km2, nc2 = c2._geom(R0)               # host prep hoisted out of the window
    ar2 = (Rf2, np.float64(wave), np.float64(c2.qmax ** 2), np.float64(c2.tol),
           np.int32(Hm2), np.int32(Km2), np.float64(0.5 * wave * float(R0[2] @ R0[2])),
           np.float64(aa2), np.float64(bb2), np.float64(ab2),
           c2._out.ravel(), c2._cnt, np.int32(c2.cap))

    def cf(tb, c2=c2, ar2=ar2, nc2=nc2):
        c2._cnt[0] = 0
        _COL_KERNEL(((nc2 + tb - 1) // tb,), (tb,), ar2)
    bestc = min(gtime(lambda tb=tb: cf(tb))[0] for tb in (32, 64, 128))
    bestf = gtime(lf)[0]
    print(f"    {dm:>5.1f} {n2:>9} {c2.ncol:>7} {n2/c2.ncol:>7.1f} | "
          f"{bestf*1000:>9.1f} {bestc*1000:>9.1f} {bestf/bestc:>8.2f}x")
    del g2, c2
    cp.get_default_memory_pool().free_all_blocks()
