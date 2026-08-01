"""Prototype: fuse project_q INTO the gate kernel.

Today the gate emits every Ewald survivor and the host then runs project_q over all of them, throwing
away the ones that miss. Two costs follow: project_q itself (0.104 ms, 40% of predict) and the fact
that the panel cone is only a CONICAL approximation of the panel, so 233 survivors reach the host
where only 140 are really on-panel.

The gate thread already holds qx,qy,qz in registers. project_q is ~20 more flops. Panel geometry is
frame-invariant, so it uploads once. Fusing therefore (a) deletes project_q from the host, (b) makes
the cull EXACT, and (c) costs nothing in a kernel that is launch-bound.

NOTE this is not the known-negative "on-device project_q". That was measured 3.5x SLOWER as a
SEPARATE kernel -- launch-bound at 2332 survivors. Fusing into an already-running kernel adds no launch.

Verifies exactness against the shipped predict, then times it.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT); sys.path.insert(0, WT + "/experiments")
import cupy as cp
from glint.lattice import cell_to_Ar
import glint.stream_driver as sd
from glint.predict import recip_from_M

LAM, CLEN, RES = 1.322, 0.1, 10000.0
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
PAN = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=RES,
            cx=-287.5, cy=-167.5, coffset=0.0, min_fs=0, max_fs=575, min_ss=0, max_ss=335)]

# per panel: Zp, res, Ainv00, Ainv01, Ainv10, Ainv11, cx, cy, min_fs, max_fs, min_ss, max_ss
NP_FIELDS = 12
_SRC = r"""
extern "C" __global__ void gate_project(const double* g, int nhkl,
    double R0,double R1,double R2,double R3,double R4,double R5,double R6,double R7,double R8,
    double wave, double qmax2, double tol, const double* pg, int npan,
    double* out, int* counter, int cap){
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= nhkl) return;
  double gx=g[3*i], gy=g[3*i+1], gz=g[3*i+2];
  double qx=gx*R0+gy*R3+gz*R6, qy=gx*R1+gy*R4+gz*R7, qz=gx*R2+gy*R5+gz*R8;
  double qn2=qx*qx+qy*qy+qz*qz;
  double exc=qz+0.5*wave*qn2;
  if (!(qn2<=qmax2 && fabs(exc)<tol)) return;
  // ---- project_q, verbatim in the same order numpy evaluates it -------------------------------
  double sx=wave*qx, sy=wave*qy, sz=wave*qz+1.0;          // s_hat = wave*q + zhat
  double nrm=sqrt(sx*sx+sy*sy+sz*sz);
  sx/=nrm; sy/=nrm; sz/=nrm;
  if (!(sz > 1e-6)) return;                                // fwd
  for (int p=0;p<npan;++p){
    const double* P = pg + p*12;
    double Zp=P[0], res=P[1], a00=P[2], a01=P[3], a10=P[4], a11=P[5], cx=P[6], cy=P[7];
    double t = Zp/sz;
    double Xx = sx*t, Xy = sy*t;
    double rx = res*Xx - cx, ry = res*Xy - cy;
    double lf = a00*rx + a01*ry;                           // lfls = inv(A) @ rhs
    double ls = a10*rx + a11*ry;
    double f = P[8] + lf, s = P[10] + ls;
    if (f>=P[8] && f<=P[9] && s>=P[10] && s<=P[11]){
      int k=atomicAdd(counter,1);
      if (k<cap){ out[6*k]=(double)i; out[6*k+1]=f; out[6*k+2]=s;
                  out[6*k+3]=(double)p; out[6*k+4]=exc; out[6*k+5]=1.0/sqrt(qn2); }
      return;                                              // first panel that catches it
    }
  }
}"""
KERN = cp.RawKernel(_SRC, "gate_project")


def panel_geom(panels, clen_m):
    rows = []
    for p in panels:
        A = np.array([[p["fs"][0], p["ss"][0]], [p["fs"][1], p["ss"][1]]], float)
        Ai = np.linalg.inv(A)
        rows.append([clen_m + p["coffset"], p["res"], Ai[0, 0], Ai[0, 1], Ai[1, 0], Ai[1, 1],
                     p["cx"], p["cy"], p["min_fs"], p["max_fs"], p["min_ss"], p["max_ss"]])
    return cp.asarray(np.asarray(rows, float).ravel())


def rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def main():
    rng = np.random.default_rng(9)
    Rs = [recip_from_M(rot(rng) @ LYSO) for _ in range(60)]
    grid = sd.HKLGrid(LYSO, 2.0, gpu=True, panels=PAN, clen_m=CLEN, wavelength_A=LAM)
    pg = panel_geom(PAN, CLEN); npan = len(PAN)
    nhkl = grid.g.shape[0]; tpb = 256; blocks = (nhkl + tpb - 1)//tpb
    gout = cp.empty((nhkl, 6), cp.float64); gcnt = cp.zeros(1, cp.int32)
    f8 = np.float64

    def fused(R, tol):
        Rr = np.ascontiguousarray(R, np.float64).ravel()
        gcnt[0] = 0
        KERN((blocks,), (tpb,), (grid._ggr, np.int32(nhkl),
             f8(Rr[0]),f8(Rr[1]),f8(Rr[2]),f8(Rr[3]),f8(Rr[4]),f8(Rr[5]),f8(Rr[6]),f8(Rr[7]),f8(Rr[8]),
             f8(LAM), f8(grid.qmax**2), f8(tol), pg, np.int32(npan),
             gout.ravel(), gcnt, np.int32(nhkl)))
        n = int(gcnt[0])
        C = cp.asnumpy(gout[:n])
        C = C[np.argsort(C[:, 0], kind="stable")]
        idx = C[:, 0].astype(np.int64); hkl = grid.g[idx]
        out = np.zeros(n, dtype=[("h",int),("k",int),("l",int),("fs",float),("ss",float),
                                 ("panel",int),("exc",float),("res",float)])
        out["h"],out["k"],out["l"] = hkl[:,0],hkl[:,1],hkl[:,2]
        out["fs"],out["ss"],out["panel"] = C[:,1],C[:,2],C[:,3].astype(int)
        out["exc"],out["res"] = C[:,4],C[:,5]
        return out

    grid.predict(Rs[0], PAN, CLEN, LAM, tol=0.002, is_recip=True); fused(Rs[0], 0.002)

    nbad = nrow = 0; dmax = 0.0
    for R in Rs:
        a = grid.predict(R, PAN, CLEN, LAM, tol=0.002, is_recip=True)
        b = fused(R, 0.002)
        if len(a) != len(b):
            nbad += 1; print(f"  LEN {len(a)} vs {len(b)}"); continue
        for fld in ("h","k","l","panel"):
            if not np.array_equal(a[fld], b[fld]):
                nbad += 1; print(f"  field {fld} differs"); break
        else:
            for fld in ("fs","ss","exc","res"):
                dmax = max(dmax, float(np.max(np.abs(a[fld]-b[fld]))) if len(a) else 0.0)
        nrow += len(a)
    print(f"exactness: {nbad}/{len(Rs)} frames with an integer-field/length mismatch; "
          f"{nrow} reflections; max |float diff| = {dmax:.3e}")

    for label, fn in (("shipped predict", lambda R: grid.predict(R, PAN, CLEN, LAM, tol=0.002, is_recip=True)),
                      ("fused gate+project", lambda R: fused(R, 0.002))):
        t0 = time.perf_counter()
        for R in Rs: o = fn(R)
        print(f"  {label:22s} {1e3*(time.perf_counter()-t0)/len(Rs):7.3f} ms/frame   "
              f"({len(o)} reflections out)")
    print(f"  gate survivors reaching the host: shipped {int(grid._gcnt[0])}  fused {int(gcnt[0])}")


if __name__ == "__main__":
    main()
