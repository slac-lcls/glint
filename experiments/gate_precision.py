"""fp32 / int16 equivalence test for the stream_driver predict_gate kernel.

Builds 4 scratch variants of _GATE_SRC (A=fp64 reference, B=all-fp32, C=int16 hkl + fp32,
D=fp32 matvec but fp64 exc combination) and asks whether the GATE DECISION ever differs,
over REAL orientations from the cxidb lysozyme frames.

Nothing in the repo is touched.
"""
import os, sys, json, math
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import cupy as cp
from glint_fast import load, cell_to_Ar, LYSO, gpass
from glint.predict import recip_from_M, _hkl_grid
import glint.replica_gpu_batch as rgb

OUT = {}

# ------------------------------------------------------------------ kernel variants ------------
TMPL = r"""
extern "C" __global__ void gate(const %(GT)s* g, int nhkl, const %(RT)s* R, %(RT)s wave,
                                %(RT)s qmax2, %(RT)s tol,
                                %(OT)s* out, int* iout, int* counter, int cap,
                                unsigned char* qmask){
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= nhkl) return;
  %(AT)s gx=(%(AT)s)g[3*i], gy=(%(AT)s)g[3*i+1], gz=(%(AT)s)g[3*i+2];
  %(AT)s qx=gx*R[0]+gy*R[3]+gz*R[6];
  %(AT)s qy=gx*R[1]+gy*R[4]+gz*R[7];
  %(AT)s qz=gx*R[2]+gy*R[5]+gz*R[8];
  %(CT)s Qx=(%(CT)s)qx, Qy=(%(CT)s)qy, Qz=(%(CT)s)qz;
  %(CT)s qn2=Qx*Qx+Qy*Qy+Qz*Qz;
  %(CT)s exc=Qz+(%(CT)s)0.5*(%(CT)s)wave*qn2;
  qmask[i] = (qn2 <= (%(CT)s)qmax2) ? 1 : 0;
  if (qn2 <= (%(CT)s)qmax2 && fabs(exc) < (%(CT)s)tol){
    int p = atomicAdd(counter,1);
    if (p < cap){ iout[p]=i;
      out[5*p]=(%(OT)s)qx; out[5*p+1]=(%(OT)s)qy; out[5*p+2]=(%(OT)s)qz;
      out[5*p+3]=(%(OT)s)qn2; out[5*p+4]=(%(OT)s)exc; }
  }
}"""

VARIANTS = {
    "A_fp64":      dict(GT="double", RT="double", AT="double", CT="double", OT="double",
                        gdt=np.float64, rdt=np.float64, odt=np.float64),
    "B_fp32":      dict(GT="float",  RT="float",  AT="float",  CT="float",  OT="float",
                        gdt=np.float32, rdt=np.float32, odt=np.float32),
    "C_i16_fp32":  dict(GT="short",  RT="float",  AT="float",  CT="float",  OT="float",
                        gdt=np.int16,   rdt=np.float32, odt=np.float32),
    "D_fp32mv_64": dict(GT="float",  RT="float",  AT="float",  CT="double", OT="double",
                        gdt=np.float32, rdt=np.float32, odt=np.float64),
}
KERN = {k: cp.RawKernel(TMPL % v, "gate") for k, v in VARIANTS.items()}


class Runner:
    def __init__(self, g, qmax, cap=400000):
        self.g = np.ascontiguousarray(g.astype(np.int64))
        self.n = g.shape[0]
        self.qmax = float(qmax)
        self.cap = cap
        self.gbuf = {}
        for k, v in VARIANTS.items():
            self.gbuf[k] = cp.ascontiguousarray(cp.asarray(self.g.astype(v["gdt"])).ravel())
        self.out = {k: cp.empty(cap * 5, v["odt"]) for k, v in VARIANTS.items()}
        self.iout = cp.empty(cap, cp.int32)
        self.cnt = cp.zeros(1, cp.int32)
        self.qmask = cp.empty(self.n, cp.uint8)

    def run(self, name, R, wave, tol):
        v = VARIANTS[name]
        Rb = cp.ascontiguousarray(cp.asarray(np.ascontiguousarray(R), dtype=v["rdt"]).ravel())
        self.cnt[0] = 0
        tpb = 256; blocks = (self.n + tpb - 1) // tpb
        KERN[name]((blocks,), (tpb,),
                   (self.gbuf[name], np.int32(self.n), Rb, v["rdt"](wave),
                    v["rdt"](self.qmax * self.qmax), v["rdt"](tol),
                    self.out[name], self.iout, self.cnt, np.int32(self.cap), self.qmask))
        nn = int(self.cnt[0])
        assert nn <= self.cap, f"cap overflow {nn}>{self.cap}"
        idx = cp.asnumpy(self.iout[:nn]).astype(np.int64)
        vals = cp.asnumpy(self.out[name][:nn * 5]).reshape(nn, 5).astype(np.float64)
        qm = cp.asnumpy(self.qmask)
        o = np.argsort(idx, kind="stable")
        return idx[o], vals[o], qm

    def ref(self, idx, R, wave):
        """exact fp64 host values for arbitrary grid indices"""
        q = self.g[idx].astype(np.float64) @ np.asarray(R, np.float64)
        qn2 = np.einsum("ij,ij->i", q, q)
        exc = q[:, 2] + 0.5 * wave * qn2
        return q, qn2, exc


def compare(runner, Rs, wave, tol, tag, variants=("B_fp32", "C_i16_fp32", "D_fp32mv_64")):
    acc = {v: dict(nA=0, nX=0, only_A=[], only_X=[], dexc=[], dqn2=[], dq=[],
                   qm_diff=0, qm_dq=[], nqm=0) for v in variants}
    for R in Rs:
        iA, vA, qmA = runner.run("A_fp64", R, wave, tol)
        for vn in variants:
            iX, vX, qmX = runner.run(vn, R, wave, tol)
            a = acc[vn]
            a["nA"] += len(iA); a["nX"] += len(iX)
            onlyA = np.setdiff1d(iA, iX, assume_unique=True)
            onlyX = np.setdiff1d(iX, iA, assume_unique=True)
            for arr, key in ((onlyA, "only_A"), (onlyX, "only_X")):
                if len(arr):
                    q, qn2, exc = runner.ref(arr, R, wave)
                    for j in range(len(arr)):
                        a[key].append(dict(idx=int(arr[j]), hkl=runner.g[arr[j]].tolist(),
                                           exc=float(exc[j]), aexc=float(abs(exc[j])),
                                           d_tol=float(abs(exc[j]) - tol),
                                           qn2=float(qn2[j]),
                                           d_qmax2=float(qn2[j] - runner.qmax ** 2)))
            common, ia, ix = np.intersect1d(iA, iX, assume_unique=True, return_indices=True)
            if len(common):
                a["dexc"].append(np.abs(vA[ia, 4] - vX[ix, 4]))
                a["dqn2"].append(np.abs(vA[ia, 3] - vX[ix, 3]))
                a["dq"].append(np.linalg.norm(vA[ia, :3] - vX[ix, :3], axis=1))
            # qmax-only cut (tol-independent)
            a["nqm"] += int(qmA.sum())
            dm = np.nonzero(qmA != qmX)[0]
            a["qm_diff"] += len(dm)
            if len(dm):
                _, qn2, _ = runner.ref(dm, R, wave)
                a["qm_dq"].extend((qn2 - runner.qmax ** 2).tolist())
    res = {}
    for vn, a in acc.items():
        def mr(lst):
            if not lst:
                return (0.0, 0.0)
            x = np.concatenate(lst)
            return (float(x.max()), float(np.sqrt((x ** 2).mean())))
        r = dict(tag=tag, tol=tol, variant=vn, n_orient=len(Rs),
                 n_surv_A=a["nA"], n_surv_X=a["nX"],
                 n_only_A=len(a["only_A"]), n_only_X=len(a["only_X"]),
                 n_disagree=len(a["only_A"]) + len(a["only_X"]),
                 frac_disagree=(len(a["only_A"]) + len(a["only_X"])) / max(a["nA"], 1),
                 qmax_cut_pass_A=a["nqm"], qmax_cut_disagree=a["qm_diff"],
                 qmax_cut_dq_max=(float(np.max(np.abs(a["qm_dq"]))) if a["qm_dq"] else None))
        r["exc_max"], r["exc_rms"] = mr(a["dexc"])
        r["qn2_max"], r["qn2_rms"] = mr(a["dqn2"])
        r["q_max"], r["q_rms"] = mr(a["dq"])
        flips = a["only_A"] + a["only_X"]
        r["flips"] = sorted(flips, key=lambda d: abs(d["d_tol"]))[:20]
        r["flip_dtol_max"] = float(max((abs(f["d_tol"]) for f in flips), default=0.0))
        res[vn] = r
        print(f"  [{tag} tol={tol:g}] {vn:<13} survA={a['nA']:>7d} disagree={r['n_disagree']:>4d}"
              f" (A-only {r['n_only_A']}, X-only {r['n_only_X']})  frac={r['frac_disagree']:.2e}"
              f"  |exc|-tol max={r['flip_dtol_max']:.2e}"
              f"  dexc max={r['exc_max']:.2e} rms={r['exc_rms']:.2e}"
              f"  dq max={r['q_max']:.2e}  qmaxcut_diff={a['qm_diff']}", flush=True)
    return res


# ------------------------------------------------------------------ real orientations ----------
frames = [q for q in load(os.path.join(HERE, "frames_cxidb_clean.txt")) if len(q) >= 6]
print(f"cxidb frames: {len(frames)}", flush=True)
Ms = [M for i in range(0, len(frames), 32)
      for M in rgb.index_known_gpu_cell_batch(frames[i:i + 32], LYSO)]
good = [(np.asarray(M, float), q) for M, q in zip(Ms, frames)
        if M is not None and abs(np.linalg.det(np.asarray(M, float))) > 1.0]
print(f"indexed orientations: {len(good)}", flush=True)

# wavelength implied by the data: for an observed reflection qz + 0.5*w*|q|^2 ~ 0
west = []
for M, q in good:
    q = np.asarray(q, float)
    n2 = np.einsum("ij,ij->i", q, q)
    west.extend((-2.0 * q[:, 2] / np.maximum(n2, 1e-12)).tolist())
WAVE = float(np.median(west))
print(f"implied wavelength from cxidb q-vectors: {WAVE:.5f} A  "
      f"(iqr {np.percentile(west,25):.5f}-{np.percentile(west,75):.5f})", flush=True)

Rs_lyso = [recip_from_M(M) for M, _ in good]
# pure rotation part of each real orientation, to transplant onto other cells
Us = []
for M, _ in good:
    U = M @ np.linalg.inv(LYSO)
    Us.append(U)
orth = [float(np.abs(U @ U.T - np.eye(3)).max()) for U in Us]
print(f"orientation rotation-purity max|UU^T-I| = {max(orth):.2e} (median {np.median(orth):.2e})",
      flush=True)

RESULTS = []

# ------------------------------------------------------------------ case 1: lysozyme dmin 2.0 --
def case(tag, Mc, dmin, Rs, wave, tols=(0.001, 0.002, 0.006), margin=1.02, cap=400000):
    R0 = recip_from_M(np.asarray(Mc, float))
    qmax = 1.0 / dmin
    g, _ = _hkl_grid(R0, qmax * margin)
    print(f"\n=== {tag}: dmin={dmin} nhkl={g.shape[0]} |h|max={np.abs(g).max()} "
          f"nOrient={len(Rs)} wave={wave:.4f} ===", flush=True)
    rn = Runner(g, qmax, cap=cap)
    for tol in tols:
        RESULTS.append(compare(rn, Rs, wave, tol, tag))
    del rn
    cp.get_default_memory_pool().free_all_blocks()


case("lyso_d2.0", LYSO, 2.0, Rs_lyso, WAVE)
case("lyso_d2.0_w1.0", LYSO, 2.0, Rs_lyso, 1.0, tols=(0.002,))
case("lyso_d1.2", LYSO, 1.2, Rs_lyso, WAVE)

# ------------------------------------------------------------------ bigger cells ---------------
BIG = cell_to_Ar(300.0, 300.0, 300.0, 90, 90, 90)
Rs_big = [recip_from_M(U @ BIG) for U in Us]
case("cubic300_d3.0", BIG, 3.0, Rs_big, WAVE, cap=800000)
case("cubic300_d2.0", BIG, 2.0, Rs_big[:40], WAVE, tols=(0.001, 0.002), cap=1500000)

with open("/tmp/gate_precision_results.json", "w") as f:
    json.dump(RESULTS, f, indent=1)
print("\nwrote /tmp/gate_precision_results.json", flush=True)

# ------------------------------------------------------------------ int16 range statement ------
for tag, Mc, dmin in (("lyso_d2.0", LYSO, 2.0), ("lyso_d1.2", LYSO, 1.2),
                      ("cubic300_d2.0", BIG, 2.0), ("cubic1000_d1.0", cell_to_Ar(1000., 1000., 1000., 90, 90, 90), 1.0)):
    R0 = recip_from_M(np.asarray(Mc, float))
    norms = np.linalg.norm(R0, axis=1)
    hmax = [int(np.ceil((1.02 / dmin) / n)) + 1 for n in norms]
    print(f"int16 headroom {tag}: |h|max per axis {hmax} (int16 limit 32767)", flush=True)
