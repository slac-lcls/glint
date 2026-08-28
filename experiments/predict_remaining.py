"""What is left in predict() after the panel cone and the transfer/API removal? Decides whether the
planned fp32+int16 step earns its risk (it breaks the bit-identity property the other two preserve).

Times the REAL shipped predict, with project_q wrapped in place, so nothing is re-implemented."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT); sys.path.insert(0, WT + "/experiments")
import cupy as cp
from glint.lattice import cell_to_Ar
import glint.stream_driver as sd
from glint.predict import recip_from_M
import glint.predict as gp

LAM, CLEN, RES = 1.322, 0.1, 10000.0
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
PAN = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=RES,
            cx=-287.5, cy=-167.5, coffset=0.0, min_fs=0, max_fs=575, min_ss=0, max_ss=335)]

T = {"proj": 0.0, "n": 0}
_op = sd.project_q
def timed_proj(*a, **k):
    t0 = time.perf_counter(); r = _op(*a, **k); T["proj"] += time.perf_counter() - t0; T["n"] += 1
    return r
sd.project_q = timed_proj

def rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

rng = np.random.default_rng(7)
Rs = [recip_from_M(rot(rng) @ LYSO) for _ in range(60)]
g = sd.HKLGrid(LYSO, 2.0, gpu=True, panels=PAN, clen_m=CLEN, wavelength_A=LAM)
g.predict(Rs[0], PAN, CLEN, LAM, tol=0.002, is_recip=True)
T["proj"] = 0.0; T["n"] = 0

# kernel alone, via CUDA events on a representative launch
ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
nhkl = g.g.shape[0]; tpb = 256; blocks = (nhkl + tpb - 1)//tpb
Rr = np.ascontiguousarray(Rs[0], np.float64).ravel(); f8 = np.float64
cone = g._cone; ax, ay, az, kmin = (*cone[0], cone[1])
ktimes = []
for _ in range(50):
    ev0.record()
    sd._GATE_KERNEL((blocks,), (tpb,), (g._ggr, np.int32(nhkl),
        f8(Rr[0]),f8(Rr[1]),f8(Rr[2]),f8(Rr[3]),f8(Rr[4]),f8(Rr[5]),f8(Rr[6]),f8(Rr[7]),f8(Rr[8]),
        f8(LAM), f8(g.qmax**2), f8(0.002), g._gout.ravel(), g._gcnt, np.int32(nhkl),
        f8(ax), f8(ay), f8(az), f8(kmin)))
    ev1.record(); ev1.synchronize(); ktimes.append(cp.cuda.get_elapsed_time(ev0, ev1))

t0 = time.perf_counter()
for R in Rs:
    out = g.predict(R, PAN, CLEN, LAM, tol=0.002, is_recip=True)
tot = (time.perf_counter() - t0) / len(Rs)
nsurv = int(g._gcnt[0])

print(f"6-ASIC DRP panel, tol=0.002, {len(Rs)} orientations, {nsurv} survivors, {len(out)} on-panel")
print(f"  predict TOTAL            {1e3*tot:7.3f} ms")
print(f"  project_q                {1e3*T['proj']/len(Rs):7.3f} ms   ({100*T['proj']/len(Rs)/tot:4.1f}%)")
print(f"  gate kernel (device)     {np.median(ktimes):7.3f} ms   ({100*np.median(ktimes)/1e3/tot:4.1f}%)")
print(f"  everything else          {1e3*tot - 1e3*T['proj']/len(Rs) - np.median(ktimes):7.3f} ms")
print(f"\n  D2H payload now {nsurv*6*8/1024:.1f} kB; fp32 would make it {nsurv*6*4/1024:.1f} kB,")
print(f"  index-only would make it {nsurv*4/1024:.1f} kB. Grid is {g.g.nbytes*8/3/1024/1024:.2f} MB as")
print(f"  float64, {g.g.nbytes*2/3/1024/1024:.2f} MB as int16.")
