"""Is the '25.6 us launch floor' real hardware latency, or cupy Python-side launch overhead?

If it is the latter, the full-grid gate's compute is NOT free and the claim's central
'doing 10.2x less arithmetic buys nothing' argument does not follow.
All comparisons inside ONE allocation, cp.cuda.Event, 10 warmup + 50 reps, min+median.
"""
import numpy as np, cupy as cp, time

SRC = r"""
extern "C" __global__ void predict_gate(const double* g, int nhkl, const double* R, double wave,
                                         double qmax2, double tol, double* out, int* counter, int cap){
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= nhkl) return;
  double gx=g[3*i], gy=g[3*i+1], gz=g[3*i+2];
  double qx=gx*R[0]+gy*R[3]+gz*R[6];
  double qy=gx*R[1]+gy*R[4]+gz*R[7];
  double qz=gx*R[2]+gy*R[5]+gz*R[8];
  double qn2=qx*qx+qy*qy+qz*qz;
  double exc=qz+0.5*wave*qn2;
  if (qn2<=qmax2 && fabs(exc)<tol){
    int p=atomicAdd(counter,1);
    if (p<cap){ out[6*p]=(double)i; out[6*p+1]=qx; out[6*p+2]=qy; out[6*p+3]=qz; out[6*p+4]=qn2; out[6*p+5]=exc; }
  }
}
extern "C" __global__ void noop(int n){ }
"""
mod_gate = cp.RawKernel(SRC, "predict_gate")
mod_noop = cp.RawKernel(SRC, "noop")

print(cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), "| cupy", cp.__version__)

def cell_to_M(a,b,c,al,be,ga):
    al,be,ga = np.radians([al,be,ga])
    va=np.array([a,0,0.]); vb=np.array([b*np.cos(ga),b*np.sin(ga),0.])
    cx=c*np.cos(be); cy=c*(np.cos(al)-np.cos(be)*np.cos(ga))/np.sin(ga)
    return np.stack([va,vb,np.array([cx,cy,np.sqrt(c*c-cx*cx-cy*cy)])],axis=1)

def hkl_grid(R,qmax):
    n=np.linalg.norm(R,axis=1); H,K,L=(int(np.ceil(qmax/x))+1 for x in n)
    g=np.stack(np.meshgrid(np.arange(-H,H+1),np.arange(-K,K+1),np.arange(-L,L+1),indexing="ij"),-1).reshape(-1,3)
    g=g[np.any(g!=0,axis=1)]
    return g[np.einsum("ij,ij->i",g@R,g@R)<=qmax*qmax]

M0=cell_to_M(79.,79.,38.,90,90,90); R=np.linalg.inv(M0)
wave, tol, dmin = 1.32, 0.002, 2.0; qmax=1./dmin
g = hkl_grid(R, qmax*1.02).astype(np.float64)
nhkl = g.shape[0]
print(f"nhkl = {nhkl}")

gg  = cp.ascontiguousarray(cp.asarray(g).ravel())
Rf  = cp.ascontiguousarray(cp.asarray(R).ravel())
out = cp.empty(nhkl*6, cp.float64); cnt = cp.zeros(1, cp.int32)

def timeit(fn, reps=50, warm=10):
    for _ in range(warm): fn()
    cp.cuda.Stream.null.synchronize()
    ts=[]
    for _ in range(reps):
        e0=cp.cuda.Event(); e1=cp.cuda.Event()
        e0.record(); fn(); e1.record(); e1.synchronize()
        ts.append(cp.cuda.get_elapsed_time(e0,e1)*1000.0)   # us
    return min(ts), float(np.median(ts))

def gate_n(n):
    tpb=256; bl=(n+tpb-1)//tpb
    def f():
        mod_gate((bl,),(tpb,),(gg,np.int32(n),Rf,np.float64(wave),
                 np.float64(qmax*qmax),np.float64(tol),out,cnt,np.int32(nhkl)))
    return f

print("\n--- inside the event window: kernel launch only (claim's methodology) ---")
for n in (1, 1024, 16384, 65536, nhkl):
    mn,md = timeit(gate_n(n))
    print(f"  gate over nhkl={n:>7d}   {mn:7.2f} / {md:7.2f} us")
mn,md = timeit(lambda: mod_noop((1,),(1,),(np.int32(0),)))
print(f"  EMPTY kernel, 1 thread     {mn:7.2f} / {md:7.2f} us   <- cupy Python launch overhead")

print("\n--- amortised: N launches inside ONE event window (isolates DEVICE time) ---")
def burst(n, reps=100):
    f = gate_n(n)
    def g_():
        for _ in range(reps): f()
    return g_
for n in (1, nhkl):
    mn,md = timeit(burst(n), reps=20, warm=5)
    print(f"  gate nhkl={n:>7d}  x100 launches: total {mn:8.1f} us -> per launch {mn/100:6.2f} us")
mn,md = timeit(lambda: [mod_noop((1,),(1,),(np.int32(0),)) for _ in range(100)] and None, reps=20, warm=5)
print(f"  EMPTY kernel      x100 launches: total {mn:8.1f} us -> per launch {mn/100:6.2f} us")

print("\n--- host wall clock of the pure Python launch call (no device work waited on) ---")
f = gate_n(nhkl)
for _ in range(20): f()
cp.cuda.Stream.null.synchronize()
t0=time.perf_counter()
for _ in range(1000): f()
t1=time.perf_counter(); cp.cuda.Stream.null.synchronize()
print(f"  host time per cupy RawKernel launch call: {(t1-t0)/1000*1e6:.2f} us")
