import sys, time; sys.path.insert(0,"/sdf/home/s/smarches/git/glint/experiments")
import numpy as np, cupy
from radial import RadialLUT
N=1024; px=1e-4
yy,xx=np.mgrid[-N:N,-N:N].astype(np.float64); r_pix=np.sqrt(xx*xx+yy*yy)
img=np.ascontiguousarray((np.exp(-((r_pix*px*1e3/50-1)**2)/0.02)+0.05*np.random.rand(2*N,2*N)).astype(np.float32))
try:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
except Exception:
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import logging; logging.getLogger("pyFAI").setLevel(logging.ERROR)
ai=AzimuthalIntegrator(dist=1.0,pixel1=px,pixel2=px,wavelength=1e-10); ai.poni1=px*N; ai.poni2=px*N
nbin=1449
res=ai.integrate1d(img,nbin,method=("full","csr","opencl"),unit="r_mm")
eng=ai.engines[res.method].engine
# pyFAI CSR nnz
nnz=None
for a in ("data","_data","coef"):
    o=getattr(eng,a,None)
    if o is not None:
        try: nnz=int(np.prod(o.shape))
        except Exception: pass
print("pyFAI method:",res.method," pyFAI CSR nnz~",nnz)
import pyopencl.array as cla
frame_d=cla.to_device(eng.queue,img)
omega=ai.solidAngleArray(); crc=eng.on_device.get("solidangle")
def pf(): return eng.integrate_ng(frame_d,solidangle=omega,solidangle_checksum=crc)
for _ in range(5): pf()
eng.queue.finish() if hasattr(eng,"queue") else None
t=[]; 
for _ in range(100):
    t0=time.perf_counter(); pf(); eng.queue.finish(); t.append(time.perf_counter()-t0)
print("pyFAI integrate_ng (kernel-only, frame on GPU): %.4f ms"%(np.median(t)*1e3))
# our SpMV, frame on GPU
lut=RadialLUT(cupy.asarray(r_pix*px*1e3),nbin=nbin,qmin=0.,qmax=float((r_pix*px*1e3).max()),split="area")
print("our area-LUT nnz:",lut.M.nnz,"(%.2f/pixel)"%(lut.M.nnz/(2*N)**2))
g=cupy.asarray(img); cupy.cuda.Stream.null.synchronize()
t=[]
for _ in range(100):
    t0=time.perf_counter(); lut.integrate(g); cupy.cuda.Stream.null.synchronize(); t.append(time.perf_counter()-t0)
print("our area M@x   (kernel-only, frame on GPU): %.4f ms"%(np.median(t)*1e3))
