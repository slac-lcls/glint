"""End-to-end POWDER pipeline: real cf.geom -> synthetic 2-D powder image -> radial.py ring average
(the drp-benchmarks primitive) -> 1-D ring detection -> blind cell indexing (powder_index). Closes the
loop the powder_index docstring describes (radial-average front end)."""
import os
import sys
from pathlib import Path

import cupy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
try:
    radial_dir = os.environ["DRP_RADIAL_DIR"]
except KeyError:
    raise SystemExit("set DRP_RADIAL_DIR to the radial_integration checkout")
sys.path.insert(0, radial_dir)
from radial import RadialIntegrator
import powder_index as P
from scipy.signal import find_peaks
from scipy.ndimage import minimum_filter1d
from glint.lute_bridge import lambda_from_eV
G={}
if len(sys.argv) != 2:
    raise SystemExit("usage: powder_pipeline.py GEOM")
with open(sys.argv[1], encoding="utf-8") as geom:
    for line in geom:
        line=line.split(";")[0].strip()
        if "=" in line: k,v=line.split("=",1); G[k.strip()]=v.strip()
gf=lambda k:float(G[k]); H=W=3000
res=gf("res");cx=gf("p0/corner_x");cy=gf("p0/corner_y");clen=gf("clen");lam=lambda_from_eV(gf("photon_energy"))
ss,fs=np.mgrid[0:H,0:W].astype(np.float64); x=(cx+fs)/res;y=(cy+ss)/res;z=np.full_like(x,clen);rn=np.sqrt(x*x+y*y+z*z)
qinvd=(np.sqrt((x/rn)**2+(y/rn)**2+(z/rn-1)**2)/lam)          # |q| = 1/d  (per pixel)
q2pi=(2*np.pi*qinvd).astype(np.float32)                       # crystallographers q = 2pi/d, powder_index units
qmax=float(q2pi.max()); ig=RadialIntegrator(cupy.asarray(q2pi),nbin=3000,qmin=0.0,qmax=qmax,split="linear")
rng=np.random.default_rng(1)

def make_powder(system, cell):
    q_true=P.rings(system,cell); q_true=q_true[q_true<qmax*0.98]
    img=(30.0+200.0*np.exp(-q2pi/2.0)).astype(np.float32)     # smooth background
    for qr in q_true:                                        # paint each Debye-Scherrer ring
        img=img+(600.0*np.exp(-((q2pi-qr)**2)/(2*0.010**2))).astype(np.float32)
    img=rng.poisson(img).astype(np.float32)
    return img, q_true

def detect_rings(img):
    qc,I=ig.integrate(cupy.asarray(img)); qc=cupy.asnumpy(qc); I=cupy.asnumpy(I)
    good=np.isfinite(I); qc,I=qc[good],I[good]
    Isub=I-minimum_filter1d(I,81)                            # remove the smooth background
    pk,_=find_peaks(Isub,prominence=max(Isub.max()*0.03,1.0),distance=6)
    return qc[pk]

def run(name, system, cell):
    img,q_true=make_powder(system,cell); qd=detect_rings(img)
    if system=="cubic":
        a,frac=P.index_cubic(qd); print("%-16s true a=%.3f | %2d rings detected -> a=%.3f (err %.2f%%, idx %.0f%%)"%(name,cell[0],len(qd),a,100*abs(a-cell[0])/cell[0],100*frac))
    elif system in ("tetragonal","hexagonal"):
        a,c,fom=P.index_2p(system,qd); print("%-16s true a=%.3f c=%.3f | %2d rings -> a=%.3f c=%.3f (err %.1f/%.1f%%, fom %.2f)"%(name,cell[0],cell[-1],len(qd),a,c,100*abs(a-cell[0])/cell[0],100*abs(c-cell[-1])/cell[-1],fom))
    else:
        axes,fom=P.index_ortho(qd); tr=np.sort(cell)
        print("%-16s true a,b,c=%s | %2d rings -> axes=%s (fom %.2f)"%(name,np.round(tr,3),len(qd),np.round(np.sort(axes),3),fom))

print("POWDER PIPELINE: cf.geom (%dx%d, d_min %.2f A) -> radial.py ring-avg -> detect -> blind index\n"%(H,W,2*np.pi/qmax))
run("Po (cubic)","cubic",(3.350,))
run("NaCl (cubic)","cubic",(5.640,))
run("rutile (tet)","tetragonal",(4.593,2.959))
run("Mg (hex)","hexagonal",(3.209,5.211))
