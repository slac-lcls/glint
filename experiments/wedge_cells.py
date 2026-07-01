"""Task (b): partial-rotation (oscillation-wedge) sweep across the gen_cells cells (not just lyso).
Same cell construction + rate metric as gen_cells.py/glint_cells.py; wedge machinery from bench_fft_seeded.py.
index_blind_fast (blind Fibonacci) vs index_blind_cluster_seeded per cell x {still, wedge20, wedge60}."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
from glint.glint_fast import index_blind_fast, index_blind_cluster_seeded
DMIN = 3.0; QMAX = 1.0/DMIN; REPS = 3; LAM = 1.0
CELLS = [("lyso",(79,79,38,90,90,90)), ("prok",(68.5,68.5,109,90,90,90)),
         ("thaum",(58,58,130,90,90,90)), ("hex",(105,105,75,90,90,120)),
         ("cubic",(78,78,78,90,90,90)), ("ortho",(60,110,135,90,90,90))]
WEDGES = [("still",0.0), ("wedge20",20.0), ("wedge60",60.0)]
def cell_to_B(a,b,c,al,be,ga):
    al,be,ga = np.radians([al,be,ga]); va=[a,0,0]; vb=[b*np.cos(ga),b*np.sin(ga),0]
    cx=c*np.cos(be); cy=c*(np.cos(al)-np.cos(be)*np.cos(ga))/np.sin(ga)
    vc=[cx,cy,np.sqrt(max(c*c-cx*cx-cy*cy,1e-9))]; A=np.array([va,vb,vc]).T
    return np.linalg.inv(A).T
def rot(ax,th):
    ax=ax/np.linalg.norm(ax); c,s=np.cos(th),np.sin(th); x,y,z=ax
    return np.array([[c+x*x*(1-c),x*y*(1-c)-z*s,x*z*(1-c)+y*s],
                     [y*x*(1-c)+z*s,c+y*y*(1-c),y*z*(1-c)-x*s],
                     [z*x*(1-c)-y*s,z*y*(1-c)+x*s,c+z*z*(1-c)]])
def rand_rot(r):
    u1,u2,u3=r.random(3)
    q=np.array([np.sqrt(1-u1)*np.sin(2*np.pi*u2),np.sqrt(1-u1)*np.cos(2*np.pi*u2),
                np.sqrt(u1)*np.sin(2*np.pi*u3),np.sqrt(u1)*np.cos(2*np.pi*u3)])
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def observed(Bor, wedge_deg, osc_axis):
    H=(np.ceil(QMAX/np.linalg.norm(Bor,axis=0))+1).astype(int)
    hk=np.mgrid[-H[0]:H[0]+1,-H[1]:H[1]+1,-H[2]:H[2]+1].reshape(3,-1).T
    hk=hk[np.any(hk!=0,axis=1)]; q0=hk@Bor.T; q0=q0[np.linalg.norm(q0,axis=1)<=QMAX]
    tol=0.004
    if wedge_deg<=0.0:
        return q0[np.abs(q0[:,2]+0.5*LAM*np.einsum("ij,ij->i",q0,q0))<tol]
    phis=np.deg2rad(np.linspace(0,wedge_deg,max(2,int(wedge_deg*2)))); obs=np.zeros(len(q0),bool)
    for ph in phis:
        qp=q0@rot(osc_axis,ph).T; obs|=np.abs(qp[:,2]+0.5*LAM*np.einsum("ij,ij->i",qp,qp))<tol
    return q0[obs]
def ok(M,axes):
    if M is None: return False
    L=np.sort(np.linalg.norm(np.asarray(M,float),axis=0))
    return bool(np.all(np.abs(L-axes)<=0.05*axes))
def timed(fn,g):
    t0=time.perf_counter()
    try: M=fn(g)
    except Exception: torch.cuda.empty_cache(); return None, time.perf_counter()-t0
    return M, time.perf_counter()-t0
_B=cell_to_B(*CELLS[0][1]); _r=np.random.default_rng(1)
_g=observed(rand_rot(_r)@_B,0.0,np.array([0,1,0.])); index_blind_fast(_g); index_blind_cluster_seeded(_g)
rng=np.random.default_rng(0)
print(f"multi-cell oscillation-wedge sweep  dmin {DMIN}  reps {REPS}  STEPS=8")
print(f"{'cell':7}{'regime':9}{'rlps':>7}{'Fib%':>6}{'Fib ms':>9}{'Clus%':>7}{'Clus ms':>9}{'x':>7}")
for name,cp in CELLS:
    B=cell_to_B(*cp); axes=np.sort(np.array(cp[:3],float))
    for label,W in WEDGES:
        ns=[]; okF=okS=0; tF=[]; tS=[]
        for r in range(REPS):
            g=observed(rand_rot(rng)@B, W, rng.normal(size=3)); ns.append(len(g))
            mF,dF=timed(index_blind_fast,g); tF.append(dF); okF+=ok(mF,axes)
            mS,dS=timed(index_blind_cluster_seeded,g); tS.append(dS); okS+=ok(mS,axes)
        mtF,mtS=1e3*np.median(tF),1e3*np.median(tS)
        print(f"{name:7}{label:9}{int(np.median(ns)):>7}{100*okF//REPS:>5}%{mtF:>9.1f}{100*okS//REPS:>6}%{mtS:>9.1f}{mtF/mtS:>6.1f}x",flush=True)
