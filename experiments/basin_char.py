"""GLINT M3 basin characterization (synthetic). Capture radius / recovery / fixed-point drift vs
peak-count(sparsity), position-noise, mosaic, partiality; optimizers GD/CG/BB; restart-ensemble scaling.
Reuses the still-frame Ewald cloud (gen_clouds style); target vectors = rows of inv(A) (exact lattice vecs).
Run on ampere. Synthetic only."""
import numpy as np, torch, sys
sys.path.insert(0,'/sdf/home/s/smarches/git/glint')
from glint.glint_index import refine_vec, refine_vec_cg, refine_vec_bb, objective, invq_weight
dev='cuda'; torch.set_grad_enabled(False)
QMAX=1/3.; LAM=1.; TOL=0.18
CELLS={'lyso_79_79_38':(79.,79.,38.), 'cubic_100':(100.,100.,100.), 'ortho_50_65_80':(50.,65.,80.)}
def rot(ax,th):
    ax=ax/np.linalg.norm(ax); c,s=np.cos(th),np.sin(th); x,y,z=ax
    return np.array([[c+x*x*(1-c),x*y*(1-c)-z*s,x*z*(1-c)+y*s],[y*x*(1-c)+z*s,c+y*y*(1-c),y*z*(1-c)-x*s],[z*x*(1-c)-y*s,z*y*(1-c)+x*s,c+z*z*(1-c)]])
def still_Q(A,HKL):
    q0=HKL@A.T; m=np.linalg.norm(q0,axis=1)<=QMAX; q0=q0[m]
    obs=np.abs(q0[:,2]+0.5*LAM*np.einsum('ij,ij->i',q0,q0))<0.004; return q0[obs]
def perturb(vstar,theta,n,rng): return np.stack([rot(rng.normal(size=3),np.deg2rad(theta))@vstar for _ in range(n)])
def frames_for(cell,nR,rng):
    B=np.diag(1/np.array(cell)); H=(np.ceil(QMAX*np.array(cell))+1).astype(int)
    HKL=np.mgrid[-H[0]:H[0]+1,-H[1]:H[1]+1,-H[2]:H[2]+1].reshape(3,-1).T; HKL=HKL[np.any(HKL!=0,1)].astype(float)
    out=[]
    for _ in range(nR):
        R=rot(rng.normal(size=3),rng.uniform(0,np.pi)); A=R@B; Q=still_Q(A,HKL)
        if len(Q)<40: continue
        inv=np.linalg.inv(A); out.append((Q, inv[np.argmin(cell)], inv[np.argmax(cell)]))  # short, long real vec
    return out
def rec(fn, Qn, vstar, theta, M, steps, rng, drop=0.0, noise=0.0, mos=0.0):
    Q=Qn.copy()
    if drop>0: Q=Q[rng.random(len(Q))>=drop]
    if noise>0: Q=Q+rng.normal(0,noise*QMAX,Q.shape)
    if mos>0:
        Q=np.stack([rot(rng.normal(size=3),np.deg2rad(mos)*rng.normal())@q for q in Q])
    if len(Q)<8: return 0.0
    Qt=torch.tensor(Q,dtype=torch.float64,device=dev); w=invq_weight(Qt); vs=torch.tensor(vstar,dtype=torch.float64,device=dev)
    T0=torch.tensor(perturb(vstar,theta,M,rng),dtype=torch.float64,device=dev)
    Tout=fn(T0,Qt,w,QMAX,steps=steps)
    d=torch.minimum((Tout-vs).norm(1 if False else -1) if False else (Tout-vs).norm(dim=1),(Tout+vs).norm(dim=1))/vs.norm()
    return float((d<0.05).float().mean())
rng=np.random.default_rng(0); M=120; NR=12
FR={c:frames_for(cell,NR,rng) for c,cell in CELLS.items()}
def sweep(cell, tgt, fn, steps, **kw):
    return np.mean([rec(fn, Q, (vs if tgt=='short' else vl), M=M, steps=steps, rng=rng, **kw) for (Q,vs,vl) in FR[cell]])

print("=== 1) CAPTURE RADIUS: recovery vs theta (GD steps=40, short vector) ===")
ths=[0,2,4,6,8,10,14,20]
for c in CELLS:
    r=[sweep(c,'short',refine_vec,40,theta=t) for t in ths]
    print("  %-16s "%c+" ".join("%d:%.0f%%"%(t,100*x) for t,x in zip(ths,r)))
print("=== 2) SPARSITY/PARTIALITY: recovery vs peak-drop (lyso short, theta=4, GD40) ===")
for dr in [0,0.2,0.4,0.6,0.8]:
    print("  drop=%.0f%%: %.0f%%"%(100*dr,100*sweep('lyso_79_79_38','short',refine_vec,40,theta=4,drop=dr)))
print("=== 3) POSITION NOISE: recovery vs noise/qmax (lyso short, theta=4, GD40) ===")
for nz in [0,0.002,0.005,0.01,0.02]:
    print("  noise=%.3f: %.0f%%"%(nz,100*sweep('lyso_79_79_38','short',refine_vec,40,theta=4,noise=nz)))
print("=== 4) MOSAIC: recovery vs mosaic-deg (lyso short, theta=4, GD40) ===")
for ms in [0,0.1,0.2,0.5,1.0]:
    print("  mosaic=%.1fdeg: %.0f%%"%(ms,100*sweep('lyso_79_79_38','short',refine_vec,40,theta=4,mos=ms)))
print("=== 5) OPTIMIZERS at basin edge (lyso short, theta=8) ===")
for name,fn in [('GD',refine_vec),('CG',refine_vec_cg),('BB',refine_vec_bb)]:
    print("  %-4s steps=40: %.0f%%"%(name,100*sweep('lyso_79_79_38','short',fn,40,theta=8)))
print("=== 6) RESTART-ENSEMBLE scaling (GD steps=40, lyso short, theta=8, keep-best-objective) ===")
def rec_restart(Qn,vstar,theta,K,jit,rng):
    vs=torch.tensor(vstar,dtype=torch.float64,device=dev); Qt=torch.tensor(Qn,dtype=torch.float64,device=dev); w=invq_weight(Qt)
    base=perturb(vstar,theta,M,rng); bestf=torch.full((M,),-1e18,dtype=torch.float64,device=dev); bestT=torch.zeros(M,3,dtype=torch.float64,device=dev)
    for k in range(K):
        arr=base if k==0 else np.stack([rot(rng.normal(size=3),np.deg2rad(jit))@base[i] for i in range(M)])
        Tout=refine_vec(torch.tensor(arr,dtype=torch.float64,device=dev),Qt,w,QMAX,steps=40)
        f=objective(Tout,Qt,w,TOL,sharp=True)[0]; pick=f>bestf; bestT[pick]=Tout[pick]; bestf[pick]=f[pick]
    d=torch.minimum((bestT-vs).norm(dim=1),(bestT+vs).norm(dim=1))/vs.norm(); return float((d<0.05).float().mean())
for K in [1,3,5,9,15,25]:
    r=np.mean([rec_restart(Q,vs,8,K,4,rng) for (Q,vs,vl) in FR['lyso_79_79_38']])
    print("  K=%2d: %.0f%%"%(K,100*r))
print("DONE")
