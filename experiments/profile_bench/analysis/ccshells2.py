import sys,numpy as np,gemmi,reciprocalspaceship as rs
sys.path.insert(0,'/pscratch/sd/k/kmdalton/glint/experiments/profile_bench')
from hkl_to_mtz import read_hkl as _read_hkl, reindex as _reindex
import os
def read_hkl(p):
    """The merge in the model's setting: REINDEX (env, default h,k,l) maps GLINT's indices to SG's."""
    x = _read_hkl(p).astype(float)
    x[:, :3] = _reindex(x[:, :3], os.environ.get("REINDEX", "h,k,l"))
    return x
PM=sys.argv[1]; ARMS=('box','profile','crystfel')
cell,sg=gemmi.UnitCell(*map(float,sys.argv[2].split(','))),gemmi.SpaceGroup(os.environ.get('SG', 'P 43 21 2'))
def load(p):
    x=read_hkl(p); d=rs.DataSet({'H':x[:,0].astype(int),'K':x[:,1].astype(int),'L':x[:,2].astype(int),'I':x[:,3],'S':x[:,4],'N':x[:,5]},cell=cell,spacegroup=sg).infer_mtz_dtypes().set_index(['H','K','L'])
    d=d[d.S>0].hkl_to_asu(anomalous=True).remove_absences(); d.merged=True; return d.drop(columns=['M/ISYM'],errors='ignore')
def pairs(arm):
    a,b=load(f'{PM}/merge/{arm}/merged.hkl1'),load(f'{PM}/merge/{arm}/merged.hkl2')
    return a[['I','N']].join(b[['I','N']],rsuffix='2',how='inner').dropna().compute_dHKL()
J={a:pairs(a) for a in ARMS}
# shell edges equal in 1/d^3 (equal reciprocal volume), the same for every arm
def table(dmax,dmin,n):
    e=np.linspace(dmax**-3,dmin**-3,n+1)**(-1/3)
    print(f'--- {dmax}-{dmin} A, {n} shells of equal reciprocal volume; CC1/2 from partialator half-datasets')
    print('   dmax   dmin |'+'|'.join(f' {a:>8s} n  CC1/2 ' for a in ARMS))
    for hi,lo in zip(e[:-1],e[1:]):
        r=[]
        for a in ARMS:
            j=J[a]; x=j[(j.dHKL<hi)&(j.dHKL>=lo)]
            cc=np.corrcoef(x.I.to_numpy(float),x.I2.to_numpy(float))[0,1] if len(x)>2 else np.nan
            r.append(f' {len(x):10d} {cc:6.3f} ')
        print(f'  {hi:5.3f}  {lo:5.3f} |'+'|'.join(r))
        print('JSON',json.dumps([dmax,dmin,n,hi,lo]+[x.split() for x in r]))
import json; R=[]
for t in sys.argv[3:]: table(*map(float,t.split(",")[:2]),int(t.split(",")[2]))
