import numpy as np,h5py
W='/pscratch/sd/k/kmdalton/cxidb_45/work'; run='run296940-1'; k=1/0.9494
tags=h5py.File(f'{W}/stacked/{run}.h5')['/LCLS/tag'][()]
def rd(arm):
    out={}; on=False; M=None; ev=None
    for l in open(f'{W}/streams/{arm}/{run}.stream'):
        if l.startswith('Event: '):
            e=l.split(': ',1)[1].strip(); ev=e if e.startswith('tag') else 'tag-%d//'%tags[int(e.split('//')[1])]
        elif l[:6] in('astar ','bstar ','cstar '):
            v=[float(x)*0.1 for x in l.split('=')[1].split()[:3]]
            M=[v] if l[0]=='a' else M+[v]
        elif l.startswith('Reflections measured'): on=True; H=[]
        elif l.startswith('End of reflections'):
            on=False; H=np.array(H); A=np.array(M).T; q=H[:,:3]@A.T
            e=np.linalg.norm(q+np.array([0,0,k]),axis=1)-k; out[ev]=np.column_stack([H,e,np.linalg.norm(q,axis=1)])
        elif on:
            w=l.split()
            if w and w[0]!='h': H.append([float(w[0]),float(w[1]),float(w[2]),float(w[3]),float(w[4])])
    return out
ARMS=('box','profile','crystfel'); R={a:rd(a) for a in ARMS}
evs=sorted(set.intersection(*[set(R[a]) for a in ARMS])); print('frames in all arms',len(evs))
key=lambda r:(r[:,0].astype(np.int64)+512)*1048576+(r[:,1].astype(np.int64)+512)*1024+(r[:,2].astype(np.int64)+512)
rows=[]
for ev in evs:
    ks=[key(R[a][ev]) for a in ARMS]; c=np.intersect1d(np.intersect1d(ks[0],ks[1]),ks[2])
    idx=[np.searchsorted(k_[np.argsort(k_)],c) for k_ in ks]; o=[np.argsort(k_) for k_ in ks]
    I=[R[a][ev][o[i][idx[i]]] for i,a in enumerate(ARMS)]
    rows.append(np.column_stack([I[0][:,5],I[0][:,6],I[0][:,3],I[1][:,3],I[2][:,3],I[0][:,4],I[1][:,4],I[2][:,4]]))
x=np.concatenate(rows); d=1/x[:,1]; ae=np.abs(x[:,0]); print('common reflections',len(x),'CrystFEL in ADU: divided by 10 below')
print('mean I (photons), identical reflections, +- standard error')
print(' shell     |e| range          n     box              profile          crystfel')
se=lambda v:v.std()/np.sqrt(len(v))
for a,b in ((3,2),(2,1.7),(1.7,1.5),(1.5,1.3),(1.3,1.2),(1.2,1.06)):
    for lo,hi in ((0,5e-4),(5e-4,1e-3),(1e-3,1.2e-3)):
        s=(d<a)&(d>=b)&(ae>=lo)&(ae<hi)
        if s.sum()>50: print(f' {a:>4}-{b:<5}{lo:.4f}-{hi:.4f}{s.sum():8d}  {x[s,2].mean():6.2f}+-{se(x[s,2]):.2f}   {x[s,3].mean():6.2f}+-{se(x[s,3]):.2f}   {x[s,4].mean()/10:6.2f}+-{se(x[s,4])/10:.2f}')
print('median sigma (photons) by shell, box profile crystfel:')
for a,b in ((3,2),(2,1.5),(1.5,1.2),(1.2,1.06)):
    s=(d<a)&(d>=b); print(f' {a}-{b}: {np.median(x[s,5]):.2f} {np.median(x[s,6]):.2f} {np.median(x[s,7])/10:.2f}')
