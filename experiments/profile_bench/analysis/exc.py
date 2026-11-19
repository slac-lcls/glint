import numpy as np,re
W='/pscratch/sd/k/kmdalton/cxidb_45/work'; run='run296940-1'
k=1/0.9494
rows=[]; M=None; on=False
for l in open(f'{W}/streams/box/{run}.stream'):
    if l[:6] in('astar ','bstar ','cstar '):
        v=[float(x)*0.1 for x in l.split('=')[1].split()[:3]]
        if l[0]=='a': M=[v]
        else: M.append(v)
    elif l.startswith('Reflections measured'): on=True; A=np.array(M).T; H=[]; next_h=True
    elif l.startswith('End of reflections'):
        on=False
        H=np.array(H); q=H[:,:3]@A.T
        for sgn in (1,):
            e=np.linalg.norm(q+np.array([0,0,sgn*k]),axis=1)-k
        e2=np.linalg.norm(q+np.array([0,0,-k]),axis=1)-k
        rows.append(np.column_stack([e,e2,np.linalg.norm(q,axis=1),H[:,3],H[:,4]]))
    elif on:
        w=l.split()
        if w and w[0]!='h': H.append([float(w[0]),float(w[1]),float(w[2]),float(w[3]),float(w[4])])
x=np.concatenate(rows); print('obs',len(x))
print('|e| with k along +z p50,99,max',np.round(np.percentile(np.abs(x[:,0]),[50,99,100]),5),' -z',np.round(np.percentile(np.abs(x[:,1]),[50,99,100]),5))
e=x[:,0] if np.percentile(np.abs(x[:,0]),99)<np.percentile(np.abs(x[:,1]),99) else x[:,1]
ed=np.array([0,1,2,3,4,5,6,8,10,12,14,16,18,20])*1e-4
sh=[(99,3),(3,2),(2,1.7),(1.7,1.5),(1.5,1.3),(1.3,1.2),(1.2,1.06)]
d=1/x[:,2]; I=x[:,3]; S=x[:,4]
print('mean photons per reflection, by |excitation error| (1e-4 /A, columns) and resolution (rows); n in final row')
print('            '+' '.join(f'{a*1e4:>5.0f}' for a in ed[:-1]))
for a,b in sh:
    s=(d<a)&(d>=b); r=[]
    for lo,hi in zip(ed[:-1],ed[1:]):
        m=s&(np.abs(e)>=lo)&(np.abs(e)<hi); r.append(I[m].mean() if m.sum()>20 else np.nan)
    print(f'{a:>4}-{b:<5}    '+' '.join(f'{v:5.1f}' for v in r))
print('n/1000      '+' '.join(f'{((np.abs(e)>=lo)&(np.abs(e)<hi)).sum()/1000:5.0f}' for lo,hi in zip(ed[:-1],ed[1:])))
print('signed e, all d: mean I by e bin (1e-4)'); be=np.arange(-20,21,4)*1e-4
print(' '.join(f'{a*1e4:+.0f}:{I[(e>=a)&(e<a+4e-4)].mean():.1f}' for a in be[:-1]))
tot=[(t,I[np.abs(e)<t].sum()/I.sum(),(np.abs(e)<t).mean()) for t in (2e-4,3e-4,4e-4,5e-4,6e-4,8e-4,1e-3,1.2e-3,1.5e-3,2e-3)]
print('tol, cumulative share of total I, share of reflections'); [print(f'  {t:.4f} {a:.3f} {b:.3f}') for t,a,b in tot]
