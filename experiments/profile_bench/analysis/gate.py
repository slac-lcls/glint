import bz2,json,re,numpy as np,h5py
D='/pscratch/sd/k/kmdalton/cxidb_45'; W=D+'/work'; run='run296940-1'
pk=np.load(f'{W}/peaks/{run}.npz'); ims=json.loads(str(pk['images'])); q_by={im['event']:pk[f'q_{i}'] for i,im in enumerate(ims)}
tags=h5py.File(f'{W}/stacked/{run}.h5')['/LCLS/tag'][()]
def stream_chunks(lines):
    cur=None
    for l in lines:
        if l.startswith('----- Begin chunk'): cur=[]
        elif cur is not None:
            if l.startswith('----- End chunk'): yield cur; cur=None
            else: cur.append(l)
def parse(ch):
    d={'n':0}
    for l in ch:
        if l.startswith('Image filename'): d['fn']=l.split(':',1)[1].strip()
        elif l.startswith('Event'): d['ev']=l.split(':',1)[1].strip()
        elif l[:6] in('astar ','bstar ','cstar '): d[l[0]]=[float(x) for x in l.split('=')[1].split()[:3]]
        elif l.startswith('profile_radius'): d['R']=float(l.split('=')[1].split()[0])
        elif l.startswith('Cell parameters'): d['cell']=[float(x) for x in re.findall(r'[-\d.]+',l)[:6]]
        elif l.startswith('num_reflections'): d['n']=int(l.split('=')[1])
    return d
G={}
for c in stream_chunks(open(f'{W}/streams/box/{run}.stream')):
    d=parse(c)
    if 'a' in d: G['tag-%d//'%tags[int(d['ev'].split('//')[1])]]=d
A={}; seen=0
with bz2.open(D+'/deposit/ProK-native.stream.bz2','rt',errors='replace') as f:
    for c in stream_chunks(f):
        seen+=1
        d=parse(c)
        if 'a' in d and 'run296940-1.h5' in d.get('fn',''): A[d['ev']]=d
        elif A and 'run296940-1.h5' not in d.get('fn',''): break
        if seen>60000: break
print('GLINT lattices',len(G),'authors lattices in this file',len(A),'chunks read',seen)
both=sorted(set(G)&set(A)); print('both',len(both))
def stats(S,name):
    C=np.array([s['cell'] for s in S]); print(name,'cell mean',np.round(C.mean(0),3),'sd',np.round(C.std(0),3),'refl/crystal median',np.median([s['n'] for s in S]),'profile radius nm-1 p10,50,90',np.round(np.percentile([s['R'] for s in S],[10,50,90]),5))
stats([G[k] for k in both],'GLINT  '); stats([A[k] for k in both],'authors')
def dist(M,q):
    h=np.linalg.solve(M,q.T).T; r=h-np.round(h); return np.linalg.norm(r@M.T,axis=1)
ev2={'tag-%d//'%tags[e]:e for e in q_by}
dG,dA,mis,qq=[],[],[],[]
for k in both:
    q=q_by[ev2[k]]
    MG=np.array([G[k]['a'],G[k]['b'],G[k]['c']]).T*0.1; MA=np.array([A[k]['a'],A[k]['b'],A[k]['c']]).T*0.1
    dG.append(dist(MG,q)); dA.append(dist(MA,q)); qq.append(np.linalg.norm(q,axis=1))
    # misorientation over the 8 tetragonal-equivalent settings of the authors' basis (422)
    best=9e9
    for sa,sb,sc,sw in [(1,1,1,0),(-1,-1,1,0),(1,-1,-1,0),(-1,1,-1,0),(1,1,1,1),(-1,-1,1,1),(1,-1,-1,1),(-1,1,-1,1)]:
        a,b=(MA[:,1],MA[:,0]) if sw else (MA[:,0],MA[:,1]); T=np.stack([sa*a,sb*b,sc*MA[:,2]],1)
        U1,_,V1=np.linalg.svd(MG@np.linalg.inv(T)); R=U1@V1; ang=np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
        best=min(best,ang)
    mis.append(best)
dG,dA,qq=map(np.concatenate,(dG,dA,qq)); mis=np.array(mis)
print('misorientation GLINT vs authors (deg) p10,50,90',np.round(np.percentile(mis,[10,50,90]),3))
print('found peak to nearest node 1/A, all d:  GLINT p50,90,95,99',np.round(np.percentile(dG,[50,90,95,99]),5),'authors',np.round(np.percentile(dA,[50,90,95,99]),5))
for a,b in ((99,3),(3,2),(2,1.7),(1.7,1.4),(1.4,1.2),(1.2,1.0)):
    s=(1/qq<a)&(1/qq>=b)
    if s.sum(): print(f'  {a}-{b} A n={s.sum():6d} GLINT p50,90 {np.percentile(dG[s],50):.5f} {np.percentile(dG[s],90):.5f}  authors {np.percentile(dA[s],50):.5f} {np.percentile(dA[s],90):.5f}')
