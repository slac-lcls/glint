"""GLINT M3 basin characterization -- EXTENDED (synthetic). Follows basin_char.py; adds:
  (1) crisp theta@50% basin edge per cell x {short,long} vector (fine grid + linear interp),
  (2) the SHORT-vs-LONG vector axis (38 vs 79 A) -- basin_char only swept short,
  (3) optimizer x restart-ensemble INTERACTION at the edge (does CG's wider basin reach 100% at lower K?).
Reuses the still-frame Ewald cloud; target vectors = rows of inv(A). Run on ampere. Synthetic only."""
import numpy as np, torch, sys
sys.path.insert(0, '/sdf/home/s/smarches/git/glint')
from glint.glint_index import refine_vec, refine_vec_cg, refine_vec_bb, objective, invq_weight
dev = 'cuda'; torch.set_grad_enabled(False)
QMAX = 1/3.; LAM = 1.; TOL = 0.18
CELLS = {'lyso_79_79_38': (79., 79., 38.), 'cubic_60': (60., 60., 60.),
         'cubic_100': (100., 100., 100.), 'ortho_50_65_80': (50., 65., 80.)}

def rot(ax, th):
    ax = ax/np.linalg.norm(ax); c, s = np.cos(th), np.sin(th); x, y, z = ax
    return np.array([[c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
                     [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
                     [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]])

def still_Q(A, HKL):
    q0 = HKL@A.T; m = np.linalg.norm(q0, axis=1) <= QMAX; q0 = q0[m]
    obs = np.abs(q0[:, 2]+0.5*LAM*np.einsum('ij,ij->i', q0, q0)) < 0.004; return q0[obs]

def perturb(vstar, theta, n, rng):
    return np.stack([rot(rng.normal(size=3), np.deg2rad(theta))@vstar for _ in range(n)])

def frames_for(cell, nR, rng):
    B = np.diag(1/np.array(cell)); H = (np.ceil(QMAX*np.array(cell))+1).astype(int)
    HKL = np.mgrid[-H[0]:H[0]+1, -H[1]:H[1]+1, -H[2]:H[2]+1].reshape(3, -1).T
    HKL = HKL[np.any(HKL != 0, 1)].astype(float)
    out = []
    for _ in range(nR):
        R = rot(rng.normal(size=3), rng.uniform(0, np.pi)); A = R@B; Q = still_Q(A, HKL)
        if len(Q) < 40: continue
        inv = np.linalg.inv(A); out.append((Q, inv[np.argmin(cell)], inv[np.argmax(cell)]))  # short, long
    return out

def rec(fn, Qn, vstar, theta, M, steps, rng):
    if len(Qn) < 8: return 0.0
    Qt = torch.tensor(Qn, dtype=torch.float64, device=dev); w = invq_weight(Qt)
    vs = torch.tensor(vstar, dtype=torch.float64, device=dev)
    T0 = torch.tensor(perturb(vstar, theta, M, rng), dtype=torch.float64, device=dev)
    Tout = fn(T0, Qt, w, QMAX, steps=steps)
    d = torch.minimum((Tout-vs).norm(dim=1), (Tout+vs).norm(dim=1))/vs.norm()
    return float((d < 0.05).float().mean())

rng = np.random.default_rng(0); M = 120; NR = 12
FR = {c: frames_for(cell, NR, rng) for c, cell in CELLS.items()}

def sweep(cell, which, fn, steps, theta):
    out = []
    for (Q, vshort, vlong) in FR[cell]:
        tv = vshort if which == 'short' else vlong
        out.append(rec(fn, Q, tv, theta, M, steps, rng))
    return np.mean(out)

def theta50(cell, which, fn=refine_vec, steps=40):
    """Linear-interp theta where recovery crosses 50%, on a fine grid."""
    ths = np.arange(0, 24.01, 1.0); r = np.array([sweep(cell, which, fn, steps, t) for t in ths])
    below = np.where(r < 0.5)[0]
    if len(below) == 0: return float(ths[-1]), r
    j = below[0]
    if j == 0: return 0.0, r
    t0, t1, r0, r1 = ths[j-1], ths[j], r[j-1], r[j]
    return float(t0 + (0.5-r0)*(t1-t0)/(r1-r0+1e-9)), r

print("=== 1) CAPTURE RADIUS theta@50% per cell x {short,long} (GD steps=40) ===")
print("  %-16s %8s %8s   (cell longest edge -> narrower basin?)" % ("cell", "short", "long"))
for c in CELLS:
    ts, _ = theta50(c, 'short'); tl, _ = theta50(c, 'long')
    print("  %-16s %6.1f d %6.1f d" % (c, ts, tl))

print("=== 2) SHORT vs LONG vector recovery vs theta (lyso 38 short / 79 long, GD40) ===")
ths = [0, 2, 4, 6, 8, 10, 14, 20]
for which in ['short', 'long']:
    r = [sweep('lyso_79_79_38', which, refine_vec, 40, t) for t in ths]
    print("  %-6s " % which + " ".join("%d:%.0f%%" % (t, 100*x) for t, x in zip(ths, r)))

print("=== 3) OPTIMIZER x RESTART-ENSEMBLE interaction at edge (lyso short, theta=8) ===")
def rec_restart(Qn, vstar, theta, K, jit, fn, rng, steps=40):
    vs = torch.tensor(vstar, dtype=torch.float64, device=dev)
    Qt = torch.tensor(Qn, dtype=torch.float64, device=dev); w = invq_weight(Qt)
    base = perturb(vstar, theta, M, rng)
    bestf = torch.full((M,), -1e18, dtype=torch.float64, device=dev)
    bestT = torch.zeros(M, 3, dtype=torch.float64, device=dev)
    for k in range(K):
        arr = base if k == 0 else np.stack([rot(rng.normal(size=3), np.deg2rad(jit))@base[i] for i in range(M)])
        Tout = fn(torch.tensor(arr, dtype=torch.float64, device=dev), Qt, w, QMAX, steps=steps)
        f = objective(Tout, Qt, w, TOL, sharp=True)[0]
        pick = f > bestf; bestT[pick] = Tout[pick]; bestf[pick] = f[pick]
    d = torch.minimum((bestT-vs).norm(dim=1), (bestT+vs).norm(dim=1))/vs.norm()
    return float((d < 0.05).float().mean())

for name, fn in [('GD', refine_vec), ('CG', refine_vec_cg)]:
    row = []
    for K in [1, 3, 5, 9, 15, 25]:
        r = np.mean([rec_restart(f[0], f[1], 8, K, 4, fn, rng) for f in FR['lyso_79_79_38']])
        row.append((K, r))
    print("  %-3s " % name + " ".join("K%d:%.0f%%" % (K, 100*x) for K, x in row))
print("DONE")
