"""Joint Kossel-arc-overlay CBXD indexer (known cell, blind orientation).

Per-streak Kossel-plane fitting recovers G exactly but collapses under realistic noise
(arc ~ straight; curvature ~ noise). The robust route POOLS all streaks: search the
orientation R so that ALL predicted lattice Bragg planes overlay ALL observed streak points
at once. Score(R) = # pooled streak points k_out with min over lattice nodes G=R B hkl of
|k_out.Ghat - |G|/2| < tol AND k_in=k_out-G inside the convergence cone. Coarse SO(3)
random search -> local refine. We test whether this recovers the orientation at the noise
level (2e-4) where the per-streak fit gave 0%.
"""
import sys
import numpy as np

E_keV = 17.5
LAM = 12.398 / E_keV
K = 1.0 / LAM
DMIN = 3.5
CELL = (16.0, 21.0, 25.0, 90.0, 90.0, 90.0)
NA = 0.028
SPUR = 0.30
COSA, SINA = np.cos(NA), np.sin(NA)


def cell_to_B(a, b, c, al, be, ga):
    al, be, ga = np.radians([al, be, ga])
    cx = c * np.cos(be); cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    A = np.array([[a, b * np.cos(ga), cx], [0, b * np.sin(ga), cy],
                  [0, 0, np.sqrt(max(c * c - cx * cx - cy * cy, 0))]])
    return np.linalg.inv(A).T


B = cell_to_B(*CELL)


def rand_rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th; K_ = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K_ + (1 - np.cos(th)) * (K_ @ K_)


def small_rot(rng, sig):
    return rotvec(rng.normal(size=3) * sig)


def hkl_grid():
    qmax = 1.0 / DMIN
    hmax = int(np.ceil(qmax * max(np.linalg.norm(np.linalg.inv(B.T), axis=0)))) + 1
    rh = np.arange(-hmax, hmax + 1)
    H = np.array(np.meshgrid(rh, rh, rh, indexing="ij")).reshape(3, -1).T
    H = H[np.any(H != 0, 1)]
    g = np.linalg.norm((B @ H.T).T, axis=1)
    return H[(g > 1e-6) & (g < 2 * K) & (g <= qmax)]


HS = hkl_grid()


def kossel_basis(G):
    g2 = G @ G; gp2 = G[0]**2 + G[1]**2; rho2 = K * K - g2 / 4.0
    if rho2 <= 0:
        return None, None
    if gp2 < 1e-9:
        return np.array([np.sqrt(rho2), 0, 0]), np.array([0, np.sqrt(rho2), 0])
    az = np.sqrt(rho2 / (1 + G[2]**2 / gp2))
    a = np.array([-az * G[2] * G[0] / gp2, -az * G[2] * G[1] / gp2, az])
    return a, np.cross(G / np.sqrt(g2), a)


def simulate(R, rng, noise):
    chi = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    cc, ss = np.cos(chi), np.sin(chi)
    pts = []
    for h in HS:
        G = R @ (B @ h)
        a, b = kossel_basis(G)
        if a is None:
            continue
        kout = G / 2 + cc[:, None] * a + ss[:, None] * b
        kin = kout - G
        m = (kin[:, 2] > K * COSA) & (np.hypot(kin[:, 0], kin[:, 1]) < K * SINA) & (kout[:, 2] > 0)
        if m.sum() >= 2:
            pts.append(kout[m][::6])                     # thin the arc (sparser streak sampling)
    cents = np.array([p.mean(0) for p in pts]) if pts else np.zeros((0, 3))  # streak centroids
    cents = cents + rng.normal(0, noise, cents.shape)
    real = np.vstack(pts) if pts else np.zeros((0, 3))
    real = real + rng.normal(0, noise, real.shape)
    n_spur = int(SPUR * len(real))                       # spurious points: random forward-cone sphere pts
    sp = rng.normal(size=(n_spur, 3)); sp /= np.linalg.norm(sp, axis=1, keepdims=True); sp *= K
    sp = sp[sp[:, 2] > 0]
    kobs = np.vstack([real, sp])
    lab = np.concatenate([np.ones(len(real), bool), np.zeros(len(sp), bool)])
    return kobs, lab, cents


def score(R, kobs, tol, ret_mask=False):
    G = (R @ (B @ HS.T)).T                                # (Nn,3)
    Gn = np.linalg.norm(G, axis=1); Gh = G / Gn[:, None]
    resid = np.abs(kobs @ Gh.T - Gn[None, :] / 2)         # (Np,Nn)
    kin_z = kobs[:, 2:3] - G[:, 2][None, :]
    kin_xy = np.hypot(kobs[:, 0:1] - G[:, 0][None, :], kobs[:, 1:2] - G[:, 1][None, :])
    cone = (kin_z > K * COSA) & (kin_xy < K * SINA)
    match = (resid < tol) & cone
    idx = match.any(1)
    return (idx if ret_mask else int(idx.sum()))


def soft_score(R, kobs, sigma):
    """smooth overlay score: sum over points of exp(-min_node Bragg-resid^2 / 2sigma^2),
    cone-gated. Wide sigma => broad smooth basin; small sigma => sharp peak."""
    G = (R @ (B @ HS.T)).T
    Gn = np.linalg.norm(G, axis=1); Gh = G / Gn[:, None]
    resid = np.abs(kobs @ Gh.T - Gn[None, :] / 2)
    kin_z = kobs[:, 2:3] - G[:, 2][None, :]
    kin_xy = np.hypot(kobs[:, 0:1] - G[:, 0][None, :], kobs[:, 1:2] - G[:, 1][None, :])
    resid = np.where((kin_z > K * COSA) & (kin_xy < K * SINA), resid, 1e9)
    r2 = (resid.min(1)) ** 2
    return float(np.exp(-r2 / (2 * sigma * sigma)).sum())


def refine(kobs, R0, rng, iters=None):
    """deterministic annealing: Nelder-Mead on the rotation-vector, sigma wide->tight."""
    from scipy.optimize import minimize
    bR = R0
    for sigma in (0.06, 0.035, 0.02, 0.012, 0.007, 0.004, 0.0025, 0.0015):
        res = minimize(lambda w: -soft_score(bR @ rotvec(w), kobs, sigma), np.zeros(3),
                       method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-2, "maxiter": 300})
        bR = bR @ rotvec(res.x)
    return bR


def cent_score(R, q, tol):
    """parallel-beam-style: # streak-centroid back-projections q near a lattice node."""
    G = (R @ (B @ HS.T)).T
    d = np.sqrt(((q[:, None, :] - G[None, :, :]) ** 2).sum(2)).min(1)
    return int((d < tol).sum())


def seed_index(kobs, cents, rng, n_coarse=20000, tol_c=0.03, keep=10):
    """DATA-DRIVEN seeder: index streak CENTROIDS (q=cent-k0, parallel-beam-equivalent,
    degrees-wide basin) by coarse orientation search; refine the top seeds with the JOINT
    Kossel-overlay on the full streak points (anneal tol) and pick the best."""
    q = cents - np.array([0, 0, K])
    cand = sorted(((cent_score(rand_rot(rng), q, tol_c), None) for _ in range(0)), key=lambda t: -t[0])
    cand = []
    for _ in range(n_coarse):
        R = rand_rot(rng); cand.append((cent_score(R, q, tol_c), R))
    cand.sort(key=lambda t: -t[0])
    best_R, best_s = None, -1
    for _, R0 in cand[:keep]:
        R = refine(kobs, R0, rng, iters=300); s = score(R, kobs, 0.0025)
        if s > best_s:
            best_s, best_R = s, R
    return best_R


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "refine"
    ncry = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    print(f"joint Kossel-overlay  lambda={LAM:.3f} NA={NA} cell={CELL[:3]} dmin={DMIN} nodes={len(HS)} mode={mode}")
    if mode == "refine":
        print("  (objective-robustness test: refine from a 5deg-perturbed seed)")
        print(f"{'noise(1/A)':>11} {'real idx':>9} {'spur idx':>9} {'success':>9}")
        for noise in (0.0, 1e-4, 2e-4, 5e-4):
            rng = np.random.default_rng(1); fr = []; fs = []; ok = 0
            for _ in range(ncry):
                Rt = rand_rot(rng); kobs, lab, _ = simulate(Rt, rng, noise)
                R0 = Rt @ small_rot(rng, np.radians(5))
                Rh = refine(kobs, R0, rng)
                idx = score(Rh, kobs, 0.0025, ret_mask=True)
                fr.append(idx[lab].mean()); fs.append(idx[~lab].mean() if (~lab).any() else 0)
                ok += idx[lab].mean() > 0.7
            print(f"{noise:11.1e} {100*np.median(fr):8.0f}% {100*np.median(fs):8.0f}% {ok:6d}/{ncry}")
    else:
        print("  (BLIND: centroid-index seeder -> joint Kossel-overlay refine)")
        for noise in (1e-4, 2e-4):
            rng = np.random.default_rng(2); ok = 0; fr = []
            for c in range(ncry):
                Rt = rand_rot(rng); kobs, lab, cents = simulate(Rt, rng, noise)
                Rh = seed_index(kobs, cents, rng)
                idx = score(Rh, kobs, 0.0025, ret_mask=True)
                fr.append(idx[lab].mean()); ok += idx[lab].mean() > 0.7
            print(f"  noise {noise:.0e}: BLIND success {ok}/{ncry}  median real indexed "
                  f"{100*np.median(fr):.0f}%")
