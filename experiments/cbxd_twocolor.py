"""Two-color convergent-beam CBXD: two Ewald spheres crack the blind wall.

Extends cbxd_joint.py (single-color Kossel-arc overlay) to a TWO-COLOR convergent beam.
Physics: two photon energies E1,E2 -> two Ewald radii K1,K2. A lattice reflection G can be
Kossel-excited at EITHER colour; each excitation is an arc (convergence cone half-angle NA).
A two-colour beam therefore lays down ~2x as many arcs as single colour -> the orientation
objective tower is ~2x taller -> recoverable at LOW NA / more noise, where single colour is
under-determined (one thin Ewald slice, too few arcs).

The two-Ewald accumulator: score orientation R by voting every observed k_out against the
predicted Kossel PLANE of every lattice node (the plane residual |k_out.Ghat - |G|/2| is
COLOUR-FREE) AND the convergence cone at EITHER radius K1 or K2, requiring |k_in| == that K.
A point excited at colour 2 is explained only by the K2 cone; indexing two-colour data with
ONE sphere leaves the other colour's arcs unexplained (they act like ~50% extra spurious) ->
the two-sphere test is NECESSARY, not merely 'more data'.

Unlike a difference-vector approach that emits each peak at both q's (which locks a SUPERCELL
ghost from the wrong-lambda copies), the arc accumulator never emits ghost points: colour is
assigned implicitly by which cone/|k_in| fits. No wrong-lambda copies -> no supercell bias.

Money result (3-way, swept over NA):
  (1) 1-colour data, 1-sphere index  = single-colour baseline
  (2) 2-colour data, 1-sphere index  = naive: ignore the 2nd colour -> its arcs are pure noise
  (3) 2-colour data, 2-sphere index  = the two-Ewald accumulator
Expect a low-NA band where (1),(2) fail and (3) recovers the blind orientation = the wall moves.
(3) also assigns each peak to lambda1/lambda2 (which cone fits) = the within-shot XTCAV-labelled
consistency check = cross-frame consensus INSIDE one shot.

This file is STEP 1 only: the forward sim + accumulator testbed. The measured results, the two
evaluation traps, and the step-2 spec item (the centroid seeder's parallel-beam assumption breaks
for wide cones) are written up in CBXD_TWOCOLOR_STEP1.md next to this file -- read that first.

Steps 2-4 are Yuan Ni's:
  (2) replace the placeholder random-SO(3) seeder with a real arc-Hough orientation accumulator
      over SO(3) -- the money figure: blind-recovery-vs-single-colour at matched candidate budget
      across (NA, noise).
  (3) real per-shot XTCAV energies -> per-peak lambda assignment + within-shot consistency filter
      (assign_colour here is the stub showing the mechanism).
  (4) run on the real Chapman two-colour frames (337 TB), which needs full BayFAI-style geometry
      refinement + peakfinder8 first -- on that data geometry, not the beam, is the barrier.

modes:  contrast [na]     fast landscape diagnostic: truth-vs-decoy tower height, 3 configs
        blind [na] [ncry] full blind recovery sweep over NA, 3 configs
"""
import os
import sys
import time
import numpy as np

# --- two-colour beam (LCLS hard-X two-colour; ~2.5 keV split) ---
E1_keV, E2_keV = 17.5, 15.0
LAM1 = 12.398 / E1_keV; K1 = 1.0 / LAM1
LAM2 = 12.398 / E2_keV; K2 = 1.0 / LAM2
KMAX = max(K1, K2)
DMIN = 3.5
CELL = (16.0, 21.0, 25.0, 90.0, 90.0, 90.0)   # same cell as cbxd_joint (orthorhombic).
# The real step-4 target is the MONOCLINIC iodine cell ~11/13/9/90/102/90. Swapping this in (plus
# mosaicity, a per-reflection convergence q-disk rather than a smear, and the true lambda1/lambda2
# split) is the step-2/3 realism work -- see CBXD_TWOCOLOR_STEP1.md. Note the ortho default makes
# the 222 symmetry-aware success test below necessary; a monoclinic cell changes that subgroup.
NA0 = 0.028                                     # default convergence half-angle (rad)
SPUR = 0.30                                     # spurious fraction (of real points)

KS_1 = [K1]                                      # single-colour index
KS_2 = [K1, K2]                                  # two-colour index


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


def ang_between(Ra, Rb):
    """rotation angle (deg) taking Ra to Rb."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


# proper rotation operators of the crystal point group, in the crystal Cartesian frame.
# The default cell is orthorhombic -> Laue mmm -> proper subgroup 222 (identity + three 180deg
# rotations about a,b,c). A recovered orientation equals truth only MODULO these, so blind
# success must be judged symmetry-aware (else a correct symmetry-equivalent reads as ~180deg off).
SYMOPS = [np.diag([1.0, 1.0, 1.0]), np.diag([1.0, -1.0, -1.0]),
          np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0])]


def ang_sym(Rt, Rh):
    """min rotation angle (deg) between Rh and truth Rt over the crystal point group."""
    return min(ang_between(Rt, Rh @ S) for S in SYMOPS)


def hkl_grid():
    qmax = 1.0 / DMIN
    hmax = int(np.ceil(qmax * max(np.linalg.norm(np.linalg.inv(B.T), axis=0)))) + 1
    rh = np.arange(-hmax, hmax + 1)
    H = np.array(np.meshgrid(rh, rh, rh, indexing="ij")).reshape(3, -1).T
    H = H[np.any(H != 0, 1)]
    g = np.linalg.norm((B @ H.T).T, axis=1)
    return H[(g > 1e-6) & (g < 2 * KMAX) & (g <= qmax)]


HS = hkl_grid()


def kossel_basis(G, K):
    """orthonormal-scaled (a,b), |a|=|b|=rho, spanning the Kossel circle of node G at radius K:
    k_out = G/2 + cos(chi) a + sin(chi) b lies on |k_out|=K with k_out.Ghat = |G|/2."""
    g2 = G @ G; gp2 = G[0]**2 + G[1]**2; rho2 = K * K - g2 / 4.0
    if rho2 <= 0:
        return None, None
    if gp2 < 1e-9:
        return np.array([np.sqrt(rho2), 0, 0]), np.array([0, np.sqrt(rho2), 0])
    az = np.sqrt(rho2 / (1 + G[2]**2 / gp2))
    a = np.array([-az * G[2] * G[0] / gp2, -az * G[2] * G[1] / gp2, az])
    return a, np.cross(G / np.sqrt(g2), a)


def simulate(R, rng, noise, na, Ks, nchi=720, thin=6):
    """emit the convergent-beam Kossel arcs for orientation R over colours Ks.
    returns kobs (Np,3), lab (real vs spurious), col (0/1 true colour, -1 spurious),
    cents (streak centroids), cent_col (their colour)."""
    chi = np.linspace(0, 2 * np.pi, nchi, endpoint=False)
    cc, ss = np.cos(chi), np.sin(chi)
    cosa, sina = np.cos(na), np.sin(na)
    pts, cols, cents, cent_col = [], [], [], []
    for h in HS:
        G = R @ (B @ h)
        for ci, K in enumerate(Ks):
            a, b = kossel_basis(G, K)
            if a is None:
                continue
            kout = G / 2 + cc[:, None] * a + ss[:, None] * b
            kin = kout - G
            m = (kin[:, 2] > K * cosa) & (np.hypot(kin[:, 0], kin[:, 1]) < K * sina) & (kout[:, 2] > 0)
            if m.sum() >= 2:
                p = kout[m][::thin]
                pts.append(p); cols.append(np.full(len(p), ci))
                cents.append(p.mean(0)); cent_col.append(ci)
    real = np.vstack(pts) if pts else np.zeros((0, 3))
    real_col = np.concatenate(cols) if cols else np.zeros(0, int)
    cents = np.array(cents) if cents else np.zeros((0, 3))
    cent_col = np.array(cent_col, int)
    real = real + rng.normal(0, noise, real.shape)
    cents = cents + rng.normal(0, noise, cents.shape)
    n_spur = int(SPUR * len(real))                         # random forward-cone shell points
    sp = rng.normal(size=(n_spur, 3)); sp /= np.linalg.norm(sp, axis=1, keepdims=True); sp *= KMAX
    sp = sp[sp[:, 2] > 0]
    kobs = np.vstack([real, sp])
    lab = np.concatenate([np.ones(len(real), bool), np.zeros(len(sp), bool)])
    col = np.concatenate([real_col, -np.ones(len(sp), int)])
    return kobs, lab, col, cents, cent_col


def _cone_masks(kobs, G, na, Ks, kmag_tol):
    """boolean (Np,Nn) cone acceptance per colour, plus the per-colour |k_in| deviations."""
    cosa, sina = np.cos(na), np.sin(na)
    kx = kobs[:, 0:1] - G[:, 0][None, :]
    ky = kobs[:, 1:2] - G[:, 1][None, :]
    kz = kobs[:, 2:3] - G[:, 2][None, :]
    kin_xy = np.hypot(kx, ky); kin_mag = np.hypot(kin_xy, kz)
    cones = []
    for K in Ks:
        cones.append((np.abs(kin_mag - K) < kmag_tol) & (kz > K * cosa) & (kin_xy < K * sina))
    return cones, kin_mag


def score(R, kobs, tol, na, Ks, kmag_tol=1e9, ret_mask=False):
    """hard two-Ewald overlay: a point is indexed if it lies on some node's Kossel PLANE
    (colour-free) AND inside the convergence cone at some colour K in Ks. The cone's
    kz>K*cosNA threshold is K-scaled, so it already separates colours (a K2 point cannot pass
    the K1 threshold); the explicit |k_in|=K gate (kmag_tol) is left OFF here to keep the
    angular basin wide, and used tight only in assign_colour for the lambda label."""
    G = (R @ (B @ HS.T)).T
    Gn = np.linalg.norm(G, axis=1); Gh = G / Gn[:, None]
    resid = np.abs(kobs @ Gh.T - Gn[None, :] / 2)          # (Np,Nn) plane residual
    cones, _ = _cone_masks(kobs, G, na, Ks, kmag_tol)
    cone = np.zeros_like(resid, bool)
    for cm in cones:
        cone |= cm
    idx = ((resid < tol) & cone).any(1)
    return (idx if ret_mask else int(idx.sum()))


def assign_colour(R, kobs, tol, na, Ks, kmag_tol=0.02):
    """for each observed point return the best-fitting colour (-1 if unindexed) = the
    within-shot lambda label the XTCAV energies would confirm."""
    G = (R @ (B @ HS.T)).T
    Gn = np.linalg.norm(G, axis=1); Gh = G / Gn[:, None]
    resid = np.abs(kobs @ Gh.T - Gn[None, :] / 2)
    cones, _ = _cone_masks(kobs, G, na, Ks, kmag_tol)
    out = np.full(len(kobs), -1, int)
    bestr = np.full(len(kobs), tol)
    for ci, cm in enumerate(cones):
        r = np.where(cm & (resid < tol), resid, np.inf).min(1)
        take = r < bestr
        out[take] = ci; bestr[take] = r[take]
    return out


def soft_score(R, kobs, sigma, na, Ks, kmag_tol=1e9):
    """smooth two-Ewald overlay for annealed refinement (direction-gated cone; see score)."""
    G = (R @ (B @ HS.T)).T
    Gn = np.linalg.norm(G, axis=1); Gh = G / Gn[:, None]
    resid = np.abs(kobs @ Gh.T - Gn[None, :] / 2)
    cones, _ = _cone_masks(kobs, G, na, Ks, kmag_tol)
    cone = np.zeros_like(resid, bool)
    for cm in cones:
        cone |= cm
    resid = np.where(cone, resid, 1e9)
    r2 = resid.min(1) ** 2
    return float(np.exp(-r2 / (2 * sigma * sigma)).sum())


# (residual sigma [1/A], angular Nelder-Mead simplex step [rad]) annealed wide -> tight.
# The explicit angular step is essential: scipy NM from x0=0 builds a ~1e-4 rad simplex by
# default and cannot move degrees; pairing each residual sigma with a real angular scale is
# what gives the refiner a capture radius of many degrees.
_SCHED = ((0.10, 0.22), (0.06, 0.14), (0.035, 0.08), (0.02, 0.045),
          (0.012, 0.025), (0.007, 0.014), (0.004, 0.008), (0.0025, 0.004))


def refine(kobs, R0, na, Ks, sched=_SCHED):
    """deterministic annealing: Nelder-Mead on the rotation vector, wide -> tight, with an
    explicit shrinking angular simplex so a degrees-off seed anneals into the sharp arc basin."""
    from scipy.optimize import minimize
    bR = R0
    for sigma, step in sched:
        simplex = np.vstack([np.zeros(3), np.eye(3) * step])
        res = minimize(lambda w: -soft_score(bR @ rotvec(w), kobs, sigma, na, Ks), np.zeros(3),
                       method="Nelder-Mead",
                       options={"initial_simplex": simplex, "xatol": 1e-4, "fatol": 1e-2, "maxiter": 300})
        bR = bR @ rotvec(res.x)
    return bR


def cent_score(R, cents, Ks, tol):
    """coarse parallel-beam-equivalent seeder: back-project each streak centroid at EVERY
    colour (q = cent - K zhat), count centroids landing near a lattice node at any colour."""
    G = (R @ (B @ HS.T)).T
    best = np.full(len(cents), 1e9)
    for K in Ks:
        q = cents - np.array([0.0, 0.0, K])
        d = np.sqrt(((q[:, None, :] - G[None, :, :]) ** 2).sum(2)).min(1)
        best = np.minimum(best, d)
    return int((best < tol).sum())


def seed_index(kobs, cents, na, Ks, rng, n_coarse=int(os.environ.get("TC_NCOARSE", 20000)),
               tol_c=0.02, keep=15):
    """blind: two-Ewald centroid seeder (parallel-beam basin) -> anneal the top seeds with the
    full arc overlay. n_coarse sized so ~several samples land within a basin of the truth
    (fraction of SO(3) within 8deg ~ 1.6e-4). tol_c must be well BELOW the reciprocal-node
    spacing (~0.049 here) or every orientation matches some node and the seeder is blind."""
    cand = []
    for _ in range(n_coarse):
        R = rand_rot(rng); cand.append((cent_score(R, cents, Ks, tol_c), R))
    cand.sort(key=lambda t: -t[0])
    best_R, best_s = None, -1
    for _, R0 in cand[:keep]:
        R = refine(kobs, R0, na, Ks); s = score(R, kobs, 0.0025, na, Ks)
        if s > best_s:
            best_s, best_R = s, R
    return best_R


# ---------------- experiments ----------------

CONFIGS = [("1-col data / 1-sphere", KS_1, KS_1),      # (label, sim colours, index colours)
           ("2-col data / 1-sphere", KS_2, KS_1),      # naive
           ("2-col data / 2-sphere", KS_2, KS_2)]      # two-Ewald accumulator


def run_contrast(na, ncry=6, seed=1):
    """Fast mechanistic diagnostic: at the TRUE orientation vs a pool of random decoys, how
    tall is the accumulator tower? Report truth score, best-decoy score, contrast, and truth's
    rank among decoys (rank 1 => truth is the global max => blind search will find it)."""
    print(f"CONTRAST  na={na:.3f} rad ({na*1e3:.0f} mrad)  ncry={ncry}  nodes={len(HS)}")
    print(f"  E1/E2={E1_keV}/{E2_keV} keV  K1/K2={K1:.3f}/{K2:.3f}  cell={CELL[:3]}")
    print(f"  {'config':<24}{'#arcs':>7}{'truth':>8}{'decoy':>8}{'contrast':>10}{'rank/pool':>11}")
    ndec = 3000
    rng0 = np.random.default_rng(seed)
    Rts = [rand_rot(rng0) for _ in range(ncry)]              # identical crystals across configs
    for label, Ksim, Kidx in CONFIGS:
        rng = np.random.default_rng(seed + 100)
        tru, dec, con, rk, na_arcs = [], [], [], [], []
        for Rt in Rts:
            kobs, lab, col, cents, _ = simulate(Rt, rng, 2e-4, na, Ksim)
            na_arcs.append(int(lab.sum()))
            st = score(Rt, kobs, 0.0025, na, Kidx)
            decoys = np.array([score(rand_rot(rng), kobs, 0.0025, na, Kidx) for _ in range(ndec)])
            tru.append(st); dec.append(int(decoys.max()))
            con.append(st / (decoys.mean() + 1e-9))
            rk.append(int((decoys >= st).sum()) + 1)          # rank of truth (1 = best)
        print(f"  {label:<24}{int(np.median(na_arcs)):>7}{int(np.median(tru)):>8}"
              f"{int(np.median(dec)):>8}{np.median(con):>10.1f}{f'{int(np.median(rk))}/{ndec}':>11}")
    print("  (rank 1 => truth is the global accumulator max => recoverable blind)")


def run_blind(na, ncry=6, seed=2):
    """Full blind recovery: seed (coarse two-Ewald centroid search) -> anneal -> check the
    recovered orientation against ground truth. Reports success, angular error, indexed
    fraction, and lambda-label accuracy (the within-shot XTCAV consistency)."""
    print(f"BLIND  na={na:.3f} rad ({na*1e3:.0f} mrad)  ncry={ncry}  nodes={len(HS)}  "
          f"budget={int(os.environ.get('TC_NCOARSE', 20000))}")
    print(f"  {'config':<24}{'success':>9}{'med.symang':>12}{'med.idx':>9}{'lambda-acc':>11}")
    for label, Ksim, Kidx in CONFIGS:
        rng = np.random.default_rng(seed)
        ok = 0; angs = []; fr = []; lacc = []
        for _ in range(ncry):
            Rt = rand_rot(rng)
            kobs, lab, col, cents, _ = simulate(Rt, rng, 2e-4, na, Ksim)
            if len(cents) < 3:
                angs.append(180.0); fr.append(0.0); continue
            Rh = seed_index(kobs, cents, na, Kidx, rng)
            if Rh is None:
                angs.append(180.0); fr.append(0.0); continue
            a = ang_sym(Rt, Rh)                               # symmetry-aware (222 for this cell)
            idx = score(Rh, kobs, 0.0025, na, Kidx, ret_mask=True)
            frac = idx[lab].mean() if lab.any() else 0.0
            angs.append(a); fr.append(frac)
            success = a < 2.0                                 # correct orientation up to symmetry
            ok += success
            if len(Kidx) == 2 and success:                    # lambda-assignment accuracy
                ca = assign_colour(Rh, kobs, 0.0025, na, Kidx)
                real = lab & (ca >= 0)
                lacc.append((ca[real] == col[real]).mean() if real.any() else 0.0)
        la = f"{100*np.median(lacc):.0f}%" if lacc else "  -"
        print(f"  {label:<24}{f'{ok}/{ncry}':>9}{np.median(angs):>13.2f}"
              f"{100*np.median(fr):>8.0f}%{la:>11}")


def run_capture(na, ncry=8, seed=3):
    """Capture-radius: how far from truth can refinement START and still return to it?
    A wider/deeper basin => the coarse seeder needs fewer candidates to hit it => the wall
    moves. This isolates the two-Ewald OBJECTIVE from the coarse-seeder engineering (a separate
    follow-on step). Report, per seed perturbation, the fraction of crystals recovered to <1 deg."""
    perts = [3, 6, 10, 16, 25]
    print(f"CAPTURE  na={na:.3f} rad ({na*1e3:.0f} mrad)  ncry={ncry}  nodes={len(HS)}  "
          f"noise=2e-4")
    print(f"  recovery (<1 deg) vs seed-perturbation (deg):")
    print(f"  {'config':<24}{'#arcs':>6}" + "".join(f"{str(p)+'deg':>8}" for p in perts))
    for label, Ksim, Kidx in CONFIGS:
        rng = np.random.default_rng(seed)
        rec = {p: 0 for p in perts}; na_arcs = []
        for _ in range(ncry):
            Rt = rand_rot(rng)
            kobs, lab, col, cents, _ = simulate(Rt, rng, 2e-4, na, Ksim)
            na_arcs.append(int(lab.sum()))
            for p in perts:
                R0 = Rt @ small_rot(rng, np.radians(p))
                Rh = refine(kobs, R0, na, Kidx)
                if ang_between(Rt, Rh) < 1.0:
                    rec[p] += 1
        cells = "".join(f"{f'{rec[p]}/{ncry}':>8}" for p in perts)
        print(f"  {label:<24}{int(np.median(na_arcs)):>6}{cells}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "contrast"
    na = float(sys.argv[2]) if len(sys.argv) > 2 else NA0
    ncry = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    t0 = time.perf_counter()
    if mode == "contrast":
        run_contrast(na, ncry)
    elif mode == "capture":
        run_capture(na, ncry)
    elif mode == "sweep":                                     # contrast across an NA band
        for na in (0.008, 0.012, 0.018, 0.028):
            run_contrast(na, ncry); print()
    else:
        run_blind(na, ncry)
    print(f"[{time.perf_counter()-t0:.1f}s]")
