"""Convergent-beam CBXD: the convergence cone is the lever; a second colour is a bonus term.

Extends cbxd_joint.py (single-colour Kossel-arc overlay) to a convergent beam that may carry one or
two photon energies. This file was originally written the other way round -- "two Ewald spheres crack
the blind wall" -- and the `cover` mode added later shows why that headline was too strong. What
thickens the Ewald slice is CONVERGENCE: tilting k_in over a cone of half-angle NA sweeps the sphere
through a shell of radial width ~K*NA, and every reflection inside that shell is reached, each along
an ARC whose direction carries the out-of-plane constraint a parallel beam cannot supply. A second
photon energy adds a second, DISCRETE sphere at dK/K = dE/E, which buys new reflections only once it
lands outside the shell convergence has already swept -- that is, once dE/E > NA.

Measured (`cover`, pure geometry, median over 16 crystals) as the extra distinct reflections a second
colour reaches, against what a step in NA reaches on its own:

    NA      2-colour gain at dE/E=15.4%     NA step        single-colour gain
    14 mrad         1.63x                   14 -> 20 mrad        1.75x
    20 mrad         1.37x                   20 -> 28 mrad        1.43x
    28 mrad         1.23x                   28 -> 40 mrad        1.60x
    40 mrad         1.17x

A ~1.4x step in convergence beats a 2.5 keV energy split at every NA, and the two-colour gain SHRINKS
as NA grows, because a wider shell has already swallowed the second sphere. By dE/E = 1% the gain is
exactly 1.00x at every NA tested.

The accumulator: score orientation R by voting every observed k_out against the predicted Kossel
PLANE of every lattice node (the plane residual |k_out.Ghat - |G|/2| is COLOUR-FREE) AND the
convergence cone at EITHER radius K1 or K2, requiring |k_in| == that K. A point excited at colour 2
is explained only by the K2 cone; indexing two-colour data with ONE sphere leaves the other colour's
arcs unexplained (they act like ~50% extra spurious) -> the two-sphere test is NECESSARY to BOOK the
points, which is not the same as its being informative.

That distinction is the trap this file walked into and `split` now exposes. The cone gate kz > K*cosNA
is K-scaled, so it keeps telling the two radii apart all the way down to dE/E ~ 1-cos(NA) ~ NA^2/2 --
2e-4 at NA = 20 mrad, a HUNDRED times below the dE/E ~ NA where the second sphere stops adding
coverage. In between, the accumulator still labels every point by colour and still reports twice the
votes at truth (23 vs the one-sphere 11 at dE/E = 0.002) while `cover` says the second colour reaches
1.00x the reflections. Vote count cannot see the difference. Coverage can.

Unlike a difference-vector approach that emits each peak at both q's (which locks a SUPERCELL
ghost from the wrong-lambda copies), the arc accumulator never emits ghost points: colour is
assigned implicitly by which cone/|k_in| fits. No wrong-lambda copies -> no supercell bias.

Three configs, held from the original write-up because the comparison is still the right one -- only
its headline changed:
  (1) 1-colour data, 1-sphere index  = single-colour baseline
  (2) 2-colour data, 1-sphere index  = naive: ignore the 2nd colour -> its arcs are pure noise
  (3) 2-colour data, 2-sphere index  = the two-Ewald accumulator
At a WIDE split (3) beats (1) and (2), and that result stands -- see CBXD_TWOCOLOR_STEP1.md. What
does not stand is reading it as a general two-colour win: it is a dE/E = 15.4% win, and 15.4% is a
wide split for a hard-X-ray source. Sources that make their second pulse by splitting and DELAYING
rather than by retuning deliver dE/E orders of magnitude smaller, and there the second sphere is
inside the convergence shell and (3) reduces to (1).
(3) also assigns each peak to lambda1/lambda2 (which cone fits) = the within-shot consistency check.
That label is only measurable while dE/E stays above the bookkeeping limit above; below it
assign_colour still returns a label and the label means nothing.

This file is STEP 1 only: the forward sim + accumulator testbed. The measured results, the two
evaluation traps, and the step-2 spec item (the centroid seeder's parallel-beam assumption breaks
for wide cones) are written up in CBXD_TWOCOLOR_STEP1.md next to this file -- read that first.

Steps 2-4 are Yuan Ni's:
  (2) replace the placeholder random-SO(3) seeder with a real arc-Hough orientation accumulator
      over SO(3) -- the money figure: blind-recovery-vs-single-colour at matched candidate budget
      across (NA, noise).
  (3) per-peak lambda assignment + within-shot consistency filter (assign_colour here is the stub
      showing the mechanism) -- but FIRST establish that the source's dE/E is above 1-cos(NA), or
      there is no label to assign and the step is moot.
  (4) run on the real Chapman two-colour frames (337 TB), which needs full BayFAI-style geometry
      refinement + peakfinder8 first -- on that data geometry, not the beam, is the barrier.

modes:  contrast [na]     fast landscape diagnostic: truth-vs-decoy tower height, 3 configs
        split [na] [ncry] sweep the ENERGY SPLIT at fixed convergence -- how much dE/E the
                          two-Ewald win actually needs, and where the three configs converge
        splitcap [na] [n] the same sweep scored on CAPTURE RADIUS, which duplicated points
                          cannot inflate the way they inflate the vote count
        cover [na] [ncry] pure geometry: distinct reflections reached by one colour vs two, as the
                          split closes -- the coverage claim, measured without a noisy statistic
        coverna [_] [n]   the same measure against CONVERGENCE at one colour: what NA buys
        blind [na] [ncry] full blind recovery sweep over NA, 3 configs
"""
import os
import sys
import time
import numpy as np

# --- two-colour beam (LCLS hard-X two-colour; ~2.5 keV split) ---
# The split is the SWEPT variable, not a constant of the problem: `set_beam` rebinds it and the
# `split` mode sweeps it, because the two-Ewald win is a function of dE/E and the default 17.5/15.0
# is a 15.4% split -- far wider than a beamline that separates its two pulses in TIME will deliver.
# Defaults are unchanged so every number already reported reproduces.
E1_keV, E2_keV = float(os.environ.get("TC_E1", 17.5)), float(os.environ.get("TC_E2", 15.0))
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


def set_beam(e1_keV, e2_keV):
    """Rebind the beam to a new pair of energies, recomputing everything derived from them.

    K = E/hc, so dK/K = dE/E: the two Ewald radii are separated by exactly the fractional energy
    split. The reflection list depends on KMAX and the index/sim colour sets on K1,K2, so all of
    them are rebuilt here -- CONFIGS included, since it CAPTURES the two lists by reference.
    """
    global E1_keV, E2_keV, LAM1, LAM2, K1, K2, KMAX, HS, KS_1, KS_2, CONFIGS
    E1_keV, E2_keV = float(e1_keV), float(e2_keV)
    LAM1 = 12.398 / E1_keV; K1 = 1.0 / LAM1
    LAM2 = 12.398 / E2_keV; K2 = 1.0 / LAM2
    KMAX = max(K1, K2)
    HS = hkl_grid()
    KS_1 = [K1]; KS_2 = [K1, K2]
    CONFIGS = [("1-col data / 1-sphere", KS_1, KS_1),
               ("2-col data / 1-sphere", KS_2, KS_1),
               ("2-col data / 2-sphere", KS_2, KS_2)]


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


def run_split(na, ncry=6, seed=1, ndec=3000):
    """How much energy SPLIT does the two-Ewald win actually require?

    The headline result was measured at one split, 17.5/15.0 keV = dE/E 15.4%, and reported as
    though two colours were the lever. But K = E/hc, so the two Ewald radii are separated by exactly
    dE/E -- shrink it and the second sphere slides into the first. Two thresholds govern that, they
    are far apart, and the gap between them is the trap:

      COVERAGE.  Convergence already smears each sphere into a shell: tilting k_in over a cone of
      half-angle NA moves the sphere surface by ~K*NA. A second sphere adds reciprocal-space
      coverage only once it sits OUTSIDE that shell, i.e. once dE/E > NA -- 2% at NA = 20 mrad.
      Below it the two shells overlap and the second colour contributes duplicate points.

      BOOKKEEPING.  The accumulator separates the colours with the K-scaled direction gate
      kz > K cos(NA), which keeps working until K2 > K1 cos(NA), i.e. down to dE/E ~ NA^2/2 --
      2e-4 at the same NA, a HUNDRED times smaller.

    So between dE/E ~ NA^2/2 and dE/E ~ NA the accumulator still labels every point by colour and
    still reports twice the votes, while the second sphere has stopped adding anything. Vote count
    cannot see the difference; `splitcap` scores the same sweep on capture radius, which can.

    The mean energy is held fixed (16.25 keV, so the widest split reproduces the original 17.5/15.0
    exactly) to keep the reflection list and arc statistics comparable across the sweep.
    """
    e1_0, e2_0 = E1_keV, E2_keV
    Ec = 16.25
    fracs = tuple(float(v) for v in os.environ["TC_SPLITS"].split(",")) \
        if os.environ.get("TC_SPLITS") else (0.15385, 0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.0)
    print(f"SPLIT SWEEP  na={na:.3f} rad ({na*1e3:.0f} mrad)  ncry={ncry}  decoys={ndec}")
    print(f"  coverage threshold  dE/E ~ NA      = {na:.2e}   "
          f"(below this the 2nd sphere sits inside the convergence-thickened shell)")
    print(f"  bookkeeping limit   dE/E ~ 1-cos(NA) = {1 - np.cos(na):.2e}   "
          f"(below this the cone gate can no longer label the two radii at all)")
    print(f"  {'dE/E':>8}  {'E1/E2 keV':>13}  " +
          "".join(f"{lab.split(' data ')[0] + lab.split('/')[-1]:>17}" for lab, _, _ in CONFIGS))
    print(f"  {'':>8}  {'':>13}  " + "".join(f"{'arcs truth decoy':>17}" for _ in CONFIGS))
    try:
        for f in fracs:
            set_beam(Ec * (1 + f / 2), Ec * (1 - f / 2))
            rng0 = np.random.default_rng(seed)
            Rts = [rand_rot(rng0) for _ in range(ncry)]      # identical crystals at every split
            cells = []
            for label, Ksim, Kidx in CONFIGS:
                rng = np.random.default_rng(seed + 100)
                tru, dec, arcs = [], [], []
                for Rt in Rts:
                    kobs, lab, col, cents, _ = simulate(Rt, rng, 2e-4, na, Ksim)
                    arcs.append(int(lab.sum()))
                    st = score(Rt, kobs, 0.0025, na, Kidx)
                    decoys = np.array([score(rand_rot(rng), kobs, 0.0025, na, Kidx)
                                       for _ in range(ndec)])
                    tru.append(st); dec.append(int(decoys.max()))
                cells.append(f"{int(np.median(arcs)):>5}{int(np.median(tru)):>6}"
                             f"{int(np.median(dec)):>6}")
            print(f"  {f:>8.5f}  {E1_keV:6.3f}/{E2_keV:6.3f}  " +
                  "".join(f"{c:>17}" for c in cells), flush=True)
    finally:
        set_beam(e1_0, e2_0)                                  # never leave the module retuned
    print("  (truth = accumulator votes at the true orientation; decoy = best of the random pool)")


def excited(R, na, Ks, nchi=720):
    """Which reflections does each colour actually excite? Returns a list of hkl-index sets, one per
    colour. This is the geometry alone -- no noise, no scoring, no refinement -- so it answers the
    coverage question without a recovery statistic's counting noise in the way."""
    chi = np.linspace(0, 2 * np.pi, nchi, endpoint=False)
    cc, ss = np.cos(chi), np.sin(chi)
    cosa, sina = np.cos(na), np.sin(na)
    sets = [set() for _ in Ks]
    for j, h in enumerate(HS):
        G = R @ (B @ h)
        for ci, K in enumerate(Ks):
            a, b = kossel_basis(G, K)
            if a is None:
                continue
            kout = G / 2 + cc[:, None] * a + ss[:, None] * b
            kin = kout - G
            m = (kin[:, 2] > K * cosa) & (np.hypot(kin[:, 0], kin[:, 1]) < K * sina) & (kout[:, 2] > 0)
            if m.sum() >= 2:
                sets[ci].add(j)
    return sets


def run_cover(na, ncry=12, seed=1):
    """Does the second colour excite DIFFERENT reflections, or the same ones twice?

    This is the claim the whole two-colour argument rests on, and it is pure geometry, so measure it
    as geometry. `arcs` counts excitations and is what the original write-up reported doubling; the
    honest quantity is the UNION of distinct reflections reached. Convergence smears each Ewald
    sphere into a shell of radial width ~K*NA, so a second sphere only reaches past it once
    dE/E > NA. Below that the two colours light up the same reflections and 'twice the arcs' is
    twice the bookkeeping.
    """
    e1_0, e2_0 = E1_keV, E2_keV
    Ec = 16.25
    fracs = tuple(float(v) for v in os.environ["TC_SPLITS"].split(",")) \
        if os.environ.get("TC_SPLITS") else (0.15385, 0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0)
    print(f"COVERAGE  na={na:.3f} rad ({na*1e3:.0f} mrad)  ncry={ncry}   "
          f"coverage threshold dE/E ~ NA = {na:.3f}")
    print(f"  {'dE/E':>8}  {'col1':>6}{'col2':>6}{'union':>7}{'both':>6}"
          f"{'union/col1':>12}{'arcs/col1':>11}")
    try:
        for f in fracs:
            set_beam(Ec * (1 + f / 2), Ec * (1 - f / 2))
            rng = np.random.default_rng(seed)
            n1, n2, nu, nb = [], [], [], []
            for _ in range(ncry):
                s1, s2 = excited(rand_rot(rng), na, [K1, K2])
                n1.append(len(s1)); n2.append(len(s2))
                nu.append(len(s1 | s2)); nb.append(len(s1 & s2))
            n1 = np.array(n1, float); n2 = np.array(n2, float)
            nu = np.array(nu, float); nb = np.array(nb, float)
            keep = n1 > 0                                     # ratios PER CRYSTAL then median:
            gain = np.median(nu[keep] / n1[keep])             # medians of numerator and denominator
            arcr = np.median((n1[keep] + n2[keep]) / n1[keep])  # separately are not a ratio
            print(f"  {f:>8.5f}  {np.median(n1):>6.0f}{np.median(n2):>6.0f}{np.median(nu):>7.0f}"
                  f"{np.median(nb):>6.0f}{gain:>12.2f}{arcr:>11.2f}", flush=True)
    finally:
        set_beam(e1_0, e2_0)
    print("  union/col1 = the real coverage gain;  arcs/col1 = what counting excitations reports")


def run_coverna(ncry=12, seed=1):
    """The other lever, measured the same way: what does CONVERGENCE buy, at a single colour?

    Tilting k_in over a cone of half-angle NA sweeps the Ewald sphere through a shell of radial
    width ~K*NA, and unlike a second discrete sphere that shell is continuous -- every reflection
    inside it is reached, and each is reached along an ARC whose direction carries the out-of-plane
    constraint. This is the quantity to put a two-colour gain next to.
    """
    e1_0, e2_0 = E1_keV, E2_keV
    set_beam(16.25, 16.25)                                    # single colour, so NA is the only lever
    nas = tuple(float(v) for v in os.environ["TC_NAS"].split(",")) \
        if os.environ.get("TC_NAS") else (0.004, 0.008, 0.014, 0.020, 0.028, 0.040, 0.056)
    print(f"COVERAGE vs CONVERGENCE  single colour at {16.25} keV  ncry={ncry}  nodes={len(HS)}")
    print(f"  {'NA (mrad)':>10}{'reflections':>13}{'per mrad':>10}{'vs previous':>13}")
    try:
        prev = None
        for na in nas:
            rng = np.random.default_rng(seed)
            m = float(np.median([len(excited(rand_rot(rng), na, [K1])[0]) for _ in range(ncry)]))
            # below a threshold NA the cone reaches no reflection at all, so there is no ratio to
            # take -- print a dash rather than a number divided by zero
            rat = "-" if not prev else f"{m / prev:.2f}"
            print(f"  {na*1e3:>10.0f}{m:>13.0f}{m / (na * 1e3):>10.2f}{rat:>13}", flush=True)
            prev = m if m > 0 else prev
    finally:
        set_beam(e1_0, e2_0)


def run_splitcap(na, ncry=8, seed=3):
    """The split sweep again, scored on CAPTURE RADIUS rather than vote count.

    The contrast tower is the wrong instrument for the small-split end and it took a while to see
    why. Votes are counts of indexed points, and as dE/E -> 0 the second colour's Kossel arc for a
    given reflection slides onto the first colour's, so the point count doubles by DUPLICATION. The
    tower can therefore keep doubling while no new reciprocal-space coverage arrives at all. What
    cannot be faked that way is the width of the basin the refiner has to find: a genuinely better
    objective is one a more distant seed still falls into.
    """
    e1_0, e2_0 = E1_keV, E2_keV
    Ec = 16.25
    fracs = tuple(float(v) for v in os.environ["TC_SPLITS"].split(",")) \
        if os.environ.get("TC_SPLITS") else (0.15385, 0.04, 0.01, 0.002, 0.0)
    perts = [3, 6, 10]
    print(f"SPLIT x CAPTURE  na={na:.3f} rad ({na*1e3:.0f} mrad)  ncry={ncry}  noise=2e-4")
    print(f"  recovery (<1 deg) vs seed perturbation, per config, as the energy split closes")
    hdr = "".join(f"{lab.split(' data ')[0] + '/' + lab.split('/')[-1].strip():>26}"
                  for lab, _, _ in CONFIGS)
    print(f"  {'dE/E':>8}  " + hdr)
    print(f"  {'':>8}  " + "".join(f"{'  '.join(str(p) + 'd' for p in perts):>26}" for _ in CONFIGS))
    try:
        for f in fracs:
            set_beam(Ec * (1 + f / 2), Ec * (1 - f / 2))
            rng0 = np.random.default_rng(seed)
            Rts = [rand_rot(rng0) for _ in range(ncry)]
            cells = []
            for label, Ksim, Kidx in CONFIGS:
                rng = np.random.default_rng(seed + 100)
                rec = {p: 0 for p in perts}
                for Rt in Rts:
                    kobs, lab, col, cents, _ = simulate(Rt, rng, 2e-4, na, Ksim)
                    for p in perts:
                        Rh = refine(kobs, Rt @ small_rot(rng, np.radians(p)), na, Kidx)
                        rec[p] += ang_between(Rt, Rh) < 1.0
                cells.append("  ".join(f"{rec[p]}/{ncry}" for p in perts))
            print(f"  {f:>8.5f}  " + "".join(f"{c:>26}" for c in cells), flush=True)
    finally:
        set_beam(e1_0, e2_0)


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
    elif mode == "split":                                     # contrast across the ENERGY SPLIT
        run_split(na, ncry)
    elif mode == "splitcap":                                  # capture radius across the split
        run_splitcap(na, ncry)
    elif mode == "cover":                                     # reciprocal-space coverage vs split
        run_cover(na, ncry)
    elif mode == "coverna":                                   # coverage vs convergence, 1 colour
        run_coverna(ncry)
    elif mode == "sweep":                                     # contrast across an NA band
        for na in (0.008, 0.012, 0.018, 0.028):
            run_contrast(na, ncry); print()
    else:
        run_blind(na, ncry)
    print(f"[{time.perf_counter()-t0:.1f}s]")
