"""Synthetic validation for the Bravais-constrained prediction refine (DESIGN B).

Self-contained (numpy only, no CrystFEL / nanoBragg).  For each Bravais system:
build a known cell + orientation, generate Ewald-filtered observed rlps with
realistic position noise + outliers, seed BOTH refiners from the SAME perturbed
orientation (and, in scenario S2, a perturbed cell), then compare:

  (1) unconstrained  = glint.index.refine        (free 3x3, 9 DOF)
  (2) constrained    = glint.refine_sym.refine_bravais (Bravais-locked)

Metrics (median over N crystals):
  - orientation error (deg), symmetry-folded over the proper lattice automorphisms
  - cell error (Buerger-invariant reduced-params, % length + |cos angle|)
  - manifold residual (constrained only): max Bravais-constraint violation
  - divergence rate

Pass criteria (see header of each block).  A flat/negative S2 result is a valid
finding; the HARD requirements are (a) non-regression, (c) manifold-to-machine-
precision, (d) no divergence.

  python synth_refine_validate.py
"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from glint.lattice import (cell_to_Ar, Ar_to_Br, cell_params, reduced_params,
                           random_cell, random_rotation)
from glint.index import refine as refine_unc
from glint.refine_sym import refine_bravais

LAMBDA = 1.319          # A (9.4 keV, lyso-ish)
DMIN = 1.6              # A
QMAX = 1.0 / DMIN
TOL_EWALD = 0.006       # 1/A stills excitation gate
SEED_BASE = 20260722

SYSTEMS = ["cubic", "tetragonal", "orthorhombic", "hexagonal",
           "trigonal", "monoclinic", "triclinic"]
NP_FREE = {"cubic": 1, "tetragonal": 2, "orthorhombic": 3, "hexagonal": 2,
           "trigonal": 2, "monoclinic": 4, "triclinic": 6}


# ---------------------------------------------------------------------------
# orientation metric: fold over proper lattice automorphisms P (det=+1, P^T G P = G)
# ---------------------------------------------------------------------------
# precompute all integer 3x3 matrices with entries in {-1,0,1} and det=+1 (once)
def _det_plus1_candidates():
    vals = (-1.0, 0.0, 1.0)
    cand = np.array(list(itertools.product(vals, repeat=9)), float).reshape(-1, 3, 3)
    d = np.linalg.det(cand)
    return cand[np.abs(d - 1.0) < 1e-9]


_CAND = _det_plus1_candidates()   # (~3480, 3, 3)


def proper_automorphisms(Ar):
    """Proper lattice automorphisms P (det=+1, P^T G P = G) -- the crystal's proper
    point-group as integer basis changes.  Batched test over the precomputed candidate set."""
    G = Ar.T @ Ar
    PtGP = np.einsum("nji,jk,nkl->nil", _CAND, G, _CAND)   # P^T G P for all candidates
    resid = np.max(np.abs(PtGP - G), axis=(1, 2))
    keep = resid < 1e-6 * np.max(np.abs(G))
    return _CAND[keep]


def _polar(X):
    U, _, Vt = np.linalg.svd(X)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] = -U[:, -1]; R = U @ Vt
    return R


def geodesic_deg(U1, U2):
    c = (np.trace(U1.T @ U2) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def orient_err_deg(M_rec, U_true, auts):
    U_ref = _polar(M_rec)
    best = 180.0
    for P in auts:
        Uc = _polar(M_rec @ P)
        best = min(best, geodesic_deg(U_true_hat, Uc))
    return best


# ---------------------------------------------------------------------------
# manifold residual of a recovered cell (constrained path)
# ---------------------------------------------------------------------------
def manifold_resid(M, system):
    a, b, c, al, be, ga = cell_params(M)
    if system == "cubic":
        return max(abs(a - b) / a, abs(b - c) / b, abs(al - 90), abs(be - 90), abs(ga - 90))
    if system == "tetragonal":
        return max(abs(a - b) / a, abs(al - 90), abs(be - 90), abs(ga - 90))
    if system == "orthorhombic":
        return max(abs(al - 90), abs(be - 90), abs(ga - 90))
    if system == "hexagonal":
        return max(abs(a - b) / a, abs(al - 90), abs(be - 90), abs(ga - 120))
    if system == "trigonal":
        return max(abs(a - b) / a, abs(b - c) / b, abs(al - be), abs(be - ga))
    if system == "monoclinic":
        return max(abs(al - 90), abs(ga - 90))
    return 0.0


def cell_err(M_rec, M_true):
    l_r, c_r = reduced_params(M_rec)
    l_t, c_t = reduced_params(M_true)
    len_pct = 100.0 * np.sqrt(np.mean(((l_r - l_t) / l_t) ** 2))
    ang_err = np.sqrt(np.mean((c_r - c_t) ** 2))
    return len_pct, ang_err


# ---------------------------------------------------------------------------
# observed-rlp generation
# ---------------------------------------------------------------------------
def gen_observed(rng, B_true, U_true, sigma_g, f_out):
    C = U_true @ B_true                     # reciprocal columns rotated
    norms = np.linalg.norm(C, axis=0)
    H = int(np.ceil(QMAX / norms[0])) + 1
    K = int(np.ceil(QMAX / norms[1])) + 1
    L = int(np.ceil(QMAX / norms[2])) + 1
    h = np.arange(-H, H + 1); k = np.arange(-K, K + 1); l = np.arange(-L, L + 1)
    hkl = np.stack(np.meshgrid(h, k, l, indexing="ij"), -1).reshape(-1, 3).astype(float)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    g = hkl @ C.T
    qn = np.linalg.norm(g, axis=1)
    keep = qn <= QMAX
    hkl, g, qn = hkl[keep], g[keep], qn[keep]
    exc = g[:, 2] + 0.5 * LAMBDA * qn ** 2   # Ewald excitation error
    near = np.abs(exc) < TOL_EWALD
    hkl, g = hkl[near], g[near]
    n = len(g)
    if n == 0:
        return None, None
    g_noisy = g + rng.normal(0, sigma_g, g.shape)
    # outliers: replace a fraction with uniform draws in the qmax ball
    n_out = int(round(f_out * n))
    if n_out:
        idx = rng.choice(n, size=n_out, replace=False)
        v = rng.normal(size=(n_out, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rad = QMAX * rng.random(n_out) ** (1 / 3)
        g_noisy[idx] = v * rad[:, None]
    return g_noisy, hkl


# ---------------------------------------------------------------------------
# per-crystal trial
# ---------------------------------------------------------------------------
def perturb_cell(rng, cell, system):
    a, b, c, al, be, ga = cell
    sl = 1.0 + rng.normal(0, 0.02)
    a2, b2, c2 = a * sl, b * (1 + rng.normal(0, 0.02)), c * (1 + rng.normal(0, 0.02))
    al2, be2, ga2 = al + rng.normal(0, 1.0), be + rng.normal(0, 1.0), ga + rng.normal(0, 1.0)
    return (a2, b2, c2, al2, be2, ga2)


def run_system(system, scenario, delta_theta, sigma_g, n_cry, sys_idx, scen_idx):
    global U_true_hat
    f_out = 0.05
    tol_abs = 4.0 * sigma_g
    oe_unc, oe_con = [], []
    oe_init = []
    ce_unc, ce_con = [], []
    manres = []
    div_unc = div_con = 0
    conv_con = 0
    n_obs_list = []
    n_skip = 0
    ci = 0
    made = 0
    while made < n_cry:
        rng = np.random.default_rng(SEED_BASE + sys_idx * 1_000_000
                                    + scen_idx * 100_000 + ci)
        ci += 1
        if ci > n_cry * 20:
            break
        cell = random_cell(rng, system, lo=30.0, hi=120.0)
        Ar_true = cell_to_Ar(*cell)
        B_true = Ar_to_Br(Ar_true)
        U_true = random_rotation(rng)
        M_true = U_true @ Ar_true
        U_true_hat = _polar(M_true)
        auts = proper_automorphisms(Ar_true)
        g_obs, hkl_true = gen_observed(rng, B_true, U_true, sigma_g, f_out)
        if g_obs is None or len(g_obs) < 25:
            n_skip += 1; continue
        n_obs_list.append(len(g_obs))
        # perturbed orientation (same for both refiners)
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        from glint.refine_sym import _expmap
        U0 = _expmap(np.radians(delta_theta) * axis) @ U_true
        if scenario == "S1":
            Ar0 = Ar_true
        else:
            Ar0 = cell_to_Ar(*perturb_cell(rng, cell, system))
        M0 = U0 @ Ar0

        # (1) unconstrained
        try:
            Mu, Cu, _, inu = refine_unc(g_obs, M0.copy(), tol_abs, n_iter=5)
        except Exception:
            Mu = M0
        # (2) constrained
        try:
            Mc, Cc, _, inc = refine_bravais(g_obs, M0.copy(), system, tol_abs, n_iter=5)
        except Exception:
            Mc = M0

        e0 = orient_err_deg(M0, U_true, auts)      # initial (folded) orientation error
        eu = orient_err_deg(Mu, U_true, auts)
        ec = orient_err_deg(Mc, U_true, auts)
        oe_init.append(e0); oe_unc.append(eu); oe_con.append(ec)
        lu, _ = cell_err(Mu, M_true); lc, _ = cell_err(Mc, M_true)
        ce_unc.append(lu); ce_con.append(lc)
        manres.append(manifold_resid(Mc, system))
        # divergence = made orientation WORSE than the start, or produced a non-physical cell
        detr_u = abs(np.linalg.det(Mu)) / abs(np.linalg.det(M_true))
        detr_c = abs(np.linalg.det(Mc)) / abs(np.linalg.det(M_true))
        if eu > e0 + 1e-3 or not (0.5 <= detr_u <= 2.0):
            div_unc += 1
        if ec > e0 + 1e-3 or not (0.5 <= detr_c <= 2.0):
            div_con += 1
        if ec < 0.1:
            conv_con += 1
        made += 1

    def med(x):
        return float(np.median(x)) if x else float("nan")

    return dict(system=system, scenario=scenario, dtheta=delta_theta, sigma=sigma_g,
                n=made, oe_init=med(oe_init), oe_unc=med(oe_unc), oe_con=med(oe_con),
                ce_unc=med(ce_unc), ce_con=med(ce_con),
                manres=float(np.max(manres)) if manres else float("nan"),
                div_unc=div_unc, div_con=div_con, conv_con=conv_con,
                n_obs=float(np.mean(n_obs_list)) if n_obs_list else 0, n_skip=n_skip)


U_true_hat = np.eye(3)


def main():
    n_cry = int(os.environ.get("NCRY", "60"))
    dtheta = float(os.environ.get("DTHETA", "1.0"))
    sigma = float(os.environ.get("SIGMA", "0.0015"))
    scenarios = os.environ.get("SCEN", "S1,S2").split(",")
    print(f"# Bravais-constrained refine synthetic validation")
    print(f"# seed_base={SEED_BASE} n_cry={n_cry} delta_theta={dtheta} deg "
          f"sigma_g={sigma} 1/A  lambda={LAMBDA} dmin={DMIN}")
    print(f"# {'system':>13} {'scen':>4} {'np':>3} {'oe_init':>8} {'oe_unc':>8} {'oe_con':>8} "
          f"{'ce_unc%':>8} {'ce_con%':>8} {'manres':>10} {'divU':>5} {'divC':>5} "
          f"{'conv':>5} {'Nobs':>6}")
    rows = []
    for si, system in enumerate(SYSTEMS):
        for ci, scen in enumerate(scenarios):
            r = run_system(system, scen, dtheta, sigma, n_cry, si, ci)
            rows.append(r)
            print(f"  {r['system']:>13} {r['scenario']:>4} {NP_FREE[system]:>3} "
                  f"{r['oe_init']:>8.4f} {r['oe_unc']:>8.4f} {r['oe_con']:>8.4f} {r['ce_unc']:>8.3f} "
                  f"{r['ce_con']:>8.3f} {r['manres']:>10.2e} {r['div_unc']:>5} "
                  f"{r['div_con']:>5} {r['conv_con']:>4}/{r['n']:<2} {r['n_obs']:>6.0f}", flush=True)

    # ---- pass criteria (nominal point) ----
    print("\n# PASS CRITERIA")
    ok = True
    max_manres = max(r["manres"] for r in rows if not np.isnan(r["manres"]))
    print(f"  (c) manifold-to-machine-precision: max manres = {max_manres:.2e}  "
          f"-> {'PASS' if max_manres < 1e-6 else 'FAIL'}")
    ok &= max_manres < 1e-6
    # (a) non-regression on orientation for every system/scenario
    reg_fail = [(r["system"], r["scenario"]) for r in rows
                if r["oe_con"] > r["oe_unc"] + 1e-3]
    print(f"  (a) non-regression (con <= unc + 1e-3 deg): "
          f"{'PASS' if not reg_fail else 'FAIL ' + str(reg_fail)}")
    ok &= not reg_fail
    # (d) no divergence: constrained's divergence rate is not systematically worse than unconstrained.
    # "divergence" = orientation ended WORSE than the initial kick OR the cell went non-physical
    # (det ratio outside [0.5,2]).  A tail tolerance of max(2, 5% of n) absorbs Poisson noise at the
    # sub-0.02-deg convergence floor; a real divergence (cell collapse) would blow past it.
    def dtol(n):
        return max(2, int(0.05 * n))
    div_fail = [(r["system"], r["scenario"], r["div_con"], r["div_unc"], r["n"]) for r in rows
                if r["n"] and r["div_con"] > r["div_unc"] + dtol(r["n"])]
    print(f"  (d) no divergence (div_con <= div_unc + tol): "
          f"{'PASS' if not div_fail else 'FAIL ' + str(div_fail)}")
    ok &= not div_fail
    # (b) S2 improvement for constrained systems (diagnostic, not hard-fail)
    print("  (b) S2 improvement (con <= 0.8*unc) [diagnostic]:")
    for r in rows:
        if r["scenario"] == "S2" and NP_FREE[r["system"]] < 6:
            imp = r["oe_con"] <= 0.8 * r["oe_unc"]
            print(f"        {r['system']:>13}: con {r['oe_con']:.4f} vs unc {r['oe_unc']:.4f} "
                  f"-> {'improved' if imp else 'flat/worse'}")
    print(f"\n# OVERALL (hard criteria a,c,d): {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
