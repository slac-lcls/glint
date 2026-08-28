"""Controlled synthetic SFX generator with KNOWN truth I_full (item 3 payoff).

Generates a multi-crystal stills dataset of a known cell where each observed
reflection is a partial ``I_obs = G_c * p_true * I_full * (1+noise)`` -- so the
FULL intensities ``I_full(hkl)`` are known and the merge can be scored against
them (``R_vs_truth``), which CC1/2 cannot be (it can be gamed by
consistent-but-biased partials).  No detector images are rendered: the generator
emits reflection TABLES (true hkl, noisy observed q, I_obs, sigma) that feed the
same asu-key / MergeAccumulator path GLINT uses on real data, so the three merge
conditions differ ONLY in geometry refinement and the scale/partiality model.

Per still: random orientation, per-crystal mosaicity eta_c, bandwidth (dl/l)_c,
scale G_c (and optional B_c); reflections gated by |excitation error| < eps_max;
partiality is the TRUE Gaussian rocking curve.  A slightly-wrong (perturbed +
symmetry-broken) orientation/cell is also emitted so the Bravais refine (item 2)
has something to fix.  Everything is seeded and reproducible.
"""
from __future__ import annotations

import numpy as np

from glint.lattice import cell_to_Ar, random_rotation
from glint.predict import recip_from_M, _hkl_grid
from glint.stream_driver import laue_ops_4mmm, _asu_key


def sigma_p_true(q_mag, eta, dloverl, wavelength_A):
    """TRUE rocking-curve width: mosaic (~|q|) + bandwidth (~|q|^2) in quadrature."""
    t2 = 0.5 * eta * q_mag
    t3 = 0.5 * wavelength_A * q_mag * q_mag * dloverl
    return np.sqrt(t2 * t2 + t3 * t3)


class SynthConfig:
    def __init__(self,
                 cell=(79.2, 79.2, 38.2, 90.0, 90.0, 90.0),
                 wavelength_A=1.0, dmin=2.0, dmax=30.0,
                 N=2000, seed=0, truth_seed=12345,
                 mu_eta_deg=0.10, sig_eta_deg=0.05, eta_clip_deg=(0.02, 0.40),
                 dloverl_ln_mean=np.log(2e-3), dloverl_ln_sd=0.30, dloverl_clip=(5e-4, 6e-3),
                 G_ln_sd=0.35, B_c_sd=4.0,
                 I0=1e4, B_wilson=25.0,
                 kappa=0.05, bg0=20.0,
                 q_noise=8e-4,
                 pert_rot_deg=0.15, pert_cell_sd=0.003, pert_angle_deg=0.15,
                 use_B_c=False, eps_max_scale=3.0,
                 noiseless=False, unit_partiality=False, no_perturb=False, unit_scale=False):
        self.__dict__.update(locals()); del self.__dict__["self"]


def build_truth(cfg):
    """Unique-ASU reflection list + KNOWN I_full(hkl) (Wilson) to dmin. Returns
    (ops, keys, I_full_by_key, qmag_by_key)."""
    a, b, c, al, be, ga = cfg.cell
    Ar = cell_to_Ar(a, b, c, al, be, ga)
    R = recip_from_M(Ar)                 # reciprocal rows a*,b*,c* for the ALIGNED cell
    qmax = 1.0 / cfg.dmin
    qmin = 1.0 / cfg.dmax
    hkl, q = _hkl_grid(R, qmax)
    qn = np.linalg.norm(q, axis=1)
    keep = (qn <= qmax) & (qn >= qmin)
    hkl, qn = hkl[keep], qn[keep]
    ops = laue_ops_4mmm()
    keys = _asu_key(hkl, ops)
    ukeys, idx = np.unique(keys, return_index=True)
    qmag_u = qn[idx]
    s2 = (qmag_u / 2.0) ** 2
    rng = np.random.default_rng(cfg.truth_seed)
    W = rng.exponential(1.0, size=len(ukeys))          # Wilson: Exp(1) per unique reflection
    I_full = cfg.I0 * W * np.exp(-2.0 * cfg.B_wilson * s2)
    return ops, ukeys, dict(zip(ukeys.tolist(), I_full.tolist())), dict(zip(ukeys.tolist(), qmag_u.tolist()))


def _perturb_M(M_true, cfg, rng):
    """A misoriented + symmetry-broken cell version of M_true (item-2's job to fix)."""
    a, b, c, al, be, ga = cfg.cell
    d = rng.normal(0.0, cfg.pert_cell_sd, 3)
    a_p = a * (1 + d[0]); b_p = a * (1 + d[1]); c_p = c * (1 + d[2])   # break a=b
    al_p = 90.0 + rng.normal(0.0, cfg.pert_angle_deg)
    be_p = 90.0 + rng.normal(0.0, cfg.pert_angle_deg)
    ga_p = 90.0 + rng.normal(0.0, cfg.pert_angle_deg)
    Ar_p = cell_to_Ar(a_p, b_p, c_p, al_p, be_p, ga_p)
    # small extra misorientation
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    ang = np.deg2rad(abs(rng.normal(0.0, cfg.pert_rot_deg)))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    dR = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    # recover the true rotation from M_true (columns = R @ real axes)
    Ar_true = cell_to_Ar(a, b, c, al, be, ga)
    R_true = M_true @ np.linalg.inv(Ar_true)
    return dR @ R_true @ Ar_p


def generate(cfg):
    """Generate the dataset. Returns (truth, stills) where truth=(ops,ukeys,I_full,qmag)
    and stills is a list of dicts with M_true, M_pert, hkl_true, q_obs, I_obs, sigma."""
    ops, ukeys, I_full, qmag_by_key = build_truth(cfg)
    a, b, c, al, be, ga = cfg.cell
    Ar = cell_to_Ar(a, b, c, al, be, ga)
    lam = cfg.wavelength_A
    R_ref = recip_from_M(Ar)
    qmax = 1.0 / cfg.dmin
    qmin = 1.0 / cfg.dmax
    # candidate hkl grid (orientation invariant set of |q| in range)
    hkl_grid, _ = _hkl_grid(R_ref, qmax)

    # typical rocking width -> gate half-width
    eta_typ = np.deg2rad(cfg.mu_eta_deg)
    dl_typ = np.exp(cfg.dloverl_ln_mean)
    q_typ = 1.0 / 3.0
    eps_max = cfg.eps_max_scale * sigma_p_true(q_typ, eta_typ, dl_typ, lam)

    rng = np.random.default_rng(cfg.seed)
    stills = []
    for c_i in range(cfg.N):
        crng = np.random.default_rng([cfg.seed, c_i])
        Rrot = random_rotation(crng)
        M_true = Rrot @ Ar
        Rc = recip_from_M(M_true)                       # inv(M_true) rows
        q = hkl_grid @ Rc
        qn2 = np.einsum("ij,ij->i", q, q)
        qn = np.sqrt(qn2)
        inres = (qn <= qmax) & (qn >= qmin)
        eps = q[:, 2] + 0.5 * lam * qn2

        # per-crystal parameters (drawn BEFORE gating: the rocking width sets BOTH which
        # reflections are in diffracting condition AND their partiality -- so the gate is
        # per-reflection at |eps| < scale*sigma_p, which bounds p_true >= exp(-scale^2/2))
        eta = np.clip(abs(crng.normal(cfg.mu_eta_deg, cfg.sig_eta_deg)),
                      cfg.eta_clip_deg[0], cfg.eta_clip_deg[1])
        eta = np.deg2rad(eta)
        dl = float(np.clip(np.exp(crng.normal(cfg.dloverl_ln_mean, cfg.dloverl_ln_sd)),
                           cfg.dloverl_clip[0], cfg.dloverl_clip[1]))
        G_c = 1.0 if cfg.unit_scale else float(np.exp(crng.normal(0.0, cfg.G_ln_sd)))
        B_c = float(crng.normal(0.0, cfg.B_c_sd)) if cfg.use_B_c else 0.0

        sp_all = sigma_p_true(qn, eta, dl, lam)
        obs = inres & (np.abs(eps) < cfg.eps_max_scale * sp_all)
        if obs.sum() < 8:
            continue
        hkl_o = hkl_grid[obs]; q_o = q[obs]; qn_o = qn[obs]; eps_o = eps[obs]

        # TRUE partiality (bounded below by exp(-eps_max_scale^2/2) by the per-refl gate)
        sp = sigma_p_true(qn_o, eta, dl, lam)
        p_true = np.exp(-eps_o ** 2 / (2.0 * np.maximum(sp, 1e-9) ** 2))
        if cfg.unit_partiality:
            p_true = np.ones_like(p_true)

        # truth I_full for these reflections (via asu key)
        keys_o = _asu_key(hkl_o, ops)
        Ifull_o = np.array([I_full.get(int(k), 0.0) for k in keys_o])
        good = Ifull_o > 0
        if good.sum() < 8:
            continue
        hkl_o, q_o, qn_o, p_true, Ifull_o, keys_o = (hkl_o[good], q_o[good], qn_o[good],
                                                     p_true[good], Ifull_o[good], keys_o[good])
        s2 = (qn_o / 2.0) ** 2

        # observed intensity: photon counting + background
        mu = G_c * np.exp(-2.0 * B_c * s2) * p_true * Ifull_o
        if cfg.noiseless:
            I_obs = mu.copy()
            sigma = np.ones_like(mu)
        else:
            kap = cfg.kappa
            Nph = crng.poisson(np.maximum(kap * mu, 0.0)) + crng.poisson(kap * cfg.bg0, size=len(mu))
            I_obs = (Nph - kap * cfg.bg0) / kap
            sigma = np.sqrt(np.maximum(Nph + kap * cfg.bg0, 1.0)) / kap

        # noisy observed reciprocal peaks (detector localisation error) for the refine/index
        if cfg.no_perturb:
            q_obs = q_o.copy()
            M_pert = M_true.copy()
        else:
            q_obs = q_o + crng.normal(0.0, cfg.q_noise, size=q_o.shape)
            M_pert = _perturb_M(M_true, cfg, crng)
        stills.append(dict(M_true=M_true, M_pert=M_pert, hkl_true=hkl_o.astype(int),
                           q_obs=q_obs, I_obs=I_obs, sigma=sigma,
                           eta=eta, dloverl=dl, G_c=G_c, B_c=B_c, p_true=p_true))
    return (ops, ukeys, I_full, qmag_by_key), stills
