"""Tests for the powder autoindexer (Engine A: cubic/tetragonal/hexagonal/orthorhombic).

CPU (numpy) tests always run; the cupy parity test skips when cupy is unavailable. Correctness is
planted-cell recovery: build a synthetic line list from a known cell, index it, assert the top solution
recovers the right system + cell and a de Wolff M20 above the reliability mark.
"""
import math
import numpy as np
import pytest

from powder_index import (index_powder, to_inv_d2, cell_to_metric, metric_to_cell, d_spacings,
                          peaks_from_profile, _centering_ok)


# --- synthetic powder line list from a known cell -------------------------------------------------
def sim_lines(cell, centering, nlines=20, imax=10, noise=0.0, seed=0):
    """Distinct d-spacings (A) of the ``nlines`` lowest-angle lattice-allowed reflections."""
    A = cell_to_metric(*cell)
    rng = np.arange(-imax, imax + 1)
    H = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    H = H[np.any(H != 0, 1)]
    H = H[np.asarray(_centering_ok(H, centering))]
    Q = np.einsum("ni,ij,nj->n", H, A, H)
    Qu = np.sort(np.unique(np.round(Q[Q > 0], 6)))[:nlines]
    d = 1.0 / np.sqrt(Qu)
    if noise:
        d = d * (1 + np.random.default_rng(seed).normal(0, noise, d.shape))
    return d


CASES = [   # name, cell, centering, expected system
    ("Si",       (5.4309, 5.4309, 5.4309, 90, 90, 90), "F", "cubic"),
    ("rutile",   (4.5937, 4.5937, 2.9587, 90, 90, 90), "P", "tetragonal"),
    ("zircon",   (6.607, 6.607, 5.982, 90, 90, 90),    "I", "tetragonal"),
    ("quartz",   (4.913, 4.913, 5.405, 90, 90, 120),   "P", "hexagonal"),
    ("forsterite", (4.75, 10.20, 5.98, 90, 90, 90),    "P", "orthorhombic"),
]


def _cell_close(got, truth, rtol=6e-3, atol_ang=1.0):
    # axis labelling is arbitrary for orthorhombic/higher -> compare sorted edges and sorted angles
    ge, gt = sorted(got[:3]), sorted(truth[:3])
    ga, ta = sorted(got[3:]), sorted(truth[3:])
    return all(abs(x - y) <= rtol * max(abs(y), 1.0) for x, y in zip(ge, gt)) and \
        all(abs(x - y) <= atol_ang for x, y in zip(ga, ta))


@pytest.mark.parametrize("name,cell,cen,sysexp", CASES)
def test_recovers_planted_cell(name, cell, cen, sysexp):
    d = sim_lines(cell, cen, nlines=20)
    sols = index_powder(d, units="d", amin=2, amax=40)
    assert sols, name + ": no solution"
    top = sols[0]
    assert top.system == sysexp, "%s: got %s" % (name, top.system)
    assert top.centering == cen, "%s: centering %s" % (name, top.centering)
    assert _cell_close(top.cell, cell), "%s: cell %s vs %s" % (name, top.cell, cell)
    assert top.M20 > 10, "%s: M20=%.1f" % (name, top.M20)
    assert top.nindexed == 20


@pytest.mark.parametrize("name,cell,cen,sysexp", CASES)
def test_recovers_with_noise(name, cell, cen, sysexp):
    d = sim_lines(cell, cen, nlines=20, noise=5e-4, seed=1)      # 0.05% d-spacing scatter
    sols = index_powder(d, units="d", amin=2, amax=40, reltol=0.012)
    assert sols and sols[0].system == sysexp
    assert _cell_close(sols[0].cell, cell, rtol=1.2e-2)


def test_units_roundtrip():
    lam = 1.0
    d = np.array([4.0, 3.0, 2.5, 2.0])
    Q = 1.0 / d ** 2
    assert np.allclose(to_inv_d2(d, units="d"), np.sort(Q))
    q = 2 * math.pi / d                                          # physics q
    assert np.allclose(to_inv_d2(q, units="q"), np.sort(Q))
    tt = np.rad2deg(2 * np.arcsin(lam / (2 * d)))               # 2theta for lambda=1
    assert np.allclose(to_inv_d2(tt, units="twotheta", wavelength=lam), np.sort(Q), rtol=1e-9)


def test_cell_metric_roundtrip():
    for cell in [(5, 5, 5, 90, 90, 90), (4.9, 4.9, 5.4, 90, 90, 120), (5, 7, 9, 90, 90, 90),
                 (7.1, 8.3, 9.7, 90, 103.5, 90)]:
        A = cell_to_metric(*cell)
        got = metric_to_cell(A)[:6]
        assert all(abs(x - y) < 1e-6 for x, y in zip(got, cell))


def test_centering_absences_present():
    # a body-centred cubic list must have (100) absent, (110) present -> indexer picks I not P
    d = sim_lines((6.0, 6.0, 6.0, 90, 90, 90), "I", nlines=18)
    sols = index_powder(d, units="d", amin=2, amax=30)
    assert sols[0].system == "cubic" and sols[0].centering == "I"
    assert _cell_close(sols[0].cell, (6, 6, 6, 90, 90, 90))


def test_peaks_from_profile():
    q = np.linspace(0.5, 6.0, 4000)
    truth = np.array([1.2, 2.1, 2.7, 3.4, 4.6])
    I = np.full_like(q, 10.0) + np.random.default_rng(0).normal(0, 0.3, q.shape)
    for q0 in truth:
        I += 60 * np.exp(-((q - q0) ** 2) / (2 * 0.01 ** 2))
    pk = peaks_from_profile(q, I)
    for q0 in truth:
        assert np.min(np.abs(pk - q0)) < 0.02


def test_profile_to_index_smoke():
    # end-to-end: cell -> d-spacings -> physics-q profile -> peak pick -> index recovers the cell
    cell = (5.64, 5.64, 5.64, 90, 90, 90)                       # NaCl-like, F
    d = sim_lines(cell, "F", nlines=16)
    qpk = 2 * math.pi / d
    q = np.linspace(0.5, qpk.max() * 1.1, 6000)
    I = np.full_like(q, 5.0)
    for q0 in qpk:
        I += 80 * np.exp(-((q - q0) ** 2) / (2 * 0.008 ** 2))
    picked = peaks_from_profile(q, I)
    sols = index_powder(picked, units="q", amin=2, amax=30)
    assert sols and sols[0].system == "cubic" and _cell_close(sols[0].cell, cell, rtol=1e-2)


@pytest.mark.parametrize("cell", [(9.0, 7.0, 11.0, 90, 105, 90), (8.2, 12.9, 7.1, 90, 116, 90)])
def test_recovers_monoclinic(cell):
    # monoclinic goes through seed-and-verify (4 linear metric params, signed reflections) + a-c reduction
    d = sim_lines(cell, "P", nlines=16)
    sols = index_powder(d, units="d", systems=("monoclinic",), centerings=("P",),
                        amin=3, amax=20, seed_lines=5, imax=5, seed_smax=5)
    assert sols and sols[0].system == "monoclinic"
    assert _cell_close(sols[0].cell, cell, rtol=1e-2, atol_ang=1.5)
    assert sols[0].M20 > 10


def _cell_volume(cell):
    a, b, c, al, be, ga = cell
    ca, cb, cg = (math.cos(math.radians(x)) for x in (al, be, ga))
    return a * b * c * math.sqrt(max(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg, 0.0))


def test_recovers_triclinic():
    # triclinic via simulated annealing (directed search). Small/fast config; the lattice is compared by
    # volume + sorted edges (a triclinic cell has many equivalent settings). Stochastic but reliable here.
    cell = (5.5, 6.2, 7.1, 88, 92, 97)
    d = sim_lines(cell, "P", nlines=18)
    sols = index_powder(d, units="d", systems=("triclinic",), amin=3, amax=15,
                        tri_nchain=1500, tri_nstep=150, tri_restarts=2, tri_seed=0)
    assert sols and sols[0].system == "triclinic"
    s = sols[0]
    vtrue = _cell_volume(cell)
    assert abs(s.volume - vtrue) < 0.02 * vtrue, "volume %.1f vs %.1f" % (s.volume, vtrue)
    assert all(abs(x - y) <= 0.1 for x, y in zip(sorted(s.cell[:3]), sorted(cell[:3])))


def test_cupy_matches_numpy():
    cupy = pytest.importorskip("cupy")
    d = sim_lines((4.5937, 4.5937, 2.9587, 90, 90, 90), "P", nlines=20)
    cpu = index_powder(d, units="d", amin=2, amax=40)
    gpu = index_powder(d, units="d", amin=2, amax=40, xp=cupy)
    assert cpu and gpu
    assert cpu[0].system == gpu[0].system == "tetragonal"
    assert _cell_close(gpu[0].cell, cpu[0].cell, rtol=1e-4)
