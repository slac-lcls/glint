"""Scoring against ground truth.

A high "fraction of spots indexed" is NOT sufficient: a spurious finer lattice
(sub-cell) sits within tolerance of most spots too. So correctness is judged by
a rigorous *lattice match*: the recovered basis M must relate to the true rotated
basis shot.M by an integer, unimodular transform

    T = M_true^{-1} M_rec   ->   integer entries and |det T| = 1

(both are in the lab frame, so this is rotation-correct). A sub-cell gives
|det T| != 1 and is rejected. When the lattice matches, we align the recovered
basis through T and report the grid-free per-axis error.
"""

from __future__ import annotations

import numpy as np


def score(shot, result, tol_frac=0.02, int_tol=0.2, det_tol=0.15):
    g = shot.g
    true = ~shot.is_spurious
    n_true = int(true.sum())
    out = dict(n_true=n_true, n_indexed=0, frac_indexed=0.0,
               lattice_match=False, solved=False, axis_err=float("nan"),
               no_index=True, wrong_index=False)
    if result is None or result.C is None:
        return out                                  # no_index stays True
    out["no_index"] = False

    C = result.C
    Cinv = np.linalg.inv(C)
    hkl = np.rint(g @ Cinv.T)
    resid = np.linalg.norm(g - hkl @ C.T, axis=1)
    tol = tol_frac * shot.meta["qmax"]          # absolute, matches the indexer
    indexed = (resid < tol) & true
    out["n_indexed"] = int(indexed.sum())
    out["frac_indexed"] = indexed.sum() / max(n_true, 1)

    # rigorous lattice-equality test
    T = np.linalg.inv(shot.M) @ result.M
    intT = np.rint(T)
    is_integer = np.max(np.abs(T - intT)) < int_tol
    is_unimodular = abs(abs(np.linalg.det(T)) - 1.0) < det_tol
    out["lattice_match"] = bool(is_integer and is_unimodular)
    out["solved"] = bool(out["frac_indexed"] >= 0.7 and out["lattice_match"])
    # returned a confident solution that is the WRONG lattice (the dangerous case)
    out["wrong_index"] = bool(out["frac_indexed"] >= 0.7 and not out["lattice_match"])

    if out["lattice_match"] and abs(np.linalg.det(intT)) > 0:
        # align recovered basis through T and measure true per-axis error (grid-free)
        Ma = result.M @ np.linalg.inv(intT)
        out["axis_err"] = float(np.max(np.linalg.norm(Ma - shot.M, axis=0)))
    return out
