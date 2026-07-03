"""AE-seeded classical indexing: does the learned head actually help the exact solver?

The autoencoder head is fast but COARSE (system 7-way + ~2 A cell). The classical `index_powder` is exact
but must (a) try every crystal system and (b) search a wide edge range -> slow, and prone to spurious
cross-system supercells. Here the AE acts as a PRIOR: its predicted system restricts `systems=`, and its
predicted edges set a tight `amin/amax` window. We then measure, on the SAME held-out patterns, exact-cell
recovery + wall-time for:

  cold   : index_powder(peaks, systems=4 high-sym, amin=2, amax=20)            -- no prior
  hybrid : index_powder(peaks, systems=(AE system,), amin/amax around AE cell) -- AE-seeded

A cell is "recovered" if the top solution's 3 edges match truth within EDGE_TOL (sorted, A). Reports
recovery + ms for both, plus the AE system/edge accuracy that drives the prior. Run after train.py (needs ae.pt).
CPU-only (index_powder is a CPU tool); AE prior runs on CPU here too.
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_dataset import make_dataset, denorm_cell, QGRID, SYSTEMS  # noqa: E402
from model import PowderAE  # noqa: E402
from powder_index import index_powder, peaks_from_profile, _HIGHSYM  # noqa: E402

# The cold baseline searches the 4 high-symmetry systems (cubic/tet/hex/ortho). Monoclinic seed-and-verify
# over dense simulated profiles is pathologically slow (minutes/pattern), so it's out of this timed head-to-head.

DEV = "cpu"                      # index_powder is CPU; keep the AE on CPU for a fair single-machine timing
EDGE_TOL = 0.5                   # A: |sorted recovered edge - sorted true edge| <= EDGE_TOL for all 3 -> hit
EDGE_MARGIN = 3.0                # A: AE edge window half-width (covers the ~2 A head MAE)


def load():
    net = PowderAE().to(DEV)
    net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "ae.pt"), map_location=DEV)["state"])
    net.eval(); return net


def edges_match(sol, cell):
    got = np.sort([sol.a, sol.b, sol.c])
    tru = np.sort(np.asarray(cell, float)[:3])
    return bool(np.all(np.abs(got - tru) <= EDGE_TOL))


def main():
    n_eval = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    net = load()
    X, cells, sysid, cens = make_dataset(1500, seed=707)
    # eval on patterns whose TRUE system the high-sym seeder solves fast (cubic/tet/hex/ortho)
    sel = [i for i in range(len(X)) if SYSTEMS[sysid[i]] in _HIGHSYM][:n_eval]
    print("device:", DEV, "n_eval:", len(sel), "EDGE_TOL:", EDGE_TOL, "A  EDGE_MARGIN:", EDGE_MARGIN, "A")

    cold_ok = hyb_ok = hedge_ok = sys_ok = 0
    t_cold = t_hyb = t_hedge = 0.0
    edge_mae = 0.0
    for i in sel:
        prof = X[i]
        true_sys = SYSTEMS[sysid[i]]
        # --- AE prior (system + coarse cell) ---
        with torch.no_grad():
            _, cp, sl, _ = net(torch.tensor(prof).view(1, 1, -1).to(DEV))
        ae_sys = SYSTEMS[int(sl.argmax(1))]
        ae_cell = denorm_cell(cp.cpu().numpy()[0])
        ae_edges = ae_cell[:3]
        sys_ok += int(ae_sys == true_sys)
        edge_mae += float(np.mean(np.abs(np.sort(ae_edges) - np.sort(np.asarray(cells[i], float)[:3]))))
        # AE-predicted system only if the seeder supports it; else fall back to all seedable systems
        hyb_systems = (ae_sys,) if ae_sys in _HIGHSYM else _HIGHSYM
        lo = max(2.0, float(ae_edges.min()) - EDGE_MARGIN)
        hi = float(ae_edges.max()) + EDGE_MARGIN

        pk = peaks_from_profile(QGRID, prof, max_peaks=25)
        if len(pk) < 4:
            continue

        # --- cold: no prior ---
        t = time.perf_counter()
        cold = index_powder(pk, units="q", systems=_HIGHSYM, amin=2.0, amax=20.0)
        t_cold += time.perf_counter() - t
        cold_ok += int(bool(cold) and edges_match(cold[0], cells[i]))

        # --- hybrid (system+edge): AE system restriction + tight edge window ---
        t = time.perf_counter()
        hyb = index_powder(pk, units="q", systems=hyb_systems, amin=lo, amax=hi)
        t_hyb += time.perf_counter() - t
        hyb_ok += int(bool(hyb) and edges_match(hyb[0], cells[i]))

        # --- hybrid (edge-only): keep all 4 systems, apply only the RELIABLE AE edge window ---
        t = time.perf_counter()
        hedge = index_powder(pk, units="q", systems=_HIGHSYM, amin=lo, amax=hi)
        t_hedge += time.perf_counter() - t
        hedge_ok += int(bool(hedge) and edges_match(hedge[0], cells[i]))

    n = len(sel)
    print("\n=== AE prior quality (drives the hybrid) ===")
    print("  system acc = %.2f   edge MAE = %.2f A" % (sys_ok / n, edge_mae / n))
    print("\n=== exact-cell recovery + wall-time (top solution, EDGE_TOL=%.1f A) ===" % EDGE_TOL)
    print("  cold      (4 high-sym, amin2-amax20):      %.0f%%   %.0f ms/pattern" % (100 * cold_ok / n, 1e3 * t_cold / n))
    print("  hyb sys+edge (AE system + AE edge window):  %.0f%%   %.0f ms/pattern" % (100 * hyb_ok / n, 1e3 * t_hyb / n))
    print("  hyb edge-only (all systems + AE edge window):%.0f%%   %.0f ms/pattern" % (100 * hedge_ok / n, 1e3 * t_hedge / n))
    if t_hyb > 0:
        print("\n  vs cold:  sys+edge  %.1fx, %+d pts   |   edge-only  %.1fx, %+d pts" %
              (t_cold / t_hyb, round(100 * (hyb_ok - cold_ok) / n),
               t_cold / t_hedge, round(100 * (hedge_ok - cold_ok) / n)))
    print("DONE")


if __name__ == "__main__":
    main()
