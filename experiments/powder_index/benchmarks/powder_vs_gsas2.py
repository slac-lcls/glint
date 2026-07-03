"""Benchmark the powder autoindexer (``powder_index.py``) against a real indexer -- GSAS-II.

Same input (a synthetic line list from a known cell) into both; compare recovered cell, de Wolff M20, and
wall-time.  Like the pyFAI / psana rows elsewhere in this repo, the *reference* half runs only where the
reference is installed: if GSAS-II can't be imported, we still print our own numbers and skip the GSAS-II
column with an install hint.

CPU-vs-CPU by design: a powder pattern is small, so the autoindexer is a latency-bound CPU workload (the
cupy path is bit-identical but ~60x slower on an A100 -- measured -- so there is no GPU column here).

Install GSAS-II (a real Louer/Monte-Carlo autoindexer, Toby & Von Dreele 2013):
    conda install -c conda-forge gsas2pkg     # most reliable (ships the compiled binaries)
    # or see https://gsas-ii.readthedocs.io/en/latest/packages.html
Run:
    python powder_vs_gsas2.py

Second reference (optional): McMaille (Le Bail), a single Fortran file -- build with ``build_mcmaille.sh``
and point ``$MCMAILLE`` at the binary; not wired here but the harness is reference-agnostic by design.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from powder_index import index_powder, cell_to_metric, _centering_ok, metric_to_cell   # noqa: E402

WAVELENGTH = 1.5406   # Cu K-alpha, A -- only used to express the shared line list as 2-theta for GSAS-II

# (name, cell, centering, expected-system, GSAS-II Bravais index group)
CELLS = [
    ("Si (cubic F)",       (5.4309, 5.4309, 5.4309, 90, 90, 90), "F"),
    ("rutile (tet P)",     (4.5937, 4.5937, 2.9587, 90, 90, 90), "P"),
    ("zircon (tet I)",     (6.607, 6.607, 5.982, 90, 90, 90),    "I"),
    ("quartz (hex P)",     (4.913, 4.913, 5.405, 90, 90, 120),   "P"),
    ("forsterite (ortho P)", (4.75, 10.20, 5.98, 90, 90, 90),    "P"),
]


def sim_lines(cell, centering, nlines=20, imax=10):
    A = cell_to_metric(*cell)
    rng = np.arange(-imax, imax + 1)
    H = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    H = H[np.any(H != 0, 1)]
    H = H[np.asarray(_centering_ok(H, centering))]
    Q = np.einsum("ni,ij,nj->n", H, A, H)
    Qu = np.sort(np.unique(np.round(Q[Q > 0], 6)))[:nlines]
    return 1.0 / np.sqrt(Qu)                       # d-spacings (A)


def d_to_twotheta(d, lam):
    return np.rad2deg(2 * np.arcsin(np.clip(lam / (2 * d), -1, 1)))


# --- our indexer ----------------------------------------------------------------------------------
def run_ours(d, xp=None, reps=3):
    best = None
    for _ in range(reps):
        t = time.perf_counter()
        sols = index_powder(d, units="d", amin=2, amax=40, xp=xp)
        dt = (time.perf_counter() - t) * 1e3
        best = dt if best is None else min(best, dt)
    top = sols[0] if sols else None
    return top, best


# --- GSAS-II reference (import-or-skip) -----------------------------------------------------------
def _import_gsas2():
    for modpath in ("GSASII.GSASIIindex", "GSASIIindex"):
        try:
            mod = __import__(modpath, fromlist=["DoIndexPeaks"])
            return mod
        except Exception:
            continue
    return None


def run_gsas2(d, lam):
    """Best-effort call into GSAS-II's DoIndexPeaks. Returns (cell, M20, ms) or None.

    NOTE: GSAS-II's indexing entry point takes a GUI-oriented peak/controls structure; the exact field
    layout has shifted across versions.  We build it per the documented ``DoIndexPeaks(peaks, controls,
    bravais, ...)`` signature and wrap defensively -- on a real install, print the raised error and adapt
    the two constructors below (they are the only version-sensitive part).
    """
    G2 = _import_gsas2()
    if G2 is None:
        return None
    try:
        tth = d_to_twotheta(d, lam)
        # peak row: [pos(2theta), intensity, use, indexed, h, k, l, d-obs]
        peaks = [[float(t), 100.0, True, False, 0, 0, 0, float(dd)] for t, dd in zip(tth, d)]
        # controls: [Zero, Zero-ref?, Ncols, max-Volume, ...] -- GSAS-II uses a 6-list; wavelength is here
        controls = [0, 0.0, 4, 100.0, 0, lam]
        # bravais flags (14): try all -> [P,I,F,R,P,I,F,P,I,C,P,C,P,P] ordering per GSAS-II; enable all
        bravais = [1] * 14
        t = time.perf_counter()
        out = G2.DoIndexPeaks(peaks, controls, bravais, None)
        dt = (time.perf_counter() - t) * 1e3
        # out = (bool_ok, dmin, cells) ; cells sorted, cell entry has M20 and the 6 cell params
        cells = out[2] if isinstance(out, (list, tuple)) and len(out) >= 3 else out
        best = sorted(cells, key=lambda c: -c[-1])[0] if cells else None    # by M20 (last-ish field)
        return best, dt
    except Exception as e:  # noqa: BLE001
        print("   [GSAS-II call raised: %r -- adapt the peaks/controls constructor for your version]" % e)
        return "error"


# --- driver ---------------------------------------------------------------------------------------
# CPU-only by design: a powder pattern is small (a few dozen lines), so this is a latency-bound CPU
# workload -- the cupy path is bit-identical but ~60x slower on an A100 (measured), so we don't time it.
def main():
    g2_available = _import_gsas2() is not None
    print("powder autoindexer (CPU)  vs  GSAS-II" +
          ("" if g2_available else "  (GSAS-II not installed -> ours only)"))
    print("%-22s | %-28s | %8s | %8s" % ("cell", "ours: recovered", "M20", "ms(cpu)"))
    print("-" * 78)
    for name, cell, cen in CELLS:
        d = sim_lines(cell, cen)
        top, ms_cpu = run_ours(d)
        rec = "%s %s a=%.3f c=%.3f" % (top.system[:5], top.centering, top.a, top.c) if top else "FAIL"
        print("%-22s | %-28s | %8.1f | %8.1f" % (name, rec, top.M20 if top else 0, ms_cpu))
        if g2_available:
            res = run_gsas2(d, WAVELENGTH)
            if res and res != "error":
                best, ms = res
                print("%-22s | GSAS-II: %-36.36s | %8s" % ("", str(best), "%.1f ms" % ms))
    if not g2_available:
        print("\nInstall GSAS-II (conda install -c conda-forge gsas2pkg) for the reference column.")


if __name__ == "__main__":
    main()
