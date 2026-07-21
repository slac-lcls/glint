"""#33 fix, exercised on REAL data end to end (cf_peaks.cxi + cf.geom).

test_top_peaks_stored.py proves the truncation logic on a synthetic fixture. This closes the other
half: that the fix works through the real `--images` entry point on a real Cheetah/peakfinder8 file,
and that the capped frames still INDEX -- a truncation that silently produced unusable frames would
pass every unit check in the synthetic test.

  python verify_top_peaks_real.py
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from glint.lute_bridge import frames_from_cxi
import glint.glint_fast as gf
from glint.multishot import same_lattice

CXI = os.path.join(HERE, "cf_peaks.cxi")
GEOM = os.path.join(HERE, "cf.geom")
LYSO = gf.LYSO
for p in (CXI, GEOM):
    if not os.path.exists(p):
        print(f"missing {p}"); sys.exit(1)

with h5py.File(CXI, "r") as f:
    npk = np.asarray(f["/entry_1/result_1/nPeaks"])
    has_int = "/entry_1/result_1/peakTotalIntensity" in f
print(f"{CXI}: {len(npk)} frames, peaks/frame median {int(np.median(npk))} max {int(npk.max())}, "
      f"peakTotalIntensity present: {has_int}\n")


def gate(M, q):
    if M is None or not same_lattice(np.asarray(M, float), LYSO):
        return 0
    r = np.asarray(q) @ M
    m = int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())
    return int(m / len(q) >= 0.25 and m >= 10)


print(f"{'top_n':>7}{'frames':>8}{'peaks med':>11}{'peaks max':>11}{'same_lat':>10}{'gate':>8}{'ms/frame':>11}")
base_counts = None
for top_n in (0, 200, 100, 50):
    fr, im = frames_from_cxi(CXI, GEOM, peakfinder="stored", top_n=top_n, min_peaks=6)
    fr = [q for q in fr if len(q) >= 6]
    counts = [len(q) for q in fr]
    if base_counts is None:
        base_counts = counts
    gf.index_blind_fast(fr[0])
    t0 = time.perf_counter()
    Ms = [gf.index_blind_fast(q) for q in fr]
    ms = 1e3 * (time.perf_counter() - t0) / len(fr)
    lat = sum(int(M is not None and same_lattice(np.asarray(M, float), LYSO)) for M in Ms)
    g = sum(gate(M, q) for M, q in zip(Ms, fr))
    print(f"{top_n if top_n else '-':>7}{len(fr):>8}{int(np.median(counts)):>11}{max(counts):>11}"
          f"{f'{lat}/{len(fr)}':>10}{f'{g}/{len(fr)}':>8}{ms:11.2f}")
    if top_n:
        assert max(counts) <= top_n, f"*** top_n={top_n} did NOT cap: max {max(counts)}"

print("\nchecks")
print(f"  [{'PASS' if max(base_counts) == int(npk.max()) else 'FAIL'}] top_n=0 leaves the full peak "
      f"list (max {max(base_counts)} == nPeaks max {int(npk.max())})")
fr100, _ = frames_from_cxi(CXI, GEOM, peakfinder="stored", top_n=100, min_peaks=6)
c100 = [len(q) for q in fr100 if len(q) >= 6]
print(f"  [{'PASS' if max(c100) <= 100 else 'FAIL'}] top_n=100 caps every frame (max {max(c100)})")
untouched = sum(1 for a, b in zip(base_counts, c100) if a <= 100 and a == b)
n_small = sum(1 for a in base_counts if a <= 100)
print(f"  [{'PASS' if untouched == n_small else 'FAIL'}] frames already under the cap are unchanged "
      f"({untouched}/{n_small})")
print("\nA cap that produced unusable frames would still pass the synthetic unit test; the rate")
print("columns above are what rule that out.")
