"""#28 lever 1: shrink the M1 start grid, and price it end-to-end.

M3 is 66% of the blind front end and linear in `starts`, so the grid is the biggest work term
available. ROADMAP.md:52 already established the ACCURACY side -- blind degrades as the grid thins
(n_dir 2200->700 = 84->65 of 120) but consensus + rescue absorb it, hybrid holding 115-118/120 --
and concluded "not a wall-time win" from the blind-standalone latency (19.2->13.8 ms).

That conclusion is worth re-testing against a stage breakdown that did not exist then: if M3 is
66% of the frame and linear in starts, a 3x smaller grid should move the front end a lot more than
those numbers suggest. So this measures BOTH halves at current main:

  blind    same_lattice rate + ms/frame        (degrades, expected)
  hybrid   gated rate + ms/frame + n_resc      (the end-to-end number that actually ships)

The honest gate is the hybrid rate: the grid may only be shrunk as far as consensus + rescue can
cover, and n_resc shows how hard the rescue is working to hide it.

  python blind_startgrid_sweep.py [frames.txt]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import glint.glint_fast as gf
import glint.glint_index as gi
from glint.glint_index import sample
from glint.hybrid_stream import hybrid_index
from glint.multishot import same_lattice

FR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "frames_cxidb_clean.txt")
frames = [q for q in gf.load(FR) if len(q) >= 6]
n = len(frames)
LYSO = gf.LYSO
BASE = gf.STARTS


def gate(M, q):
    """Table-1 gate: correct lattice AND >=25% of spots AND >=10 reflections."""
    if M is None or not same_lattice(np.asarray(M, float), LYSO):
        return 0
    r = np.asarray(q) @ M
    m = int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())
    return int(m / len(q) >= 0.25 and m >= 10)


def sync():
    if gf.DEV == "cuda":
        torch.cuda.synchronize()


print(f"start-grid sweep, {n} cxidb frames, {gf.DEV}, STEPS={gf.STEPS}")
print(f"baseline grid {tuple(BASE.shape)} = 32 length shells x n_dir 2200\n")
print(f"{'n_dir':>7}{'starts':>9} | {'blind lat':>10}{'blind ms':>10} | "
      f"{'hybrid gate':>12}{'hybrid ms':>11}{'n_resc':>8}{'speedup':>9}")

base_ms = None
for n_dir in (2200, 1600, 1100, 700, 500, 350):
    S = sample(n_dir=n_dir).to(gf.DEV)
    gf.STARTS = S; gi.STARTS = S

    gf.index_blind_fast(frames[0]); sync()                    # ---- blind alone
    t0 = time.perf_counter()
    Ms = [gf.index_blind_fast(q) for q in frames]
    sync(); bms = 1e3 * (time.perf_counter() - t0) / n
    blat = sum(int(M is not None and same_lattice(np.asarray(M, float), LYSO)) for M in Ms)

    t0 = time.perf_counter()                                  # ---- hybrid (what ships)
    res, st = hybrid_index(frames, None, Mc_known=None, nbest=3, warmup=True)
    sync(); hms = 1e3 * (time.perf_counter() - t0) / n
    hgate = sum(gate(r["M"], q) for r, q in zip(res, frames))
    if base_ms is None:
        base_ms = hms
    print(f"{n_dir:>7}{int(S.shape[0]):>9} | {f'{blat}/{n}':>10}{bms:10.2f} | "
          f"{f'{hgate}/{n}':>12}{hms:11.2f}{st.get('n_resc', -1):>8}{base_ms/hms:8.2f}x")

gf.STARTS = BASE; gi.STARTS = BASE
print("\nblind lat = same_lattice only (the 84-85/120 figure). hybrid gate = correct lattice AND")
print(">=25% spots AND >=10 refl. n_resc = frames the known-cell rescue had to recover.")
