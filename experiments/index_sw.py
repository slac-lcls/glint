"""Idea 2 test: does RECOVERING below-threshold peaks help blind indexing? cxidb_sw120.npz
has per-frame STRONG (s*) and sub-threshold WEAK (w*) peaks (peakfinder8 keeps only strong).
Index_blind at 3 peak levels and compare same_lattice rate + median matched reflections:
  strong-only | strong+weak (all) | weak adds true reflections (help) vs noise (hurt)?

  python index_sw.py [npz] [N]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
from glint_fast import index_blind_fast, LYSO, matched
from glint.multishot import same_lattice
DEV = ("cuda" if torch.cuda.is_available() else "cpu")


def run(frames, label):
    sl = 0; mts = []; t0 = time.time()
    for q in frames:
        if len(q) < 6:
            continue
        M = index_blind_fast(q)
        if M is not None and same_lattice(M, LYSO):
            sl += 1; mts.append(matched(M, q))
    if DEV == "cuda": torch.cuda.synchronize()
    n = len(frames)
    print(f"  {label:16}: same_lattice {sl}/{n} ({100*sl//n}%)  median_matched={int(np.median(mts)) if mts else 0}  "
          f"{1e3*(time.time()-t0)/n:.0f} ms/frame")
    return sl


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else "/sdf/home/s/smarches/cxidb_sw120.npz"
    d = np.load(npz, allow_pickle=True)
    N = int(d["n"]) if "n" in d else 120
    N = int(sys.argv[2]) if len(sys.argv) > 2 else N
    strong = [np.asarray(d[f"s{i}"], float) for i in range(N)]
    weak = [np.asarray(d[f"w{i}"], float) for i in range(N)]
    both = [np.vstack([s, w]) for s, w in zip(strong, weak)]
    print(f"cxidb strong/weak peaks: median strong={int(np.median([len(s) for s in strong]))} "
          f"weak={int(np.median([len(w) for w in weak]))} both={int(np.median([len(b) for b in both]))}")
    index_blind_fast(strong[0])                                  # warmup
    print(f"blind rate (STEPS={os.environ['STEPS']}):")
    run(strong, "strong-only")
    run(both, "strong+weak")
