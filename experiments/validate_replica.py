"""Validate the faithful ffbidx replica (v2 @ full sampling): run on a frame set, report
the fair metrics PLUS sanity checks — median recovered cell params (must be ~38/79/79),
median matched reflections, and what fraction are genuinely same_lattice(LYSO)."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CDIRS", "32768")                  # ffbidx sampling density
import numpy as np
sys.path.insert(0, "/Users/smarches/git/fftindex")
sys.path.insert(0, "/Users/smarches/git/fftindex/experiments")
from replica_v2 import index_known, load, LYSO
from fftindex.multishot import same_lattice


def params(M):
    return np.sort(np.linalg.norm(M, axis=0))


if __name__ == "__main__":
    path = sys.argv[1]
    frames = [q for q in load(path) if len(q) >= 6]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
    frames = frames[:N]
    f25 = fN = n = lyso = 0; mts = []; cps = []; t0 = time.time()
    for q in frames:
        n += 1
        M = index_known(q)
        if M is None:
            continue
        if not same_lattice(M, LYSO):
            continue
        lyso += 1
        m = int((np.abs(q @ M - np.rint(q @ M)).max(1) < 0.15).sum())
        mts.append(m); cps.append(params(M))
        f25 += m / len(q) >= 0.25; fN += m >= 10
    cps = np.array(cps)
    print(f"{path}  N={n}  {1e3*(time.time()-t0)/n:.0f} ms/frame")
    print(f"  frac>=.25: {f25}/{n} ({100*f25/n:.0f}%)   >=10refl: {fN}/{n} ({100*fN/n:.0f}%)")
    print(f"  same_lattice(LYSO): {lyso}/{n}   median matched refl: {int(np.median(mts)) if mts else 0}")
    if len(cps):
        med = np.median(cps, axis=0)
        print(f"  median recovered cell: [{med[0]:.1f} {med[1]:.1f} {med[2]:.1f}]  (lyso = 38.0 79.0 79.0)")
