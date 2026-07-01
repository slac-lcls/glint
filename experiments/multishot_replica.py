"""① Multishot with the FAITHFUL ffbidx replica as the rescue engine (vs the spurious
pair-angle). Blind-index -> derive consensus cell -> rescue blind FAILURES with the
validated ffbidx replica (rotation search), keeping blind successes. Score everything
GATED [same_lattice(LYSO) AND (frac>=.25 / >=10 matched refl)] so the lift is real.
Compares: blind | hybrid+pair-angle (old) | hybrid+replica (new).

  python multishot_replica.py [frames.txt] [N]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CDIRS", "16384")                  # replica sampling density for rescue
import numpy as np
sys.path.insert(0, "/Users/smarches/git/glint")
sys.path.insert(0, "/Users/smarches/git/glint/experiments")
from glint_index import index_blind
from replica_v2 import index_known as replica_rescue      # faithful ffbidx replica (tetragonal lyso)
from glint.lattice import cell_to_Ar
from glint.multishot import (consensus_cell, index_known_pairangle, reference_lattice,
                                same_lattice)

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)


def matched(M, q, tol=0.15):
    return 0 if M is None else int((np.abs(q @ M - np.rint(q @ M)).max(1) < tol).sum())


def gpass(M, q):
    """(frac>=.25, >=10refl) under same_lattice(LYSO)."""
    if M is None or not same_lattice(M, LYSO):
        return (0, 0)
    m = matched(M, q)
    return (int(m / len(q) >= 0.25), int(m >= 10))


def load(p):
    fr = []; L = open(p).read().split("\n"); i = 0
    while i < len(L):
        if not L[i].startswith("FRAME"): i += 1; continue
        _, fid, n = L[i].split(); n = int(n)
        fr.append(np.array([[float(x) for x in L[i + 1 + j].split()] for j in range(n)])); i += 1 + n
    return fr


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/frames_cxidb.txt"
    frames = [q for q in load(path) if len(q) >= 6]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
    frames = frames[:N]
    qmax_all = max(float(np.linalg.norm(q, axis=1).max()) for q in frames)
    n = len(frames)

    t0 = time.time(); blind = [index_blind(q) for q in frames]
    print(f"blind done {time.time()-t0:.0f}s", flush=True)
    Mc, support = consensus_cell([M for M in blind if M is not None])
    print(f"consensus {np.round(np.sort(np.linalg.norm(Mc,axis=0)),1)} support {support}  "
          f"lyso?={same_lattice(Mc, LYSO)}", flush=True)
    Vref = reference_lattice(Mc, qmax_all)

    tally = {k: np.zeros(2, int) for k in ("blind", "hybrid+pairangle", "hybrid+replica")}
    t1 = time.time()
    for i, q in enumerate(frames):
        Mb = blind[i]
        tally["blind"] += gpass(Mb, q)
        if Mb is not None and same_lattice(Mb, Mc):          # keep blind success for both hybrids
            tally["hybrid+pairangle"] += gpass(Mb, q)
            tally["hybrid+replica"] += gpass(Mb, q)
            continue
        qmax = float(np.linalg.norm(q, axis=1).max())
        kn = index_known_pairangle(q, qmax, Mc, Vref=Vref, tol_frac=0.03, len_tol=0.05, ang_tol=4.0)
        tally["hybrid+pairangle"] += gpass(kn.M, q)
        tally["hybrid+replica"] += gpass(replica_rescue(q), q)
    print(f"rescue done {time.time()-t1:.0f}s\n")
    print(f"{'pipeline':20} {'frac>=.25':>12} {'>=10refl':>12}")
    for k, v in tally.items():
        print(f"  {k:18} {v[0]}/{n} ({100*v[0]//n}%)   {v[1]}/{n} ({100*v[1]//n}%)")
