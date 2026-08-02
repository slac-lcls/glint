"""Can the streaming arrangement speed up the OFFLINE pipeline?

Offline runs blind + N-best on EVERY frame, then known-cell-rescues the failures. Streaming showed the
consensus cell is reachable from a median of six frames (zero false locks in 400) and equals the batch
consensus cell to <0.01 A. So blind-on-all is not required BY THE CONSENSUS -- it is the pipeline order.

This measures the reordering. Same frames, same strict gate, two arrangements:

  A  discovery-first (shipped offline): blind on all N, then known-cell rescue on the failures.
  B  registration-first (streaming's order, run offline): blind on the first K only -> consensus cell
     -> known-cell-index all N in one batch -> blind retry ONLY on the frames that fail the gate.

Reported per arrangement: yield at the gate, the number of BLIND solves actually performed (the cost
that matters, ~100x a known-cell index), and wall time. B is swept over K.

  python offline_restructure.py [frames.txt]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT); sys.path.insert(0, WT + "/experiments")
import glint.glint_fast as gf
from glint.glint_fast import index_blind_nbest
from glint.multishot import same_lattice, consensus_cell
from what_are_the_failures import strict_gate, LYSO
import glint.replica_gpu_batch as rgb

PATH = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
KS = [5, 10, 20, 40]


def blind_cells(frames, idxs, nbest=3):
    """Blind-solve a subset; return the per-frame best cells (for voting)."""
    out = []
    for i in idxs:
        try:
            for c, _s in index_blind_nbest(frames[i], nbest):
                if c is not None:
                    out.append(np.asarray(c, float))
                break
        except Exception:
            pass
    return out


def vote(cells):
    """Largest mutually-same_lattice cluster -> the consensus cell."""
    best, bn = None, 0
    for a in cells:
        n = sum(1 for b in cells if same_lattice(a, b))
        if n > bn:
            best, bn = a, n
    return best, bn


def main():
    frames = [np.asarray(f, float) for f in gf.load(PATH) if len(f) >= 6]
    N = len(frames)
    print(f"{PATH}: {N} frames, strict gate (same_lattice + >=25% spots + >=10 refl)\n")

    # ---------------- A: discovery-first, the shipped offline order ----------------
    from glint.hybrid_stream import hybrid_index
    t0 = time.time()
    res, st = hybrid_index(frames, Mc_known=LYSO, warmup=True)
    tA = time.time() - t0
    yA = sum(int(strict_gate(r["M"], q, LYSO)) for r, q in zip(res, frames))
    print(f"{'arrangement':38s} {'yield':>7s} {'blind solves':>13s} {'wall (s)':>9s}")
    print("-" * 72)
    print(f"{'A  discovery-first (shipped)':38s} {yA:4d}/{N:<3d} {N:13d} {tA:9.1f}")

    # ---------------- B: registration-first, streaming's order ----------------
    for K in KS:
        if K >= N:
            continue
        t0 = time.time()
        cells = blind_cells(frames, range(K))          # blind on the first K only
        Mc, support = vote(cells)
        if Mc is None:
            print(f"B  K={K}: no consensus"); continue
        Ms = rgb.index_fused(frames, Mc, B=N)          # one batched known-cell pass over ALL
        ok, failed = [], []
        for i, M in enumerate(Ms):
            MM = None if M is None else np.asarray(M, float)
            (ok if strict_gate(MM, frames[i], Mc) else failed).append(i)
        rec = 0
        for i in failed:                                # blind ONLY where the gate failed
            try:
                for c, _s in index_blind_nbest(frames[i], 3):
                    if c is not None and strict_gate(np.asarray(c, float), frames[i], Mc):
                        rec += 1
                        break
            except Exception:
                pass
        tB = time.time() - t0
        nblind = K + len(failed)
        yB = len(ok) + rec
        print(f"{'B  registration-first, K=' + str(K):38s} {yB:4d}/{N:<3d} {nblind:13d} {tB:9.1f}"
              f"   (support {support}/{K}, {len(failed)} retried)")

    print(f"\nblind is ~100x a known-cell index, so 'blind solves' is the cost that matters.")


if __name__ == "__main__":
    main()
