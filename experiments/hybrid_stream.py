"""Productized GLINT hybrid -> CrystFEL .stream (the LUTE/LCLS deliverable, fully blind).

Pipeline (no cell prior):
  1. blind-index every frame (GPU FAST front-end)            -> per-frame M
  2. consensus_cell over the blind successes                  -> the run cell Mc (decisive)
  3. rescue the failures against Mc with the cell-GENERAL     -> recovered orientations
     known-cell GPU indexer (index_known_gpu_cell, not the lysozyme-specialized one)
  4. write a .stream where every indexed crystal shares Mc's metric (consensus-consistent)

The rescue is now cell-general, so this runs on ANY protein, not just lysozyme.

  python hybrid_stream.py [frames.txt] [N] [out.stream]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
from glint_fast import index_blind_fast, load
from replica_gpu import index_known_gpu_cell
from fftindex.multishot import consensus_cell, same_lattice
from fftindex.stream import write_stream


def _hkl(q, M):
    H = q @ M; r = np.rint(H); inl = np.abs(H - r).max(1) < 0.15
    return r[inl].astype(int), q[inl], int(inl.sum())


def hybrid_index(frames, images=None, Mc_known=None, warmup=True, xg_fallback=False):
    """Fully-blind hybrid: blind-index all -> consensus cell -> cell-general rescue of failures ->
    consensus-consistent results (list of write_stream dicts) + a stats dict. Pass Mc_known to
    skip consensus and rescue against a supplied cell.  frames: list of (N,3) q in 1/A.
    xg_fallback=True also runs the xgandalf-paper indexer (paper_xg_gpu, defect-greedy assembly):
    its cells add independent consensus votes and rescue GLINT-wrong frames its different
    selection catches (complementary; tried before the known-cell rescue)."""
    n = len(frames)
    images = images or [{"image": "glint.cxi", "event": i} for i in range(n)]
    xg_blind = None
    if xg_fallback:
        from paper_xg_gpu import index_blind as xg_blind
    if warmup and n:
        index_blind_fast(frames[0])
        if xg_blind:
            xg_blind(frames[0])

    blind = [index_blind_fast(q) for q in frames]
    xg = [xg_blind(q) for q in frames] if xg_blind else [None] * n
    n_blind = sum(M is not None for M in blind)

    if Mc_known is not None:
        Mc, support = np.asarray(Mc_known, float), -1
    else:                                                        # xg cells add independent votes
        votes = [M for M in blind if M is not None] + [M for M in xg if M is not None]
        Mc, support = consensus_cell(votes)

    results = []; n_idx = n_resc = n_xg = 0
    for q, M, Mx, meta in zip(frames, blind, xg, images):
        consistent = M is not None and Mc is not None and same_lattice(M, Mc)
        if not consistent and Mx is not None and Mc is not None and same_lattice(Mx, Mc):
            M = Mx; consistent = True; n_xg += 1                 # xgandalf complementary catch
        if not consistent and Mc is not None:
            Mr = index_known_gpu_cell(q, Mc)
            if Mr is not None and same_lattice(Mr, Mc):
                M = Mr; consistent = True; n_resc += 1
        use = M if (consistent or (Mc is None and M is not None)) else None
        if use is not None:
            hkl, qin, _ = _hkl(q, use); n_idx += 1
        else:
            hkl, qin = None, q
        results.append({"image": meta["image"], "event": meta["event"],
                        "M": use, "q": qin, "hkl": hkl})
    edges = np.round(np.sort(np.linalg.norm(Mc, axis=0)), 1) if Mc is not None else None
    stats = {"n": n, "n_blind": n_blind, "support": support, "edges": edges,
             "n_resc": n_resc, "n_xg": n_xg, "n_idx": n_idx, "Mc": Mc}
    return results, stats


def _report(stats, out):
    n = max(stats["n"], 1)
    print(f"=== GLINT hybrid (blind+consensus+general-rescue), N={stats['n']} ===")
    print(f"  blind indexed      : {stats['n_blind']}/{stats['n']} ({100*stats['n_blind']//n}%)")
    print(f"  consensus cell     : {stats['edges']} A  support {stats['support']}")
    if stats.get("n_xg"):
        print(f"  xgandalf fallback  : {stats['n_xg']} caught")
    print(f"  rescued failures   : {stats['n_resc']}")
    print(f"  FINAL indexed      : {stats['n_idx']}/{stats['n']} ({100*stats['n_idx']//n}%)  -> {out}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = sys.argv[3] if len(sys.argv) > 3 else "glint_hybrid.stream"
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6]
    if N: frames = frames[:N]
    images = [{"image": os.path.basename(path), "event": i} for i in range(len(frames))]
    results, stats = hybrid_index(frames, images, xg_fallback=os.environ.get("XGFALL", "0") == "1")
    write_stream(results, out)
    _report(stats, out)
