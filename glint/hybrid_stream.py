"""Productized GLINT hybrid -> CrystFEL .stream (the LUTE/LCLS deliverable, fully blind).

Pipeline (no cell prior):
  1. N-BEST blind-index every frame (top-3 distinct cells)     -> per-frame hypotheses
  2. consensus over the POOLED N-best hypotheses               -> the run cell Mc (sturdier)
  3. per frame pick the consensus-consistent N-best cell       -> recovers ambiguity-demoted truth
  4. cell-GENERAL known-cell GPU rescue for the rest           -> recovered orientations
  5. write a .stream where every indexed crystal shares Mc's metric (consensus-consistent)

The rescue is cell-general (runs on ANY protein); N-best keeps the reachable-but-not-top-1
hypotheses that single-shot orientation ambiguity demotes, arbitrated by cross-frame consensus.

  python hybrid_stream.py [frames.txt] [N] [out.stream]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np
from glint.glint_fast import index_blind_fast, index_blind_nbest, load
from glint.replica_gpu import index_known_gpu_cell
from glint.multishot import consensus_cell, same_lattice
from glint.stream import write_stream


def _hkl(q, M):
    H = q @ M; r = np.rint(H); inl = np.abs(H - r).max(1) < 0.15
    return r[inl].astype(int), q[inl], int(inl.sum())


def hybrid_index(frames, images=None, Mc_known=None, warmup=True, nbest=3, cascade=None):
    """Fully-blind hybrid. (1) N-BEST blind-index every frame (top-`nbest` distinct cells, not just
    argmax). (2) consensus over the POOLED N-best hypotheses (aliases scatter, truth clusters ->
    sturdier cell). (3) per frame pick the highest-scored N-best cell consistent with the consensus
    Mc -- this recovers reachable-but-not-top-1 cells that single-shot orientation ambiguity demotes
    (+5 gated, corroborated). (4) cell-general known-cell GPU rescue for the rest. (5) OPTIONAL
    external-indexer `cascade` (e.g. ffbidx) on whatever is still unindexed -- free insurance for the
    marginal regime, redundant when consensus is strong. Returns write_stream dicts + stats. nbest=1
    reduces to top-1 consensus. Pass Mc_known to skip consensus and rescue against a supplied cell.
    `cascade` = callable(frames_subset, Mc) -> {local_index: M}. frames: list of (N,3) q (1/A)."""
    n = len(frames)
    images = images or [{"image": "glint.cxi", "event": i} for i in range(n)]
    if warmup and n:
        index_blind_nbest(frames[0], nbest)

    NB = [index_blind_nbest(q, nbest) for q in frames]           # [(cell,score),...] per frame
    top1 = [nb[0][0] if nb else None for nb in NB]
    n_blind = sum(1 for nb in NB if nb)

    if Mc_known is not None:
        Mc, support = np.asarray(Mc_known, float), -1
    else:                                                        # consensus over the POOLED hypotheses
        Mc, support = consensus_cell([c for nb in NB for c, _ in nb])

    results = []; n_idx = n_resc = n_nb = 0
    for q, nb, t1, meta in zip(frames, NB, top1, images):
        M = None
        if Mc is not None:                                       # pick best consensus-consistent N-best hypothesis
            for c, _ in nb:
                if same_lattice(c, Mc):
                    M = c
                    n_nb += (c is not t1)                        # recovered via a non-top-1 hypothesis
                    break
        if M is None and Mc is not None:                         # cell-general GPU known-cell rescue (ffbidx-style)
            Mr = index_known_gpu_cell(q, Mc)
            if Mr is not None and same_lattice(Mr, Mc):
                M = Mr; n_resc += 1
        if M is None and Mc is None:                             # no consensus formed -> top-1 fallback
            M = t1
        if M is not None:
            hkl, qin, _ = _hkl(q, M); n_idx += 1
        else:
            hkl, qin = None, q
        results.append({"image": meta["image"], "event": meta["event"], "M": M, "q": qin, "hkl": hkl})
    n_casc = 0                                                   # (5) optional external cascade on the still-unindexed
    if cascade is not None and Mc is not None:
        resid = [i for i, r in enumerate(results) if r["M"] is None]
        if resid:
            cm = cascade([results[i]["q"] for i in resid], Mc)  # {local_index: M}
            for k, i in enumerate(resid):
                Mx = cm.get(k)
                if Mx is not None and same_lattice(Mx, Mc):
                    hkl, qin, _ = _hkl(results[i]["q"], Mx)
                    results[i].update({"M": Mx, "q": qin, "hkl": hkl})
                    n_idx += 1; n_casc += 1
    edges = np.round(np.sort(np.linalg.norm(Mc, axis=0)), 1) if Mc is not None else None
    stats = {"n": n, "n_blind": n_blind, "support": support, "edges": edges, "n_nbest": n_nb,
             "n_resc": n_resc, "n_casc": n_casc, "n_idx": n_idx, "Mc": Mc}
    return results, stats


def dense_index(frames, images=None, warmup=True):
    """Dense/rotation path: each frame self-indexes via the local-cluster-FFT front end
    (`index_blind_cluster_seeded`) -- no cross-frame consensus needed because a rotation cloud is
    3D-complete. Below CLUSTER_MIN rlps the front end auto-falls-back to the Fibonacci grid, so this is
    safe on mixed data; the CLI picks this path only when the median rlp count is dense."""
    from glint.glint_fast import index_blind_cluster_seeded
    n = len(frames)
    images = images or [{"image": "glint.cxi", "event": i} for i in range(n)]
    if warmup and n:
        index_blind_cluster_seeded(frames[0])
    results = []; n_idx = 0
    for q, meta in zip(frames, images):
        M = index_blind_cluster_seeded(q)
        if M is not None:
            hkl, qin, _ = _hkl(q, M); n_idx += 1
        else:
            hkl, qin = None, q
        results.append({"image": meta["image"], "event": meta["event"], "M": M, "q": qin, "hkl": hkl})
    stats = {"n": n, "mode": "dense", "n_blind": n_idx, "support": -1, "edges": None,
             "n_nbest": 0, "n_resc": 0, "n_casc": 0, "n_idx": n_idx, "Mc": None}
    return results, stats


def _report(stats, out):
    n = max(stats["n"], 1)
    if stats.get("mode") == "dense":
        print(f"=== GLINT dense/rotation (local-cluster FFT), N={stats['n']} ===")
        print(f"  indexed            : {stats['n_idx']}/{stats['n']} ({100*stats['n_idx']//n}%)  -> {out}")
        return
    print(f"=== GLINT hybrid (blind+consensus+general-rescue), N={stats['n']} ===")
    print(f"  blind indexed      : {stats['n_blind']}/{stats['n']} ({100*stats['n_blind']//n}%)")
    print(f"  consensus cell     : {stats['edges']} A  support {stats['support']}")
    if stats.get("n_nbest"):
        print(f"  N-best recovered   : {stats['n_nbest']} (consensus-consistent non-top-1 hypothesis)")
    print(f"  rescued failures   : {stats['n_resc']}")
    if stats.get("n_casc"):
        print(f"  cascade recovered  : {stats['n_casc']} (external fallback on still-unindexed)")
    print(f"  FINAL indexed      : {stats['n_idx']}/{stats['n']} ({100*stats['n_idx']//n}%)  -> {out}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = sys.argv[3] if len(sys.argv) > 3 else "glint_hybrid.stream"
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6]
    if N: frames = frames[:N]
    images = [{"image": os.path.basename(path), "event": i} for i in range(len(frames))]
    results, stats = hybrid_index(frames, images)
    write_stream(results, out)
    _report(stats, out)
