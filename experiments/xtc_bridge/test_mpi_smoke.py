"""Real MPI + GPU smoke test for glint_xtc_mpi -- psana-free, but drives the ACTUAL wrapper path:
shard events over ranks -> per-rank GPU hybrid_index -> partial .stream -> rank-0 header-once concat.

    srun -p ampere -A lcls:default@ampere -q preemptable --gres=gpu:a100:1 -n 2 \
         python test_mpi_smoke.py            # 2 ranks share 1 GPU (tiny workload)
    # or inside an allocation:  mpirun -n 2 python test_mpi_smoke.py

It monkeypatches glint_xtc.read_qframes with a synthetic reader that plants a lysozyme cell (the shipped
ewald_spots generator) and shards by the SAME xtc_core.event_in_shard contract the real readers use, so
every piece the wrapper adds runs for real -- MPI rank/size, disjoint sharding, GPU indexing per rank,
partial-stream write, gather, and the merge -- with only the psana geometry (behind the README gate)
stubbed. Rank 0 asserts the merged stream covers all N events exactly once and recovers the cell.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # xtc_core, glint_xtc, glint_xtc_mpi
sys.path.insert(0, str(HERE.parents[1]))            # repo root for `glint`
sys.path.insert(0, str(HERE.parent))                # experiments/ for the ewald generator

import xtc_core
from test_geom_bridge import ewald_spots, LYSO      # shipped, tested planted-cell generator

N_EVENTS = 24                                        # synthetic run length (global event ids 0..N-1)


def _sim_read(args, rank=0, nranks=1, verbose=True):
    """Stand-in for glint_xtc.read_qframes: this rank's shard of a synthetic planted-cell run. Uses the
    real event_in_shard contract, returns the real reader dict shape, with global event ids preserved."""
    events = [i for i in range(N_EVENTS) if xtc_core.event_in_shard(i, rank, nranks)]
    qframes = [np.ascontiguousarray(np.asarray(ewald_spots(i), float)) for i in events]
    if verbose:
        print(f"[sim] rank {rank}/{nranks} owns {len(events)} of {N_EVENTS} events", flush=True)
    return {"qframes": qframes, "events": events,
            "n_events": len(events), "n_sent": len(qframes), "n_skipped_wl": 0}


def main():
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()

    import glint_xtc
    glint_xtc.read_qframes = _sim_read              # patch BEFORE the wrapper calls it (module singleton)

    import glint_xtc_mpi
    out = str(HERE / "_mpi_smoke.stream")
    # GPU pinning is left to the launcher / GLINT_GPUS_PER_NODE (portable across 1- and N-GPU nodes);
    # --zdist is required by the parser but unused here (the reader is stubbed).
    argv = ["--exp", "sim", "--run", "1", "--zdist", "0.1", "--psana", "1", "-o", out]
    rc = glint_xtc_mpi.main(argv)

    if rank != 0:
        return rc
    # rank 0: verify the merged stream
    text = Path(out).read_text()
    from glint.stream import HEADER
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    check("header appears exactly once", text.count(HEADER) == 1)
    present = [text.count(f"Event: //{i}\n") for i in range(N_EVENTS)]
    check(f"all {N_EVENTS} events present exactly once (sharding covered + disjoint)",
          all(c == 1 for c in present))
    n_idx = text.count("indexed_by = glint")
    check(f"planted cell indexed on a majority across ranks ({n_idx}/{N_EVENTS})",
          n_idx >= 0.6 * N_EVENTS)
    try:
        os.remove(out)
    except OSError:
        pass
    print("ALL PASS" if ok else "FAILURES ABOVE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
