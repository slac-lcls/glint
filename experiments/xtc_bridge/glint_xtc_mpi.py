"""MPI-sharded GLINT-from-xtc -- the scale-out wrapper around glint_xtc.

The per-rank body IS glint_xtc (read -> peak-find -> blind index -> .stream); this only shards events
across ranks and merges the partial streams. That is exactly how a LUTE / slurm SFX indexing task
scales (each rank an event slice, Concat at the end), so wrapping glint_xtc this way turns "works on a
run" into "scales like indexamajig", for both data eras and with no change to the indexing itself.

    # -n = ranks = event shards. Pin ONE GPU per rank so each rank's peak-find + index (and, for xtc2,
    # the conda2 bridge worker it spawns and shares an env with) land on a distinct device.
    srun -n 8 --gpus-per-task=1 python glint_xtc_mpi.py --exp X --run N --det jungfrau --zdist 0.246 -o out.stream        # xtc1
    srun -n 8 --gpus-per-task=1 python glint_xtc_mpi.py --exp X --run N --zdist 0.246 --psana 2 -o out.stream             # xtc2

Sharding: event i is owned by rank (i % nranks) -- round-robin, so indexable frames that cluster in
time still spread evenly and no rank needs the event count up front (see xtc_core.event_in_shard). The
global event index is written into each chunk, so partial streams from different ranks never collide and
merging is a plain header-once chunk concatenation.

GPU pinning: preferred is to let the launcher give each rank one device (srun --gpus-per-task=1 /
--gpu-bind, or CUDA_VISIBLE_DEVICES per task) -- then this does nothing. Otherwise pass --gpus-per-node
N and each rank is pinned to one device: one of the devices the launcher left visible when its mask lists
several (allowed[local_rank % len(allowed)]), else local_rank % N; an empty mask (a CPU rank) is kept. Set
in the environment BEFORE torch/cupy import and BEFORE the bridge worker spawns (the worker inherits the
env), so the whole rank shares one GPU. The mask is only ever narrowed, never replaced (lcls2#155).

For xtc2 the MPI lives only at the conda1 level: each rank independently calls its own conda2 bridge
worker with its shard (nranks worker processes, disjoint events), so there is no MPI inside conda2.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CHUNK_START = "----- Begin chunk -----"


def _local_rank():
    """Node-local rank from whichever launcher set it (SLURM / OpenMPI / MVAPICH / Hydra); None if
    unknown (single node or an unrecognised launcher)."""
    for k in ("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK",
              "MPI_LOCALRANKID", "PMI_LOCAL_RANK"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return None


def pin_gpu(local_rank, gpus_per_node):
    """Pin this rank to one GPU by setting CUDA_VISIBLE_DEVICES, BEFORE any torch/cupy import and before
    the conda2 bridge worker spawns (it inherits the env), so the in-env index and the bridged peak-find
    share one device. Returns the CUDA_VISIBLE_DEVICES in effect (None if unset and left untouched).

    The launcher's mask is authoritative: whatever it left visible is the ALLOWED set, and this only ever
    narrows that set, never replaces it (the rule psana2's MPI GPU path is adopting in lcls2#155, and the
    two processes share a device by inheriting this variable).
      * one device (ordinal or UUID)  -> trust it; no-op (the documented srun --gpus-per-task=1 path);
      * an empty mask                 -> a CPU-only rank; left empty, never re-enabled;
      * several devices               -> with --gpus-per-node, pick allowed[local_rank % len(allowed)];
                                         without it (0 = trust launcher) leave them all visible;
      * unset                         -> with --gpus-per-node N, pin local_rank % N; else leave unset.
    Until this version a multi-device mask was overwritten with the bare physical ordinal local_rank % N,
    which can name a GPU outside the allowed set, and an empty mask was read as unset."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    lr = local_rank if local_rank is not None else 0
    if cvd is not None:
        allowed = [d.strip() for d in cvd.split(",") if d.strip()]
        if len(allowed) <= 1:
            return cvd                       # one device, or none: the launcher decided; keep it
        if gpus_per_node and gpus_per_node > 0:
            dev = allowed[lr % len(allowed)]
            os.environ["CUDA_VISIBLE_DEVICES"] = dev
            return dev
        return cvd                           # several visible, no pin requested: trust the launcher
    if gpus_per_node and gpus_per_node > 0:
        dev = str(lr % gpus_per_node)
        os.environ["CUDA_VISIBLE_DEVICES"] = dev
        return dev
    return None                              # unset, no pin requested; caller may pin via the launcher


def concat_streams(part_paths, out_path):
    """Merge per-rank partial .stream files: emit the header once (the prefix before the first chunk,
    taken from the first non-empty part), then every part's chunk region in order. Header-format
    agnostic (keys only on the chunk marker) and tolerant of header-only empty shards. Returns the total
    chunk count merged."""
    wrote_header = False
    nchunks = 0
    with open(out_path, "w") as out:
        for p in part_paths:
            try:
                with open(p) as f:
                    text = f.read()
            except OSError:
                continue
            idx = text.find(CHUNK_START)
            head = text if idx < 0 else text[:idx]
            body = "" if idx < 0 else text[idx:]
            if not wrote_header and head:
                out.write(head)
                wrote_header = True
            out.write(body)
            nchunks += body.count(CHUNK_START)
    return nchunks


def main(argv=None):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    sys.path.insert(0, str(Path(__file__).resolve().parent))  # import sibling glint_xtc + readers
    import glint_xtc

    ap = glint_xtc.build_parser()
    ap.add_argument("--gpus-per-node", type=int, default=int(os.environ.get("GLINT_GPUS_PER_NODE", "0")),
                    help="pin this rank to one GPU: one of the launcher's CUDA_VISIBLE_DEVICES when it lists several, "
                         "else local_rank %% N; an empty mask is kept; 0 = trust the launcher (default)")
    ap.add_argument("--keep-parts", action="store_true", help="keep the per-rank partial .stream files")
    args = ap.parse_args(argv)
    if not args.exp:
        if rank == 0:
            sys.stderr.write("set --exp or GLINT_EXP (the beamtime id is not committed)\n")
        return 2

    dev = pin_gpu(_local_rank(), args.gpus_per_node)
    if rank == 0:
        print(f"[glint_xtc_mpi] {size} ranks; psana{args.psana}; {args.exp} run {args.run}; "
              f"CUDA_VISIBLE_DEVICES={dev}; out {args.out}", flush=True)

    # each rank reads + indexes its event shard into a partial stream
    out = glint_xtc.read_qframes(args, rank=rank, nranks=size, verbose=(rank == 0))
    part = f"{args.out}.rank{rank:04d}"
    _, _, n_idx = glint_xtc.index_and_write(out, args, part, report=False)

    # gather per-rank counts; rank 0 concatenates the partials in rank order and reports the totals
    counts = comm.gather((rank, out["n_sent"], n_idx, out["n_events"], out["n_skipped_wl"]), root=0)
    comm.Barrier()
    if rank == 0:
        counts.sort()
        parts = [f"{args.out}.rank{r:04d}" for r, *_ in counts]
        merged = concat_streams(parts, args.out)
        if not args.keep_parts:
            for p in parts:
                try:
                    os.remove(p)
                except OSError:
                    pass
        n_sent = sum(c[1] for c in counts)
        total_idx = sum(c[2] for c in counts)
        n_ev = sum(c[3] for c in counts)
        n_wl = sum(c[4] for c in counts)
        assert merged >= total_idx, f"stream merge lost chunks: {merged} in file < {total_idx} indexed"
        print(f"[glint_xtc_mpi] {total_idx} indexed / {n_sent} frames ({merged} chunks, {n_ev} events, "
              f"{n_wl} skipped no-wavelength) across {size} ranks -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
