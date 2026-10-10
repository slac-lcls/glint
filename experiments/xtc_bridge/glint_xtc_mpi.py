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

GPU placement belongs to the launcher; this wrapper never writes CUDA_VISIBLE_DEVICES. Give each rank
its own GPU (srun --gpus-per-task=1, --gpu-bind, or a per-rank CUDA_VISIBLE_DEVICES). The rank's torch
indexer and the conda2 bridge worker it spawns inherit the same unmodified mask, so both use its first
device. A rank that would have to guess -- several ranks on a node, each seeing several GPUs or an unset
mask -- stops with the launch fix instead, unless --share-gpu says those ranks share one GPU on purpose.
See check_gpu_binding.

For xtc2 the MPI lives only at the conda1 level: each rank independently calls its own conda2 bridge
worker with its shard (nranks worker processes, disjoint events), so there is no MPI inside conda2.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CHUNK_START = "----- Begin chunk -----"


class GpuBindingError(RuntimeError):
    """The launcher has not decided this rank's GPU, and guessing could put ranks on the wrong one."""


LAUNCH_FIX = ("give each rank its own GPU from the launcher (srun --gpus-per-task=1, or a per-rank "
              "CUDA_VISIBLE_DEVICES), run one rank per node, or pass --share-gpu if this node's ranks "
              "should share one GPU")


def check_gpu_binding(cvd, local_size, share_gpu=False):
    """Describe the GPU this rank will use, WITHOUT changing CUDA_VISIBLE_DEVICES; raise GpuBindingError
    when the launcher has not decided it. cvd is the inherited CUDA_VISIBLE_DEVICES (None if unset);
    local_size is the number of MPI ranks on this node.

    The launcher owns placement (glint#256 review). The rank's torch indexer and the conda2 bridge worker
    it spawns inherit the same unmodified mask, so both use its first device. GLINT used to write the
    variable instead, but a write cannot be relied on: importing mpi4py can initialize CUDA (lcls2#175
    measured it under psana's MPI), after which the write no longer moves this process, only the worker
    it spawns, and the two can end up on different GPUs.
      * one device (ordinal, UUID or MIG id) -> the launcher's choice;
      * an empty mask                        -> the launcher gave this rank no GPU;
      * several devices, or unset            -> fine for a node's only rank, which uses the first device;
                                                with several ranks on the node they would all use that
                                                same device, so refuse unless share_gpu says it is meant.
    """
    if cvd is not None:
        devices = [d.strip() for d in cvd.split(",") if d.strip()]
        if len(devices) == 1:
            return f"GPU {devices[0]} (set by the launcher)"
        if not devices:
            return "no GPU (CUDA_VISIBLE_DEVICES is empty)"
        seen = f"CUDA_VISIBLE_DEVICES={cvd}"
    else:
        seen = "CUDA_VISIBLE_DEVICES unset"
    if local_size <= 1:
        return f"the first visible GPU ({seen}; the only rank on this node)"
    if share_gpu:
        return f"the first visible GPU, shared by the {local_size} ranks on this node (--share-gpu; {seen})"
    raise GpuBindingError(f"{local_size} ranks on this node with {seen}: every rank would use the same "
                          f"first GPU. To fix: {LAUNCH_FIX}.")


def retired_gpus_per_node(value, environ):
    """--gpus-per-node and GLINT_GPUS_PER_NODE made GLINT write CUDA_VISIBLE_DEVICES; both are retired.
    Return the error text when either asks for pinning (anything but unset or 0), else None."""
    env = (environ.get("GLINT_GPUS_PER_NODE") or "").strip()
    if (value is None or value == 0) and env in ("", "0"):
        return None
    given = f"--gpus-per-node {value}" if value not in (None, 0) else f"GLINT_GPUS_PER_NODE={env}"
    return (f"{given}: this option was removed, because GLINT no longer sets CUDA_VISIBLE_DEVICES. "
            f"Instead, {LAUNCH_FIX}.")


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
    ap.add_argument("--gpus-per-node", type=int, default=None,
                    help="removed: GLINT no longer sets CUDA_VISIBLE_DEVICES; give each rank its GPU from the "
                         "launcher (srun --gpus-per-task=1)")
    ap.add_argument("--share-gpu", action="store_true",
                    help="allow several ranks on a node to share its first visible GPU when the launcher has "
                         "not given each rank its own")
    ap.add_argument("--keep-parts", action="store_true", help="keep the per-rank partial .stream files")
    args = ap.parse_args(argv)
    if not args.exp:
        if rank == 0:
            sys.stderr.write("set --exp or GLINT_EXP (the beamtime id is not committed)\n")
        return 2

    # Decide placement on every rank, then agree, so a refusal on one rank stops all of them here
    # rather than leaving the others blocked in the gather below.
    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_size = node.Get_size()
    node.Free()
    gpu, problem = None, retired_gpus_per_node(args.gpus_per_node, os.environ)
    if problem is None:
        try:
            gpu = check_gpu_binding(os.environ.get("CUDA_VISIBLE_DEVICES"), local_size, args.share_gpu)
        except GpuBindingError as e:
            problem = str(e)
    bad = [(r, p) for r, p in enumerate(comm.allgather(problem)) if p]
    if bad:
        if rank == 0:
            for r, p in bad[:4]:
                sys.stderr.write(f"[glint_xtc_mpi] rank {r}: {p}\n")
            if len(bad) > 4:
                sys.stderr.write(f"[glint_xtc_mpi] ... and {len(bad) - 4} more ranks\n")
        return 2
    if rank == 0:
        print(f"[glint_xtc_mpi] {size} ranks; psana{args.psana}; {args.exp} run {args.run}; "
              f"rank 0 uses {gpu}; out {args.out}", flush=True)

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
