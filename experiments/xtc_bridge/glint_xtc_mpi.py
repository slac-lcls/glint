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
device. Before reading, every rank reports the physical UUID of that device; the run stops with the
launch fix if a rank sees no GPU, or if two ranks on a node would use the same GPU, unless --share-gpu
says they share it on purpose. See decide_placement.

For xtc2 the MPI lives only at the conda1 level: each rank independently calls its own conda2 bridge
worker with its shard (nranks worker processes, disjoint events), so there is no MPI inside conda2.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CHUNK_START = "----- Begin chunk -----"


LAUNCH_FIX = "give each rank its own GPU from the launcher (srun --gpus-per-task=1, or a per-rank CUDA_VISIBLE_DEVICES)"
SHARE_FIX = "or pass --share-gpu if these ranks should share one GPU"


def gpu_identity():
    """(uuid, note) for the GPU this rank and its reader will use: CUDA device 0 under the inherited mask.

    uuid is None when no CUDA device is visible (note says so) or when this torch cannot report a device
    UUID (note "no uuid"). Initializing CUDA here is safe: nothing writes the mask afterwards, and the
    reader is a separate process started with subprocess, not a fork."""
    try:
        import torch
    except ImportError as e:
        return None, f"no uuid: torch not importable ({e})"
    if not torch.cuda.is_available():
        return None, "no GPU visible"
    props = torch.cuda.get_device_properties(0)
    uuid = getattr(props, "uuid", None)
    if uuid is None:
        return None, f"no uuid: torch {torch.__version__} does not report device UUIDs ({props.name})"
    return f"GPU-{uuid}", props.name


def placement_record(rank, host, environ, retired, identity):
    """What one rank reports before reading: who it is, its mask, and the GPU it resolved to."""
    uuid, note = identity if identity is not None else (None, "not checked")
    return {"rank": rank, "host": host, "cvd": environ.get("CUDA_VISIBLE_DEVICES"),
            "uuid": uuid, "note": note, "retired": retired}


def decide_placement(records, share_gpu=False):
    """Return the problems (a list of strings, empty = run) from every rank's placement_record.

    Deterministic in its input, so every rank, given the same allgathered list, reaches the same verdict.
    The rules, by physical identity rather than by the mask string (equal strings need not be one GPU,
    and different strings can be):
      * a retired option on any rank stops the run;
      * a rank that sees no GPU stops the run: the reader's peak finders run on the GPU (cupy);
      * two ranks on one host resolving to the same GPU UUID stop the run, unless share_gpu;
      * a rank whose torch cannot report a UUID falls back to its mask: several devices or none set,
        with other ranks on its host, stops the run unless share_gpu (it would use the same first GPU).
    """
    problems = [f"rank {r['rank']}: {r['retired']}" for r in records if r["retired"]]
    if problems:
        return problems
    by_host = {}
    for r in records:
        by_host.setdefault(r["host"], []).append(r)
    for r in records:
        if r["uuid"] is None and r["note"] == "no GPU visible":
            problems.append(f"rank {r['rank']} on {r['host']} sees no GPU (CUDA_VISIBLE_DEVICES="
                            f"{r['cvd']!r}); the reader's peak finders run on the GPU. To fix: {LAUNCH_FIX}.")
    if share_gpu:
        return problems
    for host, rs in sorted(by_host.items()):
        by_gpu = {}
        for r in rs:
            if r["uuid"] is not None:
                by_gpu.setdefault(r["uuid"], []).append(r["rank"])
        for uuid, ranks in sorted(by_gpu.items()):
            if len(ranks) > 1:
                problems.append(f"ranks {', '.join(map(str, ranks))} on {host} would all use {uuid}. "
                                f"To fix: {LAUNCH_FIX}, {SHARE_FIX}.")
        for r in rs:
            if r["uuid"] is None and r["note"].startswith("no uuid") and len(rs) > 1:
                devices = [d for d in (r["cvd"] or "").split(",") if d.strip()]
                if r["cvd"] is None or len(devices) > 1:
                    problems.append(f"rank {r['rank']} on {host} cannot check its GPU ({r['note']}) and sees "
                                    f"CUDA_VISIBLE_DEVICES={r['cvd']!r} with {len(rs)} ranks on the host, so "
                                    f"they would use the same first GPU. To fix: {LAUNCH_FIX}, {SHARE_FIX}.")
    return problems


def placement_summary(records, limit=8):
    """One line per rank for rank 0's log: host, rank, mask, and the GPU it resolved to."""
    rows = [f"  {r['host']} rank {r['rank']}: CUDA_VISIBLE_DEVICES={r['cvd']!r} -> {r['uuid'] or r['note']}"
            for r in sorted(records, key=lambda r: (r["host"], r["rank"]))]
    if len(rows) > limit:
        rows = rows[:limit] + [f"  ... and {len(rows) - limit} more ranks"]
    return rows


def retired_gpus_per_node(value, environ):
    """--gpus-per-node and GLINT_GPUS_PER_NODE made this wrapper write CUDA_VISIBLE_DEVICES; both are retired.
    Return the error text when either asks for pinning (anything but unset or 0), else None."""
    env = (environ.get("GLINT_GPUS_PER_NODE") or "").strip()
    if value not in (None, 0):
        return (f"--gpus-per-node {value} was removed (glint_xtc_mpi no longer sets CUDA_VISIBLE_DEVICES). "
                f"Drop it, and {LAUNCH_FIX}.")
    if env not in ("", "0"):
        return (f"GLINT_GPUS_PER_NODE={env} was removed (glint_xtc_mpi no longer sets CUDA_VISIBLE_DEVICES). "
                f"Unset it, and {LAUNCH_FIX}.")
    return None


def agree_on_placement(comm, args, environ, identity_fn=None, host=None):
    """Every rank reports its placement_record; all ranks decide on the same allgathered list.

    Returns (problems, records). Collective: every rank must call it, once, at the same point."""
    import socket
    retired = retired_gpus_per_node(args.gpus_per_node, environ)
    identity = None if retired else (identity_fn or gpu_identity)()
    rec = placement_record(comm.Get_rank(), host or socket.gethostname(), environ, retired, identity)
    records = comm.allgather(rec)
    return decide_placement(records, args.share_gpu), records


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


def build_parser():
    """glint_xtc's parser plus this wrapper's options. Needs no MPI, so --help text is testable on CPU."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # import sibling glint_xtc + readers
    import glint_xtc

    ap = glint_xtc.build_parser()
    ap.add_argument("--gpus-per-node", type=int, default=None,
                    help="removed: this wrapper no longer sets CUDA_VISIBLE_DEVICES; give each rank its GPU from "
                         "the launcher (srun --gpus-per-task=1)")
    ap.add_argument("--share-gpu", action="store_true",
                    help="allow several ranks on a node to use the same GPU (otherwise two ranks resolving to "
                         "one GPU stop the run)")
    ap.add_argument("--keep-parts", action="store_true", help="keep the per-rank partial .stream files")
    return ap


def main(argv=None):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    args = build_parser().parse_args(argv)   # also puts this directory on sys.path
    import glint_xtc
    if not args.exp:
        if rank == 0:
            sys.stderr.write("set --exp or GLINT_EXP (the beamtime id is not committed)\n")
        return 2

    # Every rank reports the GPU it resolved to, then all decide on the same list, so a refusal stops
    # every rank here instead of leaving the others blocked in the gather below.
    problems, records = agree_on_placement(comm, args, os.environ)
    if problems:
        if rank == 0:
            for p in problems[:6]:
                sys.stderr.write(f"[glint_xtc_mpi] {p}\n")
            if len(problems) > 6:
                sys.stderr.write(f"[glint_xtc_mpi] ... and {len(problems) - 6} more\n")
        return 2
    if rank == 0:
        print(f"[glint_xtc_mpi] {size} ranks; psana{args.psana}; {args.exp} run {args.run}; out {args.out}"
              + ("; --share-gpu" if args.share_gpu else ""), flush=True)
        print("\n".join(placement_summary(records)), flush=True)

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
