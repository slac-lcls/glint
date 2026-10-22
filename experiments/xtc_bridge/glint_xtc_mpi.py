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
indexer and its reader inherit the same unmodified mask, so both use its first device. Before reading,
every rank identifies that device physically (CuPy's PCI bus id; torch's UUID as a fallback) and the
run stops with the launch fix if a rank sees no GPU, or if two ranks on a host would use the same GPU,
unless --share-gpu says they share it on purpose. See decide_placement.

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
    """(status, key, detail) for the GPU this rank and its reader will use: CUDA device 0 under the
    inherited mask. status is "ok" (key names the physical device, unique on this host), "no-gpu", or
    "unknown" (nothing here can identify it).

    CuPy is asked first: the reader's peak finders run on it, and CuPy reports a device's PCI bus id,
    which is unique on a host. torch's device UUID is the fallback, and it needs torch >= 2.5.
    Initializing CUDA here is safe because nothing writes the mask afterwards. On --psana 1 the reader
    runs in this process, on this same device; on --psana 2 envbridge starts it as a new process
    (subprocess, not a fork) that inherits the unmodified mask."""
    try:
        import cupy
    except ImportError:
        cupy = None
    if cupy is not None:
        try:
            n = cupy.cuda.runtime.getDeviceCount()
        except Exception as e:  # CUDARuntimeError when no device or driver is usable
            return "no-gpu", None, f"CuPy: {type(e).__name__}: {e}"
        if n < 1:
            return "no-gpu", None, "CuPy sees no device"
        bus = cupy.cuda.Device(0).pci_bus_id
        return "ok", f"PCI {bus.upper()}", f"CuPy, {n} visible"
    try:
        import torch
    except ImportError:
        return "unknown", None, "neither CuPy nor torch is importable"
    if not torch.cuda.is_available():
        if getattr(torch.version, "cuda", None) is None:
            return "unknown", None, f"no CuPy, and torch {torch.__version__} is built without CUDA"
        return "no-gpu", None, "torch sees no CUDA device"
    props = torch.cuda.get_device_properties(0)
    uuid = getattr(props, "uuid", None)
    if uuid is None:
        return "unknown", None, f"no CuPy, and torch {torch.__version__} reports no device UUID (needs >= 2.5)"
    return "ok", f"GPU-{uuid}", f"torch, {props.name}"


def placement_record(rank, host, environ, retired, identity):
    """What one rank reports before reading: who it is, its mask, and the GPU it identified."""
    status, key, detail = identity if identity is not None else ("unchecked", None, "not checked")
    return {"rank": rank, "host": host, "cvd": environ.get("CUDA_VISIBLE_DEVICES"),
            "status": status, "key": key, "detail": detail, "retired": retired}


def _first_device(cvd):
    """The first device a mask makes visible, or None when it makes none visible: unset, empty, or a mask
    whose first entry is a negative ordinal (CUDA_VISIBLE_DEVICES=-1 hides every GPU)."""
    devices = [d.strip() for d in (cvd or "").split(",") if d.strip()]
    if not devices or devices[0].startswith("-"):
        return None
    return devices[0]


def _key_kind(key):
    """Which library produced an identity key: CuPy's PCI bus id or torch's UUID. Keys of different kinds name
    the same GPU in two vocabularies, so they cannot be compared."""
    return "PCI" if (key or "").startswith("PCI ") else "UUID"


def decide_placement(records, share_gpu=False):
    """Return the problems (a list of strings, empty = run) from every rank's placement_record.

    Deterministic in its input, so every rank, given the same allgathered list, reaches the same verdict.
      * a retired option on any rank stops the run;
      * a rank that sees no GPU, or whose GPU check raised, stops the run (the reader's peak finders
        run on the GPU, cupy);
      * two ranks on one host whose GPUs have the same physical key stop the run, unless share_gpu;
      * ranks that cannot identify their GPU ("unknown") are compared by mask: two on one host whose
        first visible device is the same, or unset, stop the run unless share_gpu, because nothing
        shows they use different GPUs;
      * reports that cannot be compared stop the run unless share_gpu: on one host, an identified rank
        beside an unidentified one, an unidentified rank with an unset mask beside any other rank, or
        PCI ids beside UUIDs (the two libraries name the same GPU differently). Separate buckets are not
        proof of separate GPUs.
    """
    problems = [f"rank {r['rank']}: {r['retired']}" for r in records if r["retired"]]
    if problems:
        return problems
    for r in records:
        if r["status"] == "no-gpu" or (r["status"] == "unknown" and r["cvd"] is not None and not _first_device(r["cvd"])):
            problems.append(f"rank {r['rank']} on {r['host']} sees no GPU (CUDA_VISIBLE_DEVICES={r['cvd']!r}; "
                            f"{r['detail']}); the reader's peak finders run on the GPU. To fix: {LAUNCH_FIX}.")
        elif r["status"] not in ("ok", "unknown"):
            problems.append(f"rank {r['rank']} on {r['host']} could not check its GPU ({r['status']}: {r['detail']}). "
                            f"Check that the GPU is usable, and {LAUNCH_FIX}.")
    if share_gpu or problems:
        return problems
    by_host = {}
    for r in records:
        by_host.setdefault(r["host"], []).append(r)
    for host, rs in sorted(by_host.items()):
        groups = {}
        for r in rs:
            if r["status"] == "ok":
                groups.setdefault(("key", r["key"]), []).append(r["rank"])
            elif r["status"] == "unknown":
                groups.setdefault(("mask", _first_device(r["cvd"])), []).append(r["rank"])
        for (kind, val), ranks in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            if len(ranks) < 2:
                continue
            names = ", ".join(map(str, ranks))
            if kind == "key":
                problems.append(f"ranks {names} on {host} would all use {val}. To fix: {LAUNCH_FIX}, {SHARE_FIX}.")
            else:
                first = "an unset mask" if val is None else f"first visible device {val!r}"
                problems.append(f"ranks {names} on {host} cannot identify their GPUs (no CuPy, and torch < 2.5 or "
                                f"without CUDA) and have {first}, so nothing shows they use different GPUs. "
                                f"To fix: {LAUNCH_FIX}, {SHARE_FIX}.")
        if len(rs) < 2:
            continue
        # Incomparable reports on one host: separate buckets above are not proof of separate GPUs.
        known = sorted(r["rank"] for r in rs if r["status"] == "ok")
        unknown = sorted(r["rank"] for r in rs if r["status"] == "unknown")
        unset = sorted(r["rank"] for r in rs if r["status"] == "unknown" and r["cvd"] is None)
        kinds = sorted({_key_kind(r["key"]) for r in rs if r["status"] == "ok"})
        if known and unknown:
            problems.append(f"ranks {', '.join(map(str, known))} on {host} identified their GPU and ranks "
                            f"{', '.join(map(str, unknown))} could not, so the two cannot be compared and nothing "
                            f"shows they use different GPUs. To fix: {LAUNCH_FIX}, {SHARE_FIX}.")
        elif unset and len(unknown) >= 2 and len(unset) < len(unknown):
            problems.append(f"ranks {', '.join(map(str, unset))} on {host} cannot identify their GPU and have an "
                            f"unset mask, so they may select the same GPU as ranks "
                            f"{', '.join(str(r) for r in unknown if r not in unset)}. To fix: {LAUNCH_FIX}, {SHARE_FIX}.")
        if len(kinds) > 1:
            problems.append(f"ranks on {host} report PCI bus ids and UUIDs, which cannot be compared (CuPy in some "
                            f"ranks' env, torch only in others). To fix: {LAUNCH_FIX}, with the same env on every rank, "
                            f"or {SHARE_FIX}.")
    return problems


def placement_summary(records):
    """One line per rank for rank 0's log: host, rank, mask, and the GPU it identified."""
    return [f"  {r['host']} rank {r['rank']}: CUDA_VISIBLE_DEVICES={r['cvd']!r} -> "
            + (r["key"] if r["status"] == "ok" else f"{r['status']} ({r['detail']})")
            for r in sorted(records, key=lambda r: (r["host"], r["rank"]))]


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
    identity = None
    if not retired:
        try:
            identity = (identity_fn or gpu_identity)()
        except Exception as e:  # report it rather than leave the other ranks waiting in allgather
            identity = ("error", None, f"{type(e).__name__}: {e}")
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
    if rank == 0:
        print(f"[glint_xtc_mpi] {size} ranks; psana{args.psana}; {args.exp} run {args.run}; out {args.out}"
              + ("; --share-gpu" if args.share_gpu else ""), flush=True)
        print("\n".join(placement_summary(records)), flush=True)
        for p in problems:
            sys.stderr.write(f"[glint_xtc_mpi] {p}\n")
    if problems:
        return 2

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
