#!/usr/bin/env python3
"""Arm (a): the shipped stopping rule on scrambled (control) data, frame by frame.

A group is a stream of G frames with no lattice in them, pushed one at a time into a default
StreamDriver(Mc=None) (push_q, blind mode), exactly as a real run would be. A decision in a group is a
false decision. The frames are real frames of r0278, r0058 and refined r0016, each peak rotated by its
own random azimuth about the beam (glint.multilattice.scramble_azimuth, the function
experiments/scramble_qlist.py uses): |q| and q_z, and so each peak's excitation error and the radial
structure, are kept; only the lattice coherence is destroyed.

    python stop_scramble.py TASK NTASKS [--G 100] [--groups 3333,3333,3334] [--long 20] [--hours 5.4] [--procs 2]

Group k of run r draws its G frames without replacement with default_rng([SEED, 1, r, k]) and scrambles
frame p with default_rng([SEED, 2, r, k, p]); groups never share a scramble. Task t takes the groups
with k % NTASKS == t. "Long" groups push a whole run (all its frames, permuted with
default_rng([SEED, 3, r, l]), scrambled with default_rng([SEED, 4, r, l, p])) to see whether the rule
ever decides on a full run of scrambled frames.

Every group's per-frame cells are kept (float32), so stop_replay.py can apply the support-and-lead-only
rule and the batch requirements to the same groups without indexing again. Work is checkpointed in
chunks of 25 groups; a rerun of the same task skips finished chunks, and the task stops cleanly before
--hours so a preemption or the walltime loses at most one chunk.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stop_common as C                                                       # noqa: E402
from glint.multilattice import scramble_azimuth                              # noqa: E402

RUNS = ("r0278", "r0058", "r0016ref")       # run id r = position in this tuple
CHUNK = 25


def load_run(name):
    """q-frames and references for one source run, from stop_prep.py's outputs (refined r0016 falls back
    to the 4 Aug refined cache if the re-extraction is missing; the result records which was used)."""
    P = os.path.join(C.OUT, "prep")
    src = name
    if not os.path.exists(os.path.join(P, f"votes_{name}.npz")) and name == "r0016ref":
        src = "r0016ref_aug"
    z = np.load(os.path.join(P, f"votes_{src}.npz"), allow_pickle=False)
    import stop_prep
    fr, _, _ = stop_prep.load_frames(src)
    d = C.shipped_defaults()
    fr, voted = C.clean(fr, d["min_peaks"])
    keep = np.nonzero(voted)[0]                    # frames that vote (>= min_peaks usable rows)
    refs = stop_prep.references(src, z)
    return src, [fr[i] for i in keep], refs


def run_group(frames, refs, index_rest):
    res, cells = C.live_run(frames, range(len(frames)))
    n_live = len(cells)
    if res["decided"] and n_live < len(frames):    # finish the group's cells for the replays
        for q in frames[n_live:]:
            cells.append([np.asarray(c, float) for c, _ in index_rest(q, 3) if c is not None])
    row = dict(decided=res["decided"], pushed=res["pushed"], npool=res["npool"], support=res["support"],
               runner=res["runner"], edges=C.edges(res["M"]), index_s=res["index_s"],
               vs={k: C.verdict_vs(res["M"], r) for k, r in refs.items()})
    return row, cells


def work(a, units, wid, t_start):
    """Run this worker's share of the units; each unit is one checkpoint file."""
    from glint.glint_fast import index_blind_nbest
    out = os.path.join(C.OUT, "scramble", f"task{a.task}")
    data = {r: load_run(name) for r, name in enumerate(RUNS)}
    t_unit = []
    for kind, r, c, ks in units:
        path = os.path.join(out, f"{kind}_r{r}_{c:04d}.npz")
        if os.path.exists(path):
            continue
        left = a.hours * 3600 - (time.time() - t_start)
        need = (np.median(t_unit) if t_unit else 0) * (1 if kind == "main" else 1.5) + 120
        if left < need:
            print(f"[scramble w{wid}] stopping before {kind} r{r} {c}: {left:.0f} s left, ~{need:.0f} s needed",
                  flush=True)
            break
        t0 = time.time()
        src, fr, refs = data[r]
        rows, flat, nper = [], [], []
        for k in ks:
            if kind == "main":
                idx = np.random.default_rng([C.SEED, 1, r, k]).choice(len(fr), a.G, replace=False)
                frames = [scramble_azimuth(fr[i], np.random.default_rng([C.SEED, 2, r, k, p]))
                          for p, i in enumerate(idx)]
            else:
                idx = np.random.default_rng([C.SEED, 3, r, k]).permutation(len(fr))
                frames = [scramble_azimuth(fr[i], np.random.default_rng([C.SEED, 4, r, k, p]))
                          for p, i in enumerate(idx)]
            row, cells = run_group(frames, refs, index_blind_nbest)
            row.update(kind=kind, run=RUNS[r], run_frames_from=src, group=int(k), G=len(frames))
            rows.append(row)
            f, n = C.pack_cells(cells)
            flat.append(f.astype(np.float32)); nper.append(n)
            if row["decided"]:
                print(f"  FALSE DECISION {kind} {RUNS[r]} group {k}: frame {row['pushed']} support "
                      f"{row['support']} runner {row['runner']} pool {row['npool']} cell {row['edges']} {row['vs']}",
                      flush=True)
        np.savez(path + ".tmp.npz", flat=np.concatenate(flat), nper=np.concatenate(nper),
                 G=np.array([r_["G"] for r_ in rows]), rows=np.array(json.dumps(rows)))
        os.replace(path + ".tmp.npz", path)
        dt = time.time() - t0
        if kind == "main":
            t_unit.append(dt)
        nfr = sum(r_["G"] for r_ in rows)
        print(f"[scramble w{wid}] {kind} {RUNS[r]} chunk {c}: {len(rows)} groups, {nfr} frames, "
              f"{sum(r_['decided'] for r_ in rows)} decided, {dt:.0f} s ({1e3 * dt / nfr:.1f} ms/frame)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=int)
    ap.add_argument("ntasks", type=int)
    ap.add_argument("--G", type=int, default=100)
    ap.add_argument("--groups", default="3333,3333,3334", help="groups per run, in RUNS order (sum 10,000)")
    ap.add_argument("--long", type=int, default=20, help="whole-run groups per run")
    ap.add_argument("--hours", type=float, default=5.4)
    ap.add_argument("--procs", type=int, default=2, help="worker processes sharing the one GPU")
    a = ap.parse_args()
    t_start = time.time()
    d = C.shipped_defaults()
    assert d["warmup_nbest"] == 3, d
    out = os.path.join(C.OUT, "scramble", f"task{a.task}")
    os.makedirs(out, exist_ok=True)
    C.dump_json(os.path.join(out, "provenance.json"), dict(provenance=C.provenance(), args=vars(a), runs=RUNS))
    ngroups = [int(x) for x in a.groups.split(",")]

    # work list: main chunks interleaved across runs, then the long groups
    mine = {r: [k for k in range(ngroups[r]) if k % a.ntasks == a.task] for r in range(len(RUNS))}
    units = []
    for c in range(max((len(v) + CHUNK - 1) // CHUNK for v in mine.values())):
        for r in range(len(RUNS)):
            ks = mine[r][c * CHUNK:(c + 1) * CHUNK]
            if ks:
                units.append(("main", r, c, ks))
    for r in range(len(RUNS)):
        for l in range(a.long):
            if l % a.ntasks == a.task:
                units.append(("long", r, l, [l]))
    print(f"[scramble] task {a.task}/{a.ntasks}: {len(units)} units, {a.procs} workers, G={a.G}, "
          f"groups per run {ngroups}, long {a.long} per run", flush=True)
    if a.procs <= 1:
        work(a, units, 0, t_start)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")                  # one CUDA context per worker
        ps = [ctx.Process(target=work, args=(a, units[w::a.procs], w, t_start)) for w in range(a.procs)]
        for p_ in ps:
            p_.start()
        for p_ in ps:
            p_.join()
    done = sum(os.path.exists(os.path.join(out, f"{k}_r{r}_{c:04d}.npz")) for k, r, c, _ in units)
    C.dump_json(os.path.join(out, "status.json"), dict(units=len(units), done=done,
                                                       secs=time.time() - t_start))
    print(f"[scramble] task {a.task}: {done}/{len(units)} units done", flush=True)


if __name__ == "__main__":
    main()
