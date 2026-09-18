#!/usr/bin/env python3
"""Record a replay of the SHIPPED StreamDriver on one or more real q-list datasets, index-only.

This is the recorder behind the streaming-replay monitors (docs/streaming_replay/). It replaces the
external harness that produced the first 480-frame trace, which had to replicate `_push_blind`
without pixels and monkeypatch `_integrate_one` to see which frame went where. The driver now has
both seams built in -- `push_q()` for peaks-in ingest and `events=True` for a per-frame outcome
record -- so nothing here reaches into the driver except through its public surface, and the per-frame
attribution comes from the driver's own cell registry (`cell`, `cell_name`).

Inputs are per-frame reciprocal vectors q (n_i, 3) in 1/A, 2pi-free (|q| = 1/d), the convention every
q list in this repository uses (glint.geom.peaks_to_q / glint.lute_bridge.peaks_to_q). Two formats:

  NAME=path.txt   FRAME-block text as written by stream2q.py (`FRAME <i> <n>` then n rows `qx qy qz`)
  NAME=path.npz   keys q_0 .. q_<n-1> (experiments/prok_q.npz)

With one input and no --schedule the frames are replayed in file order; that is the configuration
that must reproduce the published 480-frame result (see --expect). With several inputs a --schedule
JSON interleaves them into one stream with a planted per-frame species (the ground truth the monitor
shows as "schedule"), e.g.

  {"seed": 7, "exhausted": "skip",
   "scenes": [{"name": "lyso",  "length": 120, "weights": {"lyso": 1.0}},
              {"name": "blend", "length": 40,  "weights": {"lyso": 0.5, "prok": 0.5}, "hold": 4},
              {"name": "prok",  "length": 200, "weights": {"prok": 0.7, "lyso": 0.3}}]}

Per scene position t the species is redrawn from `weights` every `hold` frames (default 1) with one
seeded rng, and the next UNUSED frame of that species is taken in file order (so a dataset's own
drift is preserved and no frame is replayed twice). `exhausted`: "skip" redraws among species with
frames left and ends the scene when none has, "wrap" restarts a pool, "stop" ends the run.

Scoring is the paper's strict gate, per frame, against the frame's OWN species' reference cell:
same_lattice(M, ref) and matched_strict(M, q)/len(q) >= GATE_FRAC and matched_strict(M, q) >= GATE_MIN
(glint.glint_fast; the constants are imported, never re-declared -- test_gate_constants_dedup.py).
The header also reports `strict_ok_drv`, the same gate against the driver's final PRIMARY cell,
which is the convention the original harness used; on a single-species run the two coincide unless
the primary lock itself is wrong.

Geometry: the driver is built on a synthetic flat panel. Nothing depends on it -- q is fed directly
and index-only slots never predict -- but the constructor wants one.

  python experiments/record_stream_replay.py --input lyso=~/q480_fix.txt --B 20 --dmin 2.0 \\
      --warmup-rescue --adaptive-relock --min-inliers 10 --out replay480.json \\
      --expect 333/480 --expect-wresc 10 --expect-relock 1       # the published arm, as a gate

The gate numbers are what this recorder AND experiments/test_streamdriver_vs_offline.py ("BOTH new
mechanisms" row) print for the same arm on the same node -- A100 sdfampere036, 18 Sep 2026, bd79038:
453 indexed, 4 warm-up rescues, 10 watchdog rescues, 1 relock (frame 405, the same spurious
87.5/87.6/109.5 cell the published trace records), strict 331/453 post-lock + 2/4 warm-up-rescued
= 333/480. The published trace (docs/streaming_replay/cxidb480_strace_a100.json: 328 post-lock + 3
warm-up = 331, 6 watchdog rescues, relock at 364) was recorded at 2d6eacb on sdfampere042, before
glint#170 (live-gate pin), glint#187 (running_consensus recount) and glint#197 (symmetric
same_lattice), so its rescue/relock dynamics are not expected to match frame for frame; agreement
between this recorder and the harness on one node is the test that the peaks-in path is the
driver's own path. Regenerate the expectation from the harness when the driver changes.
"""
import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from glint.glint_fast import GATE_FRAC, GATE_MIN, LYSO, load, matched_strict   # noqa: E402
from glint.lattice import cell_params, cell_to_Ar                            # noqa: E402
from glint.multishot import same_lattice                                    # noqa: E402
import glint.stream_driver as sd                                            # noqa: E402

# The synthetic panel the original harness used (test_streamdriver_vs_offline.py). Irrelevant to
# indexing: q is injected and index-only slots are never predicted.
N_PX, PIX_MM, DIST_MM, WAVE_A = 400, 0.1, 100.0, 1.0
PANELS = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=1.0 / (PIX_MM / 1000.0),
               cx=-(N_PX / 2.0 - 0.5), cy=-(N_PX / 2.0 - 0.5), coffset=0.0,
               min_fs=0, max_fs=N_PX - 1, min_ss=0, max_ss=N_PX - 1)]
CLEN_M = DIST_MM / 1000.0

TERMINAL = ("blank", "warmup_vote", "warmup_lock", "indexed", "rescued_watchdog", "rescued_cascade",
            "miss", "gate_rejected")
RETRO = ("rescued_warmup", "rescued_relock")


# ----------------------------------------------------------------------------- inputs -----------
def load_frames(path):
    """FRAME-block .txt or q_<i>.npz -> list of (n_i, 3) float64 arrays, file order."""
    path = os.path.expanduser(path)
    if path.endswith(".npz"):
        z = np.load(path)
        keys = sorted((k for k in z.files if k.startswith("q_")), key=lambda k: int(k[2:]))
        return [np.asarray(z[k], np.float64) for k in keys]
    return [np.asarray(q, np.float64) for q in load(path)]


def digest(frames):
    """Length-aware sha256 of the frame list (the steps_sweep.py convention): q480_fix.txt -> bf3422."""
    h = hashlib.sha256()
    for q in frames:
        a = np.ascontiguousarray(np.asarray(q, np.float64))
        h.update(b"%d," % a.shape[0]); h.update(a.tobytes())
    return h.hexdigest()[:6]


def parse_ref(spec):
    """NAME=a,b,c,al,be,ga -> (NAME, 3x3 real-space basis)."""
    name, vals = spec.split("=", 1)
    p = [float(x) for x in vals.split(",")]
    if len(p) != 6:
        raise SystemExit(f"--ref {spec!r}: need 6 cell parameters")
    return name, cell_to_Ar(*p)


# ----------------------------------------------------------------------------- schedule ---------
def build_schedule(pools, sched):
    """-> list of (species, frame index within its pool) following the scene weights."""
    rng = np.random.default_rng(int(sched.get("seed", 0)))
    exhausted = sched.get("exhausted", "skip")
    cursor = {name: 0 for name in pools}
    order = []
    for scene in sched["scenes"]:
        weights = dict(scene["weights"])
        for name in weights:
            if name not in pools:
                raise SystemExit(f"schedule names species {name!r} but no --input has it")
        hold = max(1, int(scene.get("hold", 1)))
        sp = None
        for t in range(int(scene["length"])):
            if t % hold == 0 or sp is None:
                avail = {k: w for k, w in weights.items() if cursor[k] < len(pools[k]) or exhausted == "wrap"}
                if not avail:
                    if exhausted == "stop":
                        return order, cursor
                    break                                   # scene ends: no species has frames left
                names = sorted(avail); w = np.array([avail[k] for k in names], float); w /= w.sum()
                sp = names[int(rng.choice(len(names), p=w))]
            if cursor[sp] >= len(pools[sp]):
                if exhausted == "wrap":
                    cursor[sp] = 0
                else:
                    sp = None; continue
            order.append((sp, cursor[sp])); cursor[sp] += 1
    return order, cursor


# ----------------------------------------------------------------------------- scoring ----------
def strict_gate(M, q, ref):
    """The published gate against a reference lattice: (ok, matched, frac)."""
    if M is None:
        return False, 0, 0.0
    M = np.asarray(M, float)
    m = int(matched_strict(M, q)); frac = m / max(len(q), 1)
    ok = bool(same_lattice(M, ref) and frac >= GATE_FRAC and m >= GATE_MIN)
    return ok, m, frac


def git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:                                       # noqa: BLE001 -- provenance only
        return None


# ----------------------------------------------------------------------------- main -------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--ref", action="append", default=[], metavar="NAME=a,b,c,al,be,ga",
                    help="reference cell per species (also the driver's roster); default lyso = glint_fast.LYSO")
    ap.add_argument("--schedule", help="scene JSON (see module doc); default: inputs concatenated in order")
    ap.add_argument("--seed", type=int, default=None, help="override the schedule's seed")
    ap.add_argument("--limit", type=int, default=None, help="stop after N frames of the stream")
    # driver arm -- defaults are the published 480 arm minus the two opt-ins
    ap.add_argument("--B", type=int, default=20); ap.add_argument("--dmin", type=float, default=2.0)
    ap.add_argument("--tol", type=float, default=0.002); ap.add_argument("--warmup-nbest", type=int, default=3)
    ap.add_argument("--min-inliers", type=int, default=GATE_MIN); ap.add_argument("--min-inlier-frac", type=float, default=0.15)
    ap.add_argument("--warmup-rescue", action="store_true"); ap.add_argument("--adaptive-relock", action="store_true")
    ap.add_argument("--rescue-buffer", type=int, default=0); ap.add_argument("--retry-cascade", action="store_true")
    ap.add_argument("--lock-probe", action="store_true"); ap.add_argument("--cell-window", type=int, default=200)
    ap.add_argument("--stream-out", default=None); ap.add_argument("--cupy", action="store_true", help="use_gpu=True")
    ap.add_argument("--out", default="replay.json")
    # regression gate
    ap.add_argument("--expect", metavar="OK/N", help="refuse to write unless strict_ok/n matches")
    ap.add_argument("--expect-wresc", type=int, default=None); ap.add_argument("--expect-relock", type=int, default=None)
    ap.add_argument("--expect-mode", choices=("species", "drv"), default="species",
                    help="which strict count --expect gates: own-species reference (default) or the driver's primary")
    a = ap.parse_args(argv)

    # ---- inputs, references, schedule
    pools, meta_in = {}, []
    for spec in a.input:
        name, path = spec.split("=", 1)
        fr = load_frames(path)
        pools[name] = fr
        meta_in.append(dict(name=name, path=os.path.expanduser(path), n=len(fr), digest=digest(fr),
                            npk=[int(min(len(q) for q in fr)), int(np.median([len(q) for q in fr])), int(max(len(q) for q in fr))]))
    refs = {"lyso": np.asarray(LYSO, float)}
    for spec in a.ref:
        name, M = parse_ref(spec); refs[name] = M
    for name in pools:
        if name not in refs:
            raise SystemExit(f"no reference cell for species {name!r}: pass --ref {name}=a,b,c,al,be,ga")
    if a.schedule:
        sched = json.load(open(a.schedule))
        if a.seed is not None:
            sched["seed"] = a.seed
        order, cursor = build_schedule(pools, sched)
    else:
        sched = None
        order = [(name, i) for name in pools for i in range(len(pools[name]))]
    if a.limit:
        order = order[: a.limit]
    stream = [(sp, i, pools[sp][i]) for sp, i in order]
    n = len(stream)
    if n == 0:
        raise SystemExit("empty stream")

    # ---- driver
    kw = dict(B=a.B, dmin=a.dmin, tol=a.tol, warmup_nbest=a.warmup_nbest, min_inliers=a.min_inliers,
              min_inlier_frac=a.min_inlier_frac, warmup_rescue=a.warmup_rescue,
              adaptive_relock=a.adaptive_relock, rescue_buffer=a.rescue_buffer,
              retry_cascade=a.retry_cascade, lock_probe=a.lock_probe, cell_window=a.cell_window,
              roster={k: cell_params(v) for k, v in refs.items() if k in pools}, events=True)
    if a.stream_out:
        kw["stream_out"] = a.stream_out
    t0 = time.time()
    drv = sd.StreamDriver(None, PANELS, CLEN_M, WAVE_A, (N_PX, N_PX), dtype=np.uint16, use_gpu=a.cupy, **kw)

    # ---- run: push_q per frame, drain the event log after every push
    recs = []
    for i, (sp, j, q) in enumerate(stream):
        was_blind = not drv.locked
        drv.push_q(q)
        recs.append(dict(i=i, ev=i, truth=sp, src_index=j, npk=int(len(q)), wu=int(was_blind),
                         lock=int(drv.locked), flush=int((not was_blind) and drv._n == 0 and drv.locked)))
    drv.close()
    events = list(drv.events)

    # ---- consume events -> per-frame records
    by_ev = {r["ev"]: r for r in recs}
    n_term = 0
    for e in events:
        oc = e["outcome"]
        if oc == "relock":
            k = min(e["ev"], n - 1)                          # relocks fire inside a flush, at n_pushed
            by_ev[k].setdefault("relock", 0); by_ev[k]["relock"] += 1
            by_ev[k]["relock_cell"] = e["cell_name"]
            continue
        r = by_ev[e["ev"]]
        if oc in TERMINAL:
            assert "o" not in r, f"two terminal outcomes for ev {e['ev']}: {r['o']} then {oc}"
            n_term += 1
            r["o"] = oc; r["cell"] = e["cell"]; r["cell_name"] = e["cell_name"]
            r["buf"] = e["buffer"]; r["n_active"] = e["n_active"]
            if "support" in e:                               # blind-mode records carry the consensus state
                r["sup"] = e.get("support"); r["lead"] = e.get("lead")
            r["M"] = e["M"]; r["frac_live"] = e["frac"]; r["n_inl"] = e["n_inl"]
            r["wresc"] = int(oc == "rescued_watchdog")
        elif oc in RETRO:
            if e["ev"] is None:
                continue
            r["resc"] = oc; r["resc_cell"] = e["cell_name"]; r["M_retro"] = e["M"]
        else:
            raise AssertionError(f"unknown outcome {oc!r}")
    assert n_term == n, f"{n_term} terminal events for {n} frames"

    # ---- score
    Mc_final = np.asarray(drv.Mc, float) if drv.Mc is not None else None
    tot = dict(n=n, strict_ok=0, strict_ok_drv=0, blank=0, indexed=0, miss=0, gate_rejected=0, warmup=0)
    by_sp = {sp: dict(n=0, ok=0, ok_drv=0, indexed=0, miss=0, blank=0, warmup=0, rescued=0) for sp in pools}
    confusion = {}
    for r in recs:
        sp = r["truth"]; q = stream[r["i"]][2]; ref = refs[sp]
        by_sp[sp]["n"] += 1
        M = r.get("M"); Mr = r.get("M_retro")
        ok, m, frac = strict_gate(M, q, ref)
        if not ok and Mr is not None:
            ok, m, frac = strict_gate(Mr, q, ref)
        okd = strict_gate(M, q, Mc_final)[0] if Mc_final is not None else False
        if not okd and Mr is not None and Mc_final is not None:
            okd = strict_gate(Mr, q, Mc_final)[0]
        r["ok"] = int(ok); r["ok_drv"] = int(okd); r["m"] = m; r["frac"] = round(frac, 4)
        tot["strict_ok"] += ok; tot["strict_ok_drv"] += okd
        by_sp[sp]["ok"] += ok; by_sp[sp]["ok_drv"] += okd
        oc = r["o"]
        if oc in ("indexed", "rescued_watchdog", "rescued_cascade"):
            tot["indexed"] += 1; by_sp[sp]["indexed"] += 1
        elif oc == "miss":
            tot["miss"] += 1; by_sp[sp]["miss"] += 1
        elif oc == "blank":
            tot["blank"] += 1; by_sp[sp]["blank"] += 1
        elif oc == "gate_rejected":
            tot["gate_rejected"] += 1
        elif oc.startswith("warmup"):
            tot["warmup"] += 1; by_sp[sp]["warmup"] += 1
        if r.get("resc"):
            by_sp[sp]["rescued"] += 1
        cn = r.get("resc_cell") or r.get("cell_name") or "-"
        confusion.setdefault(sp, {}); confusion[sp][cn] = confusion[sp].get(cn, 0) + 1

    st = drv.stats()
    counters = dict(locked_after=st.get("locked_after"), n_relock=st.get("n_relock", 0),
                    n_watchdog_rescued=st.get("n_watchdog_rescued", 0), n_rescued=st.get("n_rescued", 0),
                    n_warmup_rescued=st.get("n_warmup_rescued", 0), n_cascade_rescued=st.get("n_cascade_rescued", 0),
                    indexed=st.get("indexed", 0), gate_rejected=st.get("gate_rejected", 0))
    # self-checks: the event log must agree with the driver's own counters
    s_wresc = sum(r["wresc"] for r in recs); s_relock = sum(r.get("relock", 0) for r in recs)
    s_rl = sum(1 for r in recs if r.get("resc") == "rescued_relock")
    s_wu = sum(1 for r in recs if r.get("resc") == "rescued_warmup")
    for name, got, want in (("watchdog rescues", s_wresc, counters["n_watchdog_rescued"]),
                            ("relocks", s_relock, counters["n_relock"]),
                            ("relock rescues", s_rl, counters["n_rescued"]),
                            ("warm-up rescues", s_wu, counters["n_warmup_rescued"]),
                            ("indexed", tot["indexed"], counters["indexed"])):
        assert got == want, f"event log disagrees with the driver on {name}: {got} vs {want}"

    header = dict(inputs=meta_in, refs={k: [round(x, 4) for x in cell_params(v)] for k, v in refs.items() if k in pools},
                  roster=kw["roster"], schedule=sched, n=n, driver_kw={k: v for k, v in kw.items() if k not in ("roster", "events")},
                  gate=dict(frac=GATE_FRAC, min_refl=GATE_MIN, lattice="same_lattice vs own species reference"),
                  totals=tot, by_species=by_sp, confusion=confusion, counters=counters,
                  cells=st.get("cells"), primary_cell=[round(x, 3) for x in cell_params(Mc_final)] if Mc_final is not None else None,
                  provenance=dict(git=git_head(), host=socket.gethostname(), python=platform.python_version(),
                                  numpy=np.__version__, use_gpu=a.cupy, argv=sys.argv[1:],
                                  elapsed_s=round(time.time() - t0, 1), timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    try:
        import torch; header["provenance"]["torch"] = torch.__version__
        if torch.cuda.is_available():
            header["provenance"]["gpu"] = torch.cuda.get_device_name(0)
    except Exception:                                        # noqa: BLE001
        pass

    key = "strict_ok" if a.expect_mode == "species" else "strict_ok_drv"
    print(f"{n} frames | strict (own species) {tot['strict_ok']}/{n} | strict (driver primary) {tot['strict_ok_drv']}/{n} | "
          f"locked_after {counters['locked_after']} | watchdog rescues {counters['n_watchdog_rescued']} | "
          f"relocks {counters['n_relock']} | relock rescues {counters['n_rescued']} | warm-up rescues {counters['n_warmup_rescued']}")
    for sp, d in by_sp.items():
        print(f"  {sp:12s} n={d['n']:5d}  strict {d['ok']:5d}  indexed {d['indexed']:5d}  miss {d['miss']:4d}  "
              f"blank {d['blank']:3d}  warmup {d['warmup']:3d}  rescued {d['rescued']:4d}   -> {confusion.get(sp)}")
    if st.get("cells"):
        for c in st["cells"]:
            print(f"  cell {c['id']} {c['name']:12s} {c['source']:7s} locked_at {c['locked_at']:5d} "
                  f"n_frames {c['n_frames']:5d} share {c['recent_share']:.2f} axes {c['axes']}")
    bad = []
    if a.expect:
        ok_e, n_e = (int(x) for x in a.expect.split("/"))
        if (tot[key], n) != (ok_e, n_e):
            bad.append(f"{key} {tot[key]}/{n} != expected {ok_e}/{n_e}")
    if a.expect_wresc is not None and counters["n_watchdog_rescued"] != a.expect_wresc:
        bad.append(f"watchdog rescues {counters['n_watchdog_rescued']} != {a.expect_wresc}")
    if a.expect_relock is not None and counters["n_relock"] != a.expect_relock:
        bad.append(f"relocks {counters['n_relock']} != {a.expect_relock}")
    if bad:
        print("REGRESSION GATE FAILED: " + "; ".join(bad))
        return 1
    with open(a.out, "w") as f:
        json.dump(dict(header=header, records=recs), f, separators=(",", ":"))
    print(f"wrote {a.out} ({os.path.getsize(a.out)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
