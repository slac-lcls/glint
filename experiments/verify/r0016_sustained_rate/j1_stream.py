#!/usr/bin/env python3
"""Sustained frames per second of the shipped StreamDriver on one real run, with
the strict-bar yield measured on the same pass. One A100.

Run: mfxx49820 r0016 (Epix10ka2M, lysozyme), every event of the run in arrival order (psana1
sequential mode), the refined btx geometry results/btx/geom/r0016.geom. The distance (0.102973 m),
wavelength (1.290757 A, fixed: ebeamPhotonEnergy is inf on this run), calib-dir override and pixel
mask are the ones of the validated `--geom` run in experiments/xtc_bridge/README.md (blind cell
38.3 / 79.1 / 80.3 A against sample2.cell 38.4 / 79.3 / 79.5).

Reading and calibration happen OUTSIDE the timed sections. Events are read with psana1 and calibrated
on the GPU (xtc_bridge/gpu_calib.py, bit-exact to det.calib on this run; re-checked on the first
--check-calib events here), copied back to host memory in chunks of --chunk frames, and only then
pushed. Each chunk is pushed, frame by frame, through every arm's StreamDriver in turn (the arm order
rotates by one per chunk). Only push() calls and the final close() are timed, each timed section
bracketed by a device synchronize. Frames therefore enter the driver from host memory, as from a
DAQ or network receiver, and the H2D upload is inside the timed section, as in stream_ms.

Arms (each a fresh StreamDriver over the same events):
  defaults     StreamDriver(None, ...) at its shipped defaults; per-frame event log on, for scoring
  events_off   the same with the event log off; bounds what the log costs
  wrescue      shipped opt-in warmup_rescue=True: frames seen before the lock are indexed with the
               cell once it is established (counted; not integrated)
  wrescue_pix  shipped opt-ins warmup_rescue=True + rescue_pixels=N: those frames are also integrated
  known        control: the reference cell given from frame 0 (no blind phase), the configuration the
               simulated stream_fps (275 f/s, FACTS) was measured in
Non-default constructor arguments shared by every arm, all required by real calibrated Epix10ka2M data:
dtype=float32 (calibrated frames; the uint16 default refuses them), mask = the psana pixel mask, and
pf_kw = xtc_core's Epix10ka2M peak-finder constants (PeakFinderV4's library defaults call 100% of
this detector's frames hits; xtc_bridge/README.md "Tuning the peak finder").

Before the measured arms, one throwaway driver is run over the first chunk to compile kernels (JIT,
CUDA graphs). Its timing is recorded as the cold start; it is not part of any arm.

Scoring is the strict gate, imported from experiments/record_stream_replay.py (which imports
GATE_FRAC, GATE_MIN, matched_strict from glint.glint_fast): same_lattice(M, ref) AND
matched_strict(M, q) >= 10 AND matched_strict(M, q) / len(q) >= 0.25, with q the frame's own peaks as
the driver computed them. ref = btx sample2.cell.

Outputs (in --out-dir, tagged with --tag): j1_<tag>.json (summary, provenance, per-arm timing and
yields), j1_<tag>_frames.npz (per-frame records per arm), j1_<tag>_q.npz and j1_<tag>_qscr.npz (the
defaults arm's per-frame q, every event in order, blanks empty; and its azimuth-scrambled copy, made
with experiments/scramble_qlist.scramble_frames at seed 20260928) for the null replay.
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

os.environ.setdefault("OMP_NUM_THREADS", "1")   # the stream_ms protocol (remeasure_stream_ms.py)
import numpy as np


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glint", required=True, help="pinned GLINT tree (95d51ce)")
    ap.add_argument("--exp", default="mfxx49820")
    ap.add_argument("--run", type=int, default=16)
    ap.add_argument("--det", default="MfxEndstation.0:Epix10ka2M.0")
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--geom", required=True)
    ap.add_argument("--zdist", type=float, default=0.102973, help="sample-detector distance (m), total")
    ap.add_argument("--wavelength", type=float, default=1.290757)
    ap.add_argument("--cell", default="79.327,79.461,38.406,90.111,89.981,90.083",
                    help="reference cell (btx sample2.cell): strict-gate reference and the known arm's Mc")
    ap.add_argument("--arms", default="defaults,events_off,wrescue,wrescue_pix,known")
    ap.add_argument("--rescue-pixels", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=1024, help="frames held in host memory per timed section")
    ap.add_argument("--max-events", type=int, default=0, help="0 = the whole run")
    ap.add_argument("--warm-events", type=int, default=1024, help="frames of the throwaway JIT pass")
    ap.add_argument("--check-calib", type=int, default=8, help="events compared gpu_calib vs det.calib")
    ap.add_argument("--calib", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--window", type=int, default=2048, help="frames per window of the rate time series")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU only (use_gpu=False, det.calib), a few events, no timing claims")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    t_start = time.time()
    G = os.path.abspath(os.path.expanduser(a.glint))
    sys.path.insert(0, G)
    sys.path.insert(0, os.path.join(G, "experiments"))
    sys.path.insert(0, os.path.join(G, "experiments", "xtc_bridge"))
    os.makedirs(a.out_dir, exist_ok=True)

    import glint.stream_driver as sd
    from glint.lattice import cell_params, cell_to_Ar
    from glint.lute_bridge import parse_geom
    from glint.multishot import same_lattice
    from record_stream_replay import ACCEPTED, RETRO, TERMINAL, strict_gate   # the shipped scorer
    from glint.glint_fast import GATE_FRAC, GATE_MIN, GATE_TOL                 # the bar's constants, as shipped
    from scramble_qlist import scramble_frames                                 # the shipped null
    import xtc_core

    gpu = not a.dry_run
    if gpu:
        import cupy as cp

        def sync():
            cp.cuda.Device().synchronize()
    else:
        cp = None

        def sync():
            pass

    commit = open(os.path.join(G, "COMMIT")).read().strip() if os.path.exists(os.path.join(G, "COMMIT")) else None
    ref6 = [float(x) for x in a.cell.split(",")]
    REF = np.asarray(cell_to_Ar(*ref6), float)

    # ---------------------------------------------------------------- geometry
    panels, gl = parse_geom(os.path.expanduser(a.geom))
    coffs = sorted({round(float(p["coffset"]), 12) for p in panels})
    if len(coffs) != 1:
        raise SystemExit(f"panel coffsets differ ({coffs}); one clen cannot reproduce --zdist")
    clen_m = a.zdist - coffs[0]                    # driver: z = clen + coffset; the reader used z = zdist
    shape = (max(p["max_ss"] for p in panels) + 1, max(p["max_fs"] for p in panels) + 1)
    geom_md5 = hashlib.md5(open(os.path.expanduser(a.geom), "rb").read()).hexdigest()

    # ---------------------------------------------------------------- psana
    import psana
    psana.setOption("psana.calib-dir", os.path.expanduser(a.calib_dir))
    ds = psana.DataSource(f"exp={a.exp}:run={a.run}")
    det = psana.Detector(a.det)
    gcal = None
    if a.calib == "gpu" and gpu:
        import gpu_calib
        gcal = gpu_calib.GpuCalibrator(det, int(a.run))

    PF_KW = dict(min_pix=xtc_core.PF_MIN_PIX, son_min=xtc_core.PF_SON_MIN,
                 thr_high=xtc_core.PF_THR_HIGH, thr_low=xtc_core.PF_THR_LOW)
    read = dict(n_seen=0, n_none=0, s_read=0.0, src_dtype=None, calib_check=[])
    good = None

    def frames_iter():
        """(psana event index, float32 slab frame), arrival order. Reading and calibration only."""
        nonlocal good
        for i, evt in enumerate(ds.events()):
            if a.max_events and i >= a.max_events:
                break
            read["n_seen"] += 1
            t0 = time.perf_counter()
            if gcal is not None:
                raw = det.raw(evt)
                fr = None if raw is None else cp.asnumpy(gcal(raw))
            else:
                fr = det.calib(evt)
            if fr is None:
                read["n_none"] += 1
                read["s_read"] += time.perf_counter() - t0
                continue
            fr = np.asarray(fr)
            if read["src_dtype"] is None:
                read["src_dtype"] = str(fr.dtype)
                if fr.dtype != np.float32:
                    raise SystemExit(f"calibrated frames are {fr.dtype}, not float32: refusing to recast silently")
            if good is None:
                m = det.mask(evt, status=True, calib=True, edges=False, central=False)   # the reader's mask
                good = (np.ones(shape, bool) if m is None else np.asarray(m).astype(bool).reshape(shape))
            if gcal is not None and len(read["calib_check"]) < a.check_calib:
                ref_fr = np.asarray(det.calib(evt), np.float32).reshape(shape)
                read["calib_check"].append(float(np.abs(ref_fr - fr.reshape(shape)).max()))
                if read["calib_check"][-1] > 0.01:          # validated bit-exact (0.000 ADU, job 34277115)
                    raise SystemExit(f"gpu_calib differs from det.calib by {read['calib_check'][-1]:.4g} ADU "
                                     f"on event {i}: stopping before any timing")
            read["s_read"] += time.perf_counter() - t0
            yield i, fr.reshape(shape)

    def chunk_iter():
        buf = np.empty((a.chunk,) + shape, np.float32)
        evs = []
        for i, fr in frames_iter():
            buf[len(evs)] = fr
            evs.append(i)
            if len(evs) == a.chunk:
                yield evs, buf
                evs = []
        if evs:
            yield evs, buf[:len(evs)]

    # ---------------------------------------------------------------- arms
    specs = {
        "defaults": dict(blind=True, events=True, kw={}),
        "events_off": dict(blind=True, events=False, kw={}),
        "wrescue": dict(blind=True, events=True, kw=dict(warmup_rescue=True)),
        "wrescue_pix": dict(blind=True, events=True, kw=dict(warmup_rescue=True, rescue_pixels=int(a.rescue_pixels))),
        "known": dict(blind=False, events=True, kw={}),
    }
    names = [s.strip() for s in a.arms.split(",") if s.strip()]
    for n in names:
        if n not in specs:
            raise SystemExit(f"unknown arm {n!r}; known: {sorted(specs)}")

    def make_driver(spec):
        kw = dict(dtype=np.float32, mask=good, pf_kw=dict(PF_KW), use_gpu=gpu, events=spec["events"])
        kw.update(spec["kw"])
        drv = sd.StreamDriver(REF if not spec["blind"] else None, panels, clen_m, a.wavelength, shape, **kw)
        if spec["events"]:
            drv.events_keep_q = True              # the scorer needs each frame's q (record_stream_replay does the same)
        return drv

    class Arm:
        def __init__(self, name):
            self.name, self.spec = name, specs[name]
            self.drv = make_driver(self.spec)
            self.tcum = []                        # arm clock (timed seconds only) at the return of each push
            self.clock = 0.0
            self.lock_i = 0 if not self.spec["blind"] else None
            self.n = 0
            self.t_close = None

        def run_chunk(self, frames):
            drv, tc, base = self.drv, self.tcum, self.clock
            locked = drv.locked
            sync()
            t0 = time.perf_counter()
            for j in range(len(frames)):
                drv.push(frames[j])
                tc.append(base + (time.perf_counter() - t0))
                if not locked and drv.locked:
                    locked = True
                    self.lock_i = self.n + j
            sync()
            self.clock = base + (time.perf_counter() - t0)
            self.n += len(frames)

        def finish(self):
            sync()
            t0 = time.perf_counter()
            self.drv.close()
            sync()
            self.t_close = time.perf_counter() - t0
            self.clock += self.t_close

    # ---------------------------------------------------------------- run
    arms = None
    cold = None
    all_evs = []
    t_run0 = time.time()
    for c, (evs, frames) in enumerate(chunk_iter()):
        if arms is None:
            # throwaway JIT pass over the first frames, then the measured arms, built fresh
            w = frames[: max(1, min(a.warm_events, len(frames)))]
            wd = make_driver(specs["events_off"])      # shipped defaults, no event log; not scored
            sync(); t0 = time.perf_counter()
            for f in w:
                wd.push(f)
            wd.close(); sync()
            cold = dict(frames=len(w), seconds=time.perf_counter() - t0, locked=bool(wd.locked),
                        locked_after=(wd.stats().get("locked_after") if wd.locked else None))
            del wd
            if gpu:
                cp.get_default_memory_pool().free_all_blocks()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            arms = [Arm(n) for n in names]
        order = arms[c % len(arms):] + arms[:c % len(arms)]
        for arm in order:
            arm.run_chunk(frames)
        all_evs.extend(evs)
        print(f"chunk {c}: {len(all_evs)} events, " + " ".join(
            f"{arm.name}={arm.n / max(arm.clock, 1e-9):.1f}f/s" for arm in arms), flush=True)
    if arms is None:
        raise SystemExit("no frames read")
    for arm in arms:
        arm.finish()
    t_run = time.time() - t_run0
    N = len(all_evs)

    # ---------------------------------------------------------------- score
    out_arms = {}
    frames_npz = {"psana_event": np.asarray(all_evs, np.int64)}
    hit_ref = None
    q_ref = None
    for arm in arms:
        drv, B = arm.drv, arm.drv.B
        st = drv.stats()
        tc = np.asarray(arm.tcum)
        rec = dict(name=arm.name, driver_kw={k: v for k, v in dict(arm.spec["kw"]).items()},
                   blind_start=arm.spec["blind"], events=arm.spec["events"], B=B, n=N,
                   seconds_timed=arm.clock, seconds_close=arm.t_close,
                   locked=bool(drv.locked), lock_i=arm.lock_i,
                   locked_after=st.get("locked_after"), consensus_support=st.get("consensus_support"),
                   counters={k: st.get(k) for k in ("indexed", "integrated", "gate_rejected", "n_warmup_rescued",
                                                     "n_warmup_integrated", "n_relock", "flush_errors",
                                                     "n_fanout_errors", "pixels_evicted") if k in st},
                   cell=([round(x, 3) for x in cell_params(np.asarray(drv.Mc, float))] if drv.Mc is not None else None),
                   cell_same_as_ref=(bool(same_lattice(np.asarray(drv.Mc, float), REF)) if drv.Mc is not None else None),
                   finder={k: (float(v) if isinstance(v, float) else v) for k, v in drv.finder.p.items()}
                   if hasattr(drv.finder, "p") else None)
        # steady state: from one full ring (B pushes) after the lock to the end of the run
        s0 = None if arm.lock_i is None else min(N, arm.lock_i + 1 + B)
        if s0 is not None and s0 < N:
            t_s0 = tc[s0 - 1] if s0 > 0 else 0.0
            rec["steady"] = dict(first=int(s0), frames=int(N - s0), seconds=float(arm.clock - t_s0),
                                 fps=float((N - s0) / max(arm.clock - t_s0, 1e-12)))
            wins = []
            for s in range(s0, N, a.window):
                e = min(N, s + a.window)
                if e - s < a.window // 2:
                    break
                t_e = tc[e - 1] if e < N else arm.clock
                t_s = tc[s - 1] if s > 0 else 0.0
                wins.append(dict(first=s, frames=e - s, fps=float((e - s) / max(t_e - t_s, 1e-12))))
            rec["windows"] = wins
        rec["whole_run_fps"] = float(N / max(arm.clock, 1e-12))
        frames_npz[f"{arm.name}_tcum"] = tc.astype(np.float64)

        # Scoring runs after an hour of timed work, so a failure in it must not lose the timing: an
        # exception here is recorded in the arm's JSON record, and the raw event log is pickled next to
        # the outputs so the arm can be re-scored offline.
        if arm.spec["events"]:
            try:
                recs = {}
                n_relock = 0
                n_int = np.zeros(N, np.int8)
                for e in drv.events:
                    oc = e["outcome"]
                    if oc == "relock":
                        n_relock += 1
                        continue
                    if oc == "integrated":
                        if e.get("ev") is not None and 0 <= e["ev"] < N:
                            n_int[e["ev"]] = 1
                        continue
                    if oc == "effort":
                        continue
                    ev = e["ev"]
                    r = recs.setdefault(ev, {})
                    if oc in TERMINAL:
                        if "o" in r:
                            raise RuntimeError(f"{arm.name}: two terminal outcomes for frame {ev}")
                        r["o"] = oc; r["M"] = e.get("M"); r["q"] = e.get("q"); r["npk"] = int(e.get("n_peaks") or 0)
                    elif oc in RETRO:
                        if ev is not None:
                            r["M_retro"] = e.get("M")
                    else:
                        raise RuntimeError(f"{arm.name}: unknown outcome {oc!r}")
                if len([k for k in recs if "o" in recs[k]]) != N:
                    raise RuntimeError(f"{arm.name}: {len(recs)} terminal records for {N} frames")
                Mc_final = np.asarray(drv.Mc, float) if drv.Mc is not None else None
                oc_codes = {o: k for k, o in enumerate(TERMINAL)}
                o_arr = np.zeros(N, np.int8); npk = np.zeros(N, np.int32)
                ok = np.zeros(N, np.int8); ok_drv = np.zeros(N, np.int8); acc = np.zeros(N, np.int8)
                retro = np.zeros(N, np.int8)
                qs = []
                for i in range(N):
                    r = recs[i]
                    o_arr[i] = oc_codes[r["o"]]; npk[i] = r["npk"]
                    q = np.asarray(r["q"], float).reshape(-1, 3) if (r.get("q") is not None and r["o"] != "blank") else np.zeros((0, 3))
                    qs.append(q)
                    good_ = False
                    for M in (r.get("M"), r.get("M_retro")):
                        if M is not None and len(q):
                            if strict_gate(np.asarray(M, float), q, REF)[0]:
                                good_ = True
                            if Mc_final is not None and strict_gate(np.asarray(M, float), q, Mc_final)[0]:
                                ok_drv[i] = 1
                    ok[i] = int(good_)
                    acc[i] = int(r["o"] in ACCEPTED or r.get("M_retro") is not None)
                    retro[i] = int(r.get("M_retro") is not None)
                hit = (o_arr != oc_codes["blank"])
                if hit_ref is None:
                    hit_ref = hit.copy()
                rec["hits_match_first_scored_arm"] = bool(np.array_equal(hit, hit_ref))
                nh = int(hit.sum())
                # hits with at least GATE_MIN peaks: the only frames the bar's count floor lets through, so the
                # yield is also given over them (a 6-9-peak hit can never pass; the denominator is stated)
                hit10 = hit & (npk >= GATE_MIN)
                nh10 = int(hit10.sum())
                rec["yield"] = dict(
                    hits=nh, blanks=int(N - nh), hit_fraction=nh / N,
                    hits_ge_gate_min=nh10, strict_of_hits_ge_gate_min=(float(ok[hit10].sum()) / nh10 if nh10 else None),
                    npk_hits=[int(np.percentile(npk[hit], p)) for p in (10, 50, 90)] if nh else None,
                    strict=int(ok.sum()), strict_of_hits=(float(ok.sum()) / nh if nh else None),
                    strict_vs_driver_cell=int(ok_drv.sum()),
                    live_accepted=int(acc.sum()), warmup_rescued=int(retro.sum()),
                    integrated=int(n_int.sum()), relocks=n_relock,
                    outcomes={o: int((o_arr == k).sum()) for o, k in oc_codes.items() if (o_arr == k).any()})
                if "steady" in rec:
                    s0 = rec["steady"]["first"]
                    hs = hit[s0:]
                    rec["steady"].update(hits=int(hs.sum()), hits_per_s=float(hs.sum() / max(rec["steady"]["seconds"], 1e-12)),
                                         hits_ge_gate_min=int(hit10[s0:].sum()),
                                         strict=int(ok[s0:].sum()),
                                         strict_of_hits=(float(ok[s0:].sum()) / hs.sum() if hs.sum() else None))
                    for wdw in rec.get("windows", []):
                        sl = slice(wdw["first"], wdw["first"] + wdw["frames"])
                        wdw.update(hits=int(hit[sl].sum()), strict=int(ok[sl].sum()))
                if arm.lock_i is not None and arm.spec["blind"]:
                    pre = slice(0, arm.lock_i + 1)
                    rec["before_lock"] = dict(frames=arm.lock_i + 1, hits=int(hit[pre].sum()), strict=int(ok[pre].sum()))
                for key, arr in (("outcome", o_arr), ("npk", npk), ("strict", ok), ("strict_drv", ok_drv),
                                 ("live", acc), ("retro", retro), ("integrated", n_int)):
                    frames_npz[f"{arm.name}_{key}"] = arr
                if arm.name == "defaults":
                    q_ref = qs
            except Exception as exc:                  # noqa: BLE001 -- keep the timing, report the failure
                import pickle
                import traceback
                rec["scoring_error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                dump = os.path.join(a.out_dir, f"j1_{a.tag}_{arm.name}_events.pkl")
                try:
                    with open(dump, "wb") as fh:
                        pickle.dump(dict(events=drv.events, N=N, psana_event=all_evs), fh, protocol=4)
                    rec["events_dump"] = dump
                except Exception as exc2:             # noqa: BLE001
                    rec["events_dump_error"] = repr(exc2)
                print(f"WARNING {arm.name}: scoring failed ({rec['scoring_error']}); events -> {rec.get('events_dump')}",
                      flush=True)
        out_arms[arm.name] = rec

    # hits/s for the events_off arm, from the scored arms' hit flags (same frames, same finder)
    if hit_ref is not None:
        for name, rec in out_arms.items():
            if not rec["events"] and "steady" in rec:
                s0 = rec["steady"]["first"]
                rec["steady"]["hits"] = int(hit_ref[s0:].sum())
                rec["steady"]["hits_per_s"] = float(hit_ref[s0:].sum() / max(rec["steady"]["seconds"], 1e-12))
                rec["steady"]["hits_from"] = "the scored arms' per-frame hit flags"

    # ---------------------------------------------------------------- the null's inputs
    if q_ref is not None:
        qz = {f"q_{i}": q for i, q in enumerate(q_ref)}
        np.savez(os.path.join(a.out_dir, f"j1_{a.tag}_q.npz"), **qz)
        scr = scramble_frames(q_ref, 20260928)
        np.savez(os.path.join(a.out_dir, f"j1_{a.tag}_qscr.npz"), **{f"q_{i}": q for i, q in enumerate(scr)})
    np.savez_compressed(os.path.join(a.out_dir, f"j1_{a.tag}_frames.npz"), **frames_npz)

    # ---------------------------------------------------------------- provenance
    prov = dict(argv=sys.argv, commit_file=commit, glint=G, host=socket.gethostname(),
                slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
                conda_env=os.environ.get("CONDA_DEFAULT_ENV"), python=platform.python_version(),
                numpy=np.__version__, omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), elapsed_s=round(time.time() - t_start, 1),
                crystfel="not used by this measurement")
    try:
        import torch
        prov["torch"] = torch.__version__
        if torch.cuda.is_available():
            prov["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if gpu:
        prov["cupy"] = cp.__version__
        prov["cupy_pool_total_bytes"] = int(cp.get_default_memory_pool().total_bytes())
        try:
            import torch
            prov["torch_max_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
        try:
            prov["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm", "--format=csv,noheader"],
                text=True).strip()
        except Exception:
            pass
    summary = dict(
        question="sustained frames per second through the shipped StreamDriver on a real multi-thousand-frame "
                 "run, with the strict-bar yield on the same pass",
        run=dict(exp=a.exp, run=a.run, det=a.det, events_read=read["n_seen"], events_pushed=N,
                 events_without_frame=read["n_none"], read_and_calib_s=round(read["s_read"], 1),
                 calib=a.calib, calib_check_max_abs_adu=read["calib_check"], src_dtype=read["src_dtype"]),
        geometry=dict(geom=os.path.expanduser(a.geom), md5=geom_md5, n_panels=len(panels), shape=list(shape),
                      coffset_m=coffs[0], clen_m=clen_m, zdist_m=a.zdist, wavelength_A=a.wavelength,
                      calib_dir=os.path.expanduser(a.calib_dir),
                      mask_good_fraction=(float(good.mean()) if good is not None else None)),
        pf_kw=PF_KW, reference_cell=ref6,
        bar=dict(name="strict correct-lattice bar", definition="same_lattice(M, ref) and matched_strict(M,q) >= GATE_MIN "
                 "and matched_strict(M,q)/len(q) >= GATE_FRAC; matched_strict: |q@M - round(q@M)| < GATE_TOL componentwise",
                 GATE_MIN=int(GATE_MIN), GATE_FRAC=float(GATE_FRAC), GATE_TOL=float(GATE_TOL),
                 scorer="experiments/record_stream_replay.strict_gate"),
        cold_start=cold, chunk=a.chunk, run_wall_s=round(t_run, 1), arms=out_arms, provenance=prov)
    out = os.path.join(a.out_dir, f"j1_{a.tag}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    for name, rec in out_arms.items():
        st = rec.get("steady", {})
        y = rec.get("yield", {})
        print(f"{name:12s} steady {st.get('fps', float('nan')):8.1f} frames/s  {st.get('hits_per_s', float('nan')):8.1f} hits/s"
              f"  whole run {rec['whole_run_fps']:8.1f} f/s | strict {y.get('strict')} of {y.get('hits')} hits"
              f"  lock at frame {rec['lock_i']}  cell {rec['cell']} same_as_ref {rec['cell_same_as_ref']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
