#!/usr/bin/env python3
"""Step 1 (one A100): q-frames, the cells each frame votes with, the whole-run reference cells, and
the live arrival-order decision of the shipped StreamDriver on every real dataset.

    python j5_prep.py extract    # psana1: q-frames of r0278, r0058, refined r0016 from xtc (pinned code)
    python j5_prep.py votes      # every frame's candidate cells (index_blind_nbest at warmup_nbest = 3)
    python j5_prep.py live       # StreamDriver(Mc=None).push_q in arrival order, until it decides or the run ends
    python j5_prep.py all        # the three, in order; each skips work whose output already exists

Outputs, all under $J5_OUT/prep/:
    q_<name>.npz       q-frames (q0..q{n-1}, n, events, lams) + the call that made them
    votes_<name>.npz   per-frame cells (flat, nper), voted flag, peaks per frame, index ms per frame,
                       the whole-run batch cell (consensus_cell at the shipped batch requirements)
    live_<name>.json   the live decision in arrival order + parity against the cached replay
    prep_summary.json  frame counts, the FACTS cross-checks, ms per frame (the scramble job sizes itself on it)

Every cell here is recomputed at 95d51ce from q-frames; no cached candidate pool is read.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import j5_common as C                                                       # noqa: E402

P = os.path.join(C.OUT, "prep")
GEOM = f"{C.BTX}/geom/r0016.geom"
DET = "MfxEndstation.0:Epix10ka2M.0"

# The call behind each real run's q-frames: the recipe that produced the FACTS-era inputs, run at 95d51ce.
#   r0278 / r0058: dump_candidates.py (8 Aug), the source of indexed_r0278 / indexed_r0058 / nstar_*;
#   r0016ref     : stream_find_cell.py (4 Aug), the refined-geometry r0016 q-set ("geom" tag).
EXTRACT = {
    "r0278": dict(exp="mfxl1038923", run=278, det=DET, zdist=0.060, wavelength=1.26296, min_peaks=6,
                  max_events=3000, geom=GEOM, gpu_calib=True),
    "r0058": dict(exp="mfxl1038923", run=58, det=DET, zdist=0.050, wavelength=0.0, min_peaks=6,
                  max_events=3000, geom=GEOM, gpu_calib=True),
    "r0016ref": dict(exp="mfxx49820", run=16, det=DET, zdist=0.102973, wavelength=1.290757, min_peaks=6,
                     max_events=6000, calib_dir=f"{C.GT}/calib", geom=GEOM, gpu_calib=False),
}
# q-lists that already exist (read-only inputs; q-frames, not candidate pools)
QFILES = {
    "r0016ref_aug": f"{C.GT}/q6000_geom.npz",     # refined geometry, cached 4 Aug by stream_find_cell.py: cross-check
    "r0016unref": f"{C.GT}/r0016_q6000.npz",      # psana's unrefined geometry: the q-set behind poolgate_bad_*
    "cxidb480": os.path.join(os.environ["VERIFY_INPUTS"], "q480_fix.txt"),   # cxidb-17, 480 frames (md5 d6d86c1b), the seqstop_* set
}
REAL = ("r0278", "r0058", "r0016ref", "r0016ref_aug", "r0016unref")


def qpath(name):
    return os.path.join(P, f"q_{name}.npz")


def load_frames(name):
    if name == "cxidb480":
        from glint.glint_fast import load
        fr = [np.asarray(q, float) for q in load(QFILES[name])]
        return fr, np.arange(len(fr)), dict(source=QFILES[name], md5=C.md5(QFILES[name]))
    path = qpath(name) if name in EXTRACT else QFILES[name]
    z = np.load(path, allow_pickle=False)
    n = int(z["n"]) if "n" in z.files else len([k for k in z.files if k[0] == "q" and k[1:].isdigit()])
    fr = [np.asarray(z[f"q{i}"], float) for i in range(n)]
    ev = np.asarray(z["events"]) if "events" in z.files else np.arange(n)
    return fr, ev, dict(source=path, md5=C.md5(path))


def extract_one(name):
    """One run per process: psana1 reads options such as psana.calib-dir only before the first
    DataSource, so the three extractions must not share an interpreter."""
    sys.path.insert(0, os.path.join(C.GLINT, "experiments", "xtc_bridge"))
    import xtc_qreader_psana1 as rd
    kw = EXTRACT[name]
    out = qpath(name)
    t0 = time.time()
    print(f"[extract] {name}: run_to_qframes_psana1({kw})", flush=True)
    r = rd.run_to_qframes_psana1(kw["exp"], kw["run"], kw["det"], zdist=kw["zdist"],
                                 wavelength=kw["wavelength"], min_peaks=kw["min_peaks"],
                                 max_events=kw["max_events"], calib_dir=kw.get("calib_dir"),
                                 geom=kw["geom"], gpu_calib=kw["gpu_calib"])
    fr = [np.asarray(q, np.float64) for q in r["qframes"]]
    np.savez(out + ".tmp.npz", n=len(fr), events=np.asarray(r["events"]), lams=np.asarray(r["lams"]),
             n_events=r.get("n_events", -1), call=repr(kw), commit=C.commit(),
             geom_md5=C.md5(kw["geom"]), **{f"q{i}": q for i, q in enumerate(fr)})
    os.replace(out + ".tmp.npz", out)
    print(f"[extract] {name}: {len(fr)} hit frames of {r.get('n_events')} events in "
          f"{time.time() - t0:.0f} s -> {out}", flush=True)


def extract():
    import subprocess
    for name in EXTRACT:
        if os.path.exists(qpath(name)):
            print(f"[extract] {name}: exists, skipped", flush=True)
            continue
        rc = subprocess.call([sys.executable, "-u", os.path.abspath(__file__), "extract_one", name])
        if rc:
            print(f"[extract] {name}: FAILED (exit {rc}); its later steps are skipped", flush=True)


def votes():
    from glint.glint_fast import index_blind_nbest, LYSO
    d = C.shipped_defaults()
    index_blind_nbest(np.random.default_rng(0).normal(size=(40, 3)) * 0.05, d["warmup_nbest"])   # warm-up, not kept
    for name in C.DATASETS:
        out = os.path.join(P, f"votes_{name}.npz")
        if os.path.exists(out):
            print(f"[votes] {name}: exists, skipped", flush=True)
            continue
        if name in EXTRACT and not os.path.exists(qpath(name)):
            print(f"[votes] {name}: no q-frames (extract failed?), skipped", flush=True)
            continue
        fr, ev, src = load_frames(name)
        fr, voted = C.clean(fr, d["min_peaks"])
        cells, ms = [], np.zeros(len(fr))
        t0 = time.time()
        for i, q in enumerate(fr):
            t = time.time()
            nb = index_blind_nbest(q, d["warmup_nbest"]) if voted[i] else []
            ms[i] = 1e3 * (time.time() - t)
            cells.append([np.asarray(c, float) for c, _ in nb if c is not None])
            if (i + 1) % 500 == 0:
                print(f"  [{name}] {i + 1}/{len(fr)} frames, {np.mean(ms[:i + 1]):.1f} ms/frame", flush=True)
        flat, nper = C.pack_cells(cells)
        # The whole-run cell: the batch consensus over every frame's cells at the shipped batch
        # requirements (support 3, 2% of the pool, 1.5x the runner-up; densest-neighborhood seeding),
        # the same call hybrid_index makes and the definition behind the FACTS nstar_* keys.
        Mb, sup_b = C.consensus_cell(list(flat), min_frac=C.CONSENSUS_MIN_FRAC, min_lead=C.CONSENSUS_MIN_LEAD)
        np.savez(out + ".tmp.npz", flat=flat, nper=nper, voted=voted, npk=np.array([len(q) for q in fr]),
                 ms=ms, events=ev, Mbatch=(np.zeros((3, 3)) if Mb is None else Mb), sup_batch=int(sup_b),
                 batch_found=Mb is not None, lyso=np.asarray(LYSO, float), source=src["source"],
                 source_md5=src["md5"], commit=C.commit())
        os.replace(out + ".tmp.npz", out)
        print(f"[votes] {name}: {len(fr)} frames ({int(voted.sum())} vote), {int((nper > 0).sum())} with >=1 cell, "
              f"{len(flat)} cells, {np.mean(ms[voted]):.1f} ms/frame; batch cell {C.edges(Mb)} support {sup_b} "
              f"in {time.time() - t0:.0f} s", flush=True)


def references(name, z):
    """The reference(s) a decided cell is judged against, by dataset."""
    refs = {"batch_whole_run": (z["Mbatch"] if bool(z["batch_found"]) else None)}
    if name.startswith("r0016"):
        refs["btx_sample2"] = C.cell_to_Ar(*C.read_cell_file(f"{C.BTX}/cell/sample2.cell"))
    if name == "cxidb480":
        refs["lyso_textbook"] = z["lyso"]
    return refs


def live():
    d = C.shipped_defaults()
    summary = dict(provenance=C.provenance(), rule=d, datasets={})
    for name in C.DATASETS:
        vp = os.path.join(P, f"votes_{name}.npz")
        if not os.path.exists(vp):
            continue
        z = np.load(vp, allow_pickle=False)
        per_frame = C.unpack_cells(z["flat"], z["nper"])
        voted = z["voted"]
        fr, ev, src = load_frames(name)
        fr, _ = C.clean(fr, d["min_peaks"])
        order = np.arange(len(fr))                       # arrival order = event order of the run / file order
        refs = references(name, z)
        out = os.path.join(P, f"live_{name}.json")
        t0 = time.time()
        res, live_cells = C.live_run(fr, order)
        rep = {v: C.replay(per_frame, voted, order, v, d) for v in ("shipped", "gaponly")}
        # determinism: the cells the live driver voted with against the cached ones, frame by frame
        n_cmp = n_same = 0
        for k, c in enumerate(live_cells):
            if c is None:
                continue
            n_cmp += 1
            a, b = c, per_frame[order[k]]
            n_same += int(len(a) == len(b) and all(C.same_lattice(x, y) for x, y in zip(a, b)))
        parity = dict(decided=res["decided"] == rep["shipped"]["decided"],
                      pushed=res["pushed"] == rep["shipped"]["pushed"],
                      support=res["support"] == rep["shipped"]["support"],
                      cell=(res["M"] is None and rep["shipped"]["M"] is None) or
                           (res["M"] is not None and rep["shipped"]["M"] is not None and
                            bool(C.same_lattice(res["M"], rep["shipped"]["M"]))),
                      frames_compared=n_cmp, frames_identical_cells=n_same)
        row = dict(source=src, n_frames=len(fr), n_vote=int(voted.sum()),
                   n_with_cells=int((z["nper"] > 0).sum()), n_cells=int(len(z["flat"])),
                   ms_per_frame=float(np.mean(z["ms"][voted])) if voted.any() else None,
                   batch=dict(edges=C.edges(refs["batch_whole_run"]), support=int(z["sup_batch"])),
                   live=dict({k: v for k, v in res.items() if k != "M"}, edges=C.edges(res["M"]),
                             event=int(ev[res["pushed"] - 1]) if res["decided"] else None,
                             vs={k: C.verdict_vs(res["M"], r) for k, r in refs.items()}),
                   replay={v: dict({k: x for k, x in r.items() if k != "M"}, edges=C.edges(r["M"]),
                                   vs={k: C.verdict_vs(r["M"], ref) for k, ref in refs.items()})
                           for v, r in rep.items()},
                   parity=parity, secs=time.time() - t0)
        C.dump_json(out, dict(provenance=C.provenance(), dataset=name, **row))
        summary["datasets"][name] = row
        print(f"[live] {name}: decided={res['decided']} at pushed frame {res['pushed']} "
              f"support {res['support']} runner {res['runner']} pool {res['npool']} cell {C.edges(res['M'])} "
              f"| replay shipped {rep['shipped']['decided']}@{rep['shipped']['pushed']} "
              f"gaponly {rep['gaponly']['decided']}@{rep['gaponly']['pushed']} | parity {parity}", flush=True)
    # FACTS cross-checks (inputs the same as the FACTS-era ones?), printed, never edited
    facts = dict(r0278=("indexed_r0278", 1785), r0058=("indexed_r0058", 2319),
                 r0016unref=("poolgate r0016 indexed frames / cells (check_numbers.py comment)", "2098 / 6294"))
    summary["facts_crosscheck"] = {k: dict(key=v[0], recorded=v[1],
                                           measured=(summary["datasets"].get(k, {}).get("n_with_cells"),
                                                     summary["datasets"].get(k, {}).get("n_cells")))
                                   for k, v in facts.items()}
    C.dump_json(os.path.join(P, "prep_summary.json"), summary)


if __name__ == "__main__":
    os.makedirs(P, exist_ok=True)
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step == "extract_one":
        extract_one(sys.argv[2])
        sys.exit(0)
    print(C.provenance(), flush=True)
    if step in ("extract", "all"):
        extract()
    if step in ("votes", "all"):
        votes()
    if step in ("live", "all"):
        live()
