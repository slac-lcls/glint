"""End-to-end GPU test of StreamDriver(stream_out=...) on real synthetic pixel frames.

The CPU test (test_stream_drift.py) guards the WRITER. This guards the WIRING: that the driver
actually emits one chunk per integrated frame, that the per-frame drift/provenance fields carry the
live GeomRefiner state rather than a run-level constant, and -- the control that matters most -- that
turning the stream on changes NOTHING about the science path. A monitoring side-channel that perturbs
the merge would be worse than no side-channel.

  python test_driver_stream_out.py
"""
import os, sys, tempfile
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import cupy as cp
from glint.glint_fast import cell_to_Ar
import glint.stream_driver as sd

SIM = os.environ.get("GLINT_SIM", "/sdf/home/s/smarches/glint_sim")
imgs = np.load(f"{SIM}/images.npy")
t = np.load(f"{SIM}/truth.npz")
N = int(t["det_n"]); pix_mm = float(t["pix_mm"]); dist_mm = float(t["dist_mm"]); wave = float(t["wave_A"])
cell = t["cell"]
panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
               res=1.0 / (pix_mm / 1000.0), cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5),
               coffset=0.0, min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
Mc = cell_to_Ar(float(cell[0]), float(cell[1]), float(cell[2]), 90, 90, 90)
clen = dist_mm / 1000.0
fails = []


def check(name, cond, msg=""):
    print(f"  {name:38s}: {'PASS' if cond else 'FAIL'}   {msg}")
    if not cond:
        fails.append(name)


def run(stream_out):
    # update_every=4 (vs the 32 default) so the refiner re-solves ~9x over these 40 frames: the point
    # of per-frame stamping is that the correction MOVES, and a single-solve run cannot demonstrate it.
    drv = sd.StreamDriver(Mc, panels, clen, wave, (N, N), dtype=imgs.dtype, B=40, dmin=2.0,
                          tol=0.002, geom_refine=True, qc_frac_threshold=0.25,
                          geom_refine_kw=dict(update_every=4, min_frames=4),
                          stream_out=stream_out)
    for f in imgs:
        drv.push(f)
    drv.flush()
    cp.cuda.Stream.null.synchronize()
    s = drv.stats()
    drv.close()
    return drv, s


with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "live.stream")
    drv_on, s_on = run(path)
    drv_off, s_off = run(None)                      # CONTROL: identical run, no stream

    print(f"\nframes={len(imgs)}  indexed={s_on['indexed']}  integrated={s_on['integrated']}")

    # ---- 1. THE SIDE-CHANNEL CONTROL: science path must be bit-identical -----------------------
    # Compare key SETS, not the intersection: an intersection-only check cannot fail when a key
    # APPEARS or VANISHES, which is exactly how a stats block gets broken by an inserted branch.
    STREAM_ONLY = {"stream_out", "stream_chunks", "stream_indexed"}
    extra = (set(s_on) - set(s_off)) - STREAM_ONLY
    missing = set(s_off) - set(s_on)
    check("stream adds no stats but its own", not extra, f"unexpected extra keys: {sorted(extra)}")
    check("stream removes no stats", not missing, f"keys lost when streaming: {sorted(missing)}")
    KEYS = sorted(set(s_off) & set(s_on) - STREAM_ONLY)
    diff = [k for k in KEYS if repr(s_on[k]) != repr(s_off[k])]
    check("stream is a pure side-channel", not diff, f"differing keys: {diff}" if diff else
          f"all {len(KEYS)} shared stats identical (incl. indexed/integrated/CC/completeness)")

    # ---- 2. one chunk per integrated frame ------------------------------------------------------
    txt = open(path).read()
    nb = txt.count("----- Begin chunk -----")
    ncry = txt.count("--- Begin crystal")
    check("chunk count == n_integrated", nb == s_on["integrated"] == s_on["stream_chunks"],
          f"{nb} chunks, integrated={s_on['integrated']}, stats={s_on['stream_chunks']}")
    check("every chunk is indexed", ncry == nb == s_on["stream_indexed"], f"{ncry} crystals")
    check("chunk/crystal blocks balanced",
          txt.count("----- End chunk -----") == nb and txt.count("--- End crystal") == ncry)

    # ---- 3. per-frame drift fields present and LIVE ---------------------------------------------
    shifts, dclens, solves, fracs, cells, gens, events = [], [], [], [], [], [], []
    for ln in txt.split("\n"):
        if ln.startswith("predict_refine/det_shift"):
            p = ln.split(); shifts.append((float(p[3]), float(p[6])))
        elif ln.startswith("glint/dclen_m = "):
            dclens.append(float(ln.split(" = ")[1]))
        elif ln.startswith("glint/geom_n_solves = "):
            solves.append(int(ln.split(" = ")[1]))
        elif ln.startswith("glint/matched_frac = "):
            fracs.append(float(ln.split(" = ")[1]))
        elif ln.startswith("glint/cell_id = "):
            cells.append(int(ln.split(" = ")[1]))
        elif ln.startswith("glint/lock_generation = "):
            gens.append(int(ln.split(" = ")[1]))
        elif ln.startswith("Event: //"):
            events.append(int(ln.split("//")[1]))
    check("det_shift on every crystal", len(shifts) == ncry, f"{len(shifts)}")
    check("dclen_m on every crystal", len(dclens) == ncry, f"{len(dclens)}")
    check("matched_frac on every crystal", len(fracs) == ncry, f"{len(fracs)}")
    check("cell_id all 0 (single cell)", set(cells) == {0}, f"{sorted(set(cells))}")
    check("lock_generation all 0 (no relock)", set(gens) == {0}, f"{sorted(set(gens))}")

    # the WHOLE POINT: the geometry correction must EVOLVE across the run, not be one constant
    uniq = {s for s in shifts}
    check("det_shift VARIES across the run", len(uniq) >= 5,
          f"{len(uniq)} distinct (fs,ss) shifts -- a run-level field would collapse these to 1")
    check("early != late correction", shifts[0] != shifts[-1],
          f"first {shifts[0]} mm -> last {shifts[-1]} mm; applying the last to the first would be wrong")
    check("geom solves progress", len(set(solves)) > 1 and solves == sorted(solves),
          f"n_solves {solves[0]}..{solves[-1]}, monotone")
    final = drv_on.stats()["geom_correction"]
    mm_px = 1000.0 / panels[0]["res"]
    check("last chunk == final refiner state",
          abs(shifts[-1][0] - final["dfs"] * mm_px) < 1e-3
          and abs(shifts[-1][1] - final["dss"] * mm_px) < 1e-3
          and abs(dclens[-1] - final["dclen_m"]) < 1e-9,
          f"stream {shifts[-1]} mm vs refiner ({final['dfs']*mm_px:.3f},{final['dss']*mm_px:.3f}) mm")

    # ---- 4. provenance sanity -------------------------------------------------------------------
    check("events are the arrival indices", len(events) == nb and events == sorted(events)
          and max(events) < len(imgs), f"{events[:4]}..{events[-2:]}, max {max(events)}")
    check("matched_frac in [0,1]", all(0.0 <= f <= 1.0 for f in fracs),
          f"min {min(fracs):.3f} max {max(fracs):.3f}")
    nlow = txt.count("glint/low_confidence = 1")
    check("low_confidence matches driver count", nlow == s_on.get("n_low_confidence"),
          f"stream {nlow} vs stats {s_on.get('n_low_confidence')}")

    # ---- 4b. det_shift must be in LAB x/y, not the panel's fs/ss --------------------------------
    # GeomRefiner's (dfs,dss) lives in the panel's data-array basis; CrystFEL's det_shift is lab mm.
    # On this harness's identity panel the two coincide, which hides a wrong conversion -- so check a
    # rotated panel (fs=-y, ss=+x), routine on CSPAD/Jungfrau/epix quadrants.
    rot = [dict(panels[0], fs=np.array([0.0, -1.0, 0.0]), ss=np.array([1.0, 0.0, 0.0]))]
    d_rot = sd.StreamDriver(Mc, rot, clen, wave, (N, N), dtype=imgs.dtype, B=4, dmin=2.0,
                            stream_out=os.path.join(d, "rot.stream"))
    mm_px = 1000.0 / panels[0]["res"]
    got = d_rot._shift_basis @ np.array([1.0, 0.0])          # a pure +1 px FAST-axis error
    d_rot.close()
    check("det_shift uses the panel basis", abs(got[0]) < 1e-12 and abs(got[1] + mm_px) < 1e-12,
          f"1px fs on a (fs=-y,ss=+x) panel -> ({got[0]:.4f},{got[1]:.4f}) mm, want (0,{-mm_px:.4f})")

    # ---- 5. clen reported is the one USED (refiner is diagnostic-only, never fed back) -----------
    clens = {float(l.split(" = ")[1].split()[0]) for l in txt.split("\n")
             if l.startswith("average_camera_length")}
    check("clen reported == clen used", clens == {round(clen, 6)},
          f"{clens} vs driver clen_m {clen}; correction is separate (dclen_m), not baked in")

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
