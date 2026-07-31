"""Guard the .stream drift/provenance wiring: (1) a record with NO new fields must produce output
byte-identical to the pre-change writer, so every existing caller is untouched; (2) the new per-frame
fields must actually appear, with correct values and units; (3) StreamWriter's incremental output must
be byte-identical to write_stream_integrated's batch output on the same records, so the live and
offline paths cannot drift apart; (4) the result must still parse as a CrystFEL stream.

CPU-only, no GPU, no real data.  python test_stream_drift.py   # exit 0 = all pass
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from glint.lattice import cell_to_Ar
from glint.predict import write_stream_integrated, StreamWriter, predict_spots, integrate_spots

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
LAM = 1.322
NPX = 1500
fails = []


def check(name, cond, msg=""):
    print(f"  {name:34s}: {'PASS' if cond else 'FAIL'}   {msg}")
    if not cond:
        fails.append(name)


# ---- the exact pre-change chunk text, transcribed from git HEAD~ of predict.py -------------------
OLD_CRYSTAL_LINES = ["lattice_type = triclinic", "centering = P", "unique_axis = *",
                     "profile_radius = 0.00200 nm^-1",
                     "predict_refine/det_shift x = 0.000 y = 0.000 mm"]

panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
               res=10000.0, cx=-750.0, cy=-750.0, coffset=0.0,
               min_fs=0, max_fs=NPX - 1, min_ss=0, max_ss=NPX - 1)]
CLEN = 0.1
pred = predict_spots(LYSO, panels, CLEN, LAM, dmin=3.0, tol=0.004)[:40]
img = np.zeros((NPX, NPX), np.float32)
yy, xx = np.mgrid[-3:4, -3:4]
g = np.exp(-(xx * xx + yy * yy) / 2.0); g /= g.sum()
for p in pred:
    cf, cs = int(round(p["fs"])), int(round(p["ss"]))
    if 5 <= cs < NPX - 5 and 5 <= cf < NPX - 5:
        img[cs - 3:cs + 4, cf - 3:cf + 4] += 1000.0 * g
I, sig, pk, bg = integrate_spots(img, pred)

BASE = dict(image="run.cxi", event=7, M=LYSO, pred=pred, I=I, sigma=sig, peak=pk, bg=bg)

with tempfile.TemporaryDirectory() as d:
    # ---- (1) BACKWARD COMPATIBILITY: no new keys -> the old constant lines, unchanged ------------
    p_old = os.path.join(d, "old.stream")
    n = write_stream_integrated([dict(BASE)], p_old, clen_m=CLEN)
    old = open(p_old).read()
    check("legacy: indexed count", n == 1, f"n_idx={n}")
    check("legacy: constant lines intact", all(l + "\n" in old for l in OLD_CRYSTAL_LINES))
    check("legacy: clen from kwarg", f"average_camera_length = {CLEN:.6f} m\n" in old)
    check("legacy: no glint/ fields", "glint/" not in old,
          "absent keys must emit nothing at all")

    # ---- (2) NEW FIELDS present and correct ------------------------------------------------------
    rec = dict(BASE, det_shift_mm=(0.1234, -0.5678), clen_m=0.1057, dclen_m=5.7e-4,
               geom_n_solves=3, cell_id=1, lock_generation=2, matched_frac=0.1832,
               low_confidence=True, frame_no=41, lattice_type="tetragonal", centering="P",
               unique_axis="c")
    p_new = os.path.join(d, "new.stream")
    write_stream_integrated([rec], p_new, clen_m=CLEN)
    new = open(p_new).read()
    check("drift: det_shift mm", "predict_refine/det_shift x = 0.123 y = -0.568 mm\n" in new)
    check("drift: per-frame clen overrides", "average_camera_length = 0.105700 m\n" in new,
          "must use the record's clen, not the run-level kwarg")
    check("drift: dclen_m", "glint/dclen_m = 0.00057\n" in new)
    check("prov: cell_id", "glint/cell_id = 1\n" in new)
    check("prov: lock_generation", "glint/lock_generation = 2\n" in new)
    check("prov: matched_frac", "glint/matched_frac = 0.1832\n" in new)
    check("prov: low_confidence as int", "glint/low_confidence = 1\n" in new,
          "bool must not serialise as 'True'")
    check("prov: frame_no", "glint/frame_no = 41\n" in new)
    check("prov: geom_n_solves", "glint/geom_n_solves = 3\n" in new)
    check("symmetry overridable", "lattice_type = tetragonal\ncentering = P\nunique_axis = c\n" in new)

    # a falsey-but-present value must still be emitted (0 is meaningful: primary cell / first lock)
    p_zero = os.path.join(d, "zero.stream")
    write_stream_integrated([dict(BASE, cell_id=0, lock_generation=0, low_confidence=False)],
                            p_zero, clen_m=CLEN)
    z = open(p_zero).read()
    check("prov: zero values still emitted",
          "glint/cell_id = 0\n" in z and "glint/lock_generation = 0\n" in z
          and "glint/low_confidence = 0\n" in z, "0 != absent")

    # ---- (3) INCREMENTAL == BATCH ---------------------------------------------------------------
    recs = [dict(rec, event=i, frame_no=i) for i in range(4)]
    p_batch = os.path.join(d, "b.stream"); p_inc = os.path.join(d, "i.stream")
    write_stream_integrated(recs, p_batch, clen_m=CLEN)
    w = StreamWriter(p_inc, clen_m=CLEN, flush_every=2)
    for r in recs:
        w.write(r)
    ni = w.close()
    check("incremental == batch (bytes)", open(p_batch).read() == open(p_inc).read(),
          "live and offline writers must not diverge")
    check("incremental counts", (w.n_chunks, ni) == (4, 4), f"{w.n_chunks} chunks, {ni} indexed")
    check("close() is idempotent", w.close() == 4)
    # a partial file (never closed) is still a valid, parseable stream
    p_part = os.path.join(d, "p.stream")
    w2 = StreamWriter(p_part, clen_m=CLEN, flush_every=1)
    w2.write(recs[0])
    part = open(p_part).read()
    check("partial file is valid", part.count("----- Begin chunk -----") == 1
          and part.rstrip().endswith("----- End chunk -----"))
    w2.close()

    # ---- (3b) MULTI-PANEL: each row named by ITS OWN panel ---------------------------------------
    # predict_spots tags every reflection with the panel it landed on; a single scalar name would
    # mislabel every reflection not on panel 0, and partialator keys per-panel geometry off that name.
    mp = pred.copy()
    mp["panel"][1::2] = 1                                   # pretend every other spot is on q1
    p_mp = os.path.join(d, "mp.stream")
    write_stream_integrated([dict(BASE, pred=mp)], p_mp, clen_m=CLEN, panel_names=["p0", "q1"])
    mtxt = open(p_mp).read()
    mbody = mtxt.split("Reflections measured after indexing\n")[1].split("End of reflections")[0]
    mrows = [l.split() for l in mbody.split("\n") if l.strip() and not l.lstrip().startswith("h ")]
    got = [r[-1] for r in mrows]
    check("multi-panel: per-row panel names", got == ["p0" if i % 2 == 0 else "q1"
                                                      for i in range(len(mrows))],
          f"{got[:6]}...")
    # without the map, behaviour is unchanged (scalar name everywhere)
    p_sp = os.path.join(d, "sp.stream")
    write_stream_integrated([dict(BASE, pred=mp)], p_sp, clen_m=CLEN, panel_name="zz")
    sbody = open(p_sp).read().split("Reflections measured after indexing\n")[1].split("End of reflections")[0]
    srows = [l.split()[-1] for l in sbody.split("\n") if l.strip() and not l.lstrip().startswith("h ")]
    check("multi-panel: scalar fallback", set(srows) == {"zz"}, f"{set(srows)}")

    # ---- (3c) NUMPY SCALARS: what these values actually are in practice --------------------------
    # np.bool_ is NOT a bool subclass and np.float32 is not a float subclass, so a naive isinstance
    # test emits 'True' / a full binary expansion. `python_float < np.float64` returns np.bool_, which
    # is exactly how the driver's low_confidence flag is produced when the threshold is a numpy float.
    p_np = os.path.join(d, "np.stream")
    write_stream_integrated([dict(BASE, low_confidence=np.bool_(True), matched_frac=np.float32(0.183),
                                  cell_id=np.int64(2), dclen_m=np.float64(5.7e-4),
                                  lock_generation=np.bool_(False))], p_np, clen_m=CLEN)
    nt = open(p_np).read()
    check("numpy: np.bool_ -> 0/1", "glint/low_confidence = 1\n" in nt
          and "glint/lock_generation = 0\n" in nt, "must not emit True/False")
    check("numpy: np.float32 rounded", "glint/matched_frac = 0.183\n" in nt,
          "must not emit 0.18299999833106995")
    check("numpy: np.int64 plain", "glint/cell_id = 2\n" in nt)
    check("numpy: np.float64", "glint/dclen_m = 0.00057\n" in nt)

    # ---- (4) STRUCTURE: parses as a stream --------------------------------------------------------
    nb = new.count("----- Begin chunk -----"); ne = new.count("----- End chunk -----")
    check("chunk balance", nb == ne == 1, f"{nb} begin / {ne} end")
    check("crystal balance", new.count("--- Begin crystal") == new.count("--- End crystal") == 1)
    body = new.split("Reflections measured after indexing\n")[1].split("End of reflections")[0]
    rows = [l for l in body.split("\n") if l.strip() and not l.lstrip().startswith("h ")]
    check("reflection rows", new.count("End of reflections") == 1 and len(rows) == len(pred),
          f"{len(rows)} rows vs {len(pred)} predicted")
    check("reflection row parses", len(rows[0].split()) == 10 and rows[0].split()[-1] == "p0",
          rows[0].strip()[:60])
    check("geometry block in header", "----- Begin geometry file -----" in new)
    # every glint/ line must be a well-formed "key = value" a reader can skip
    gl = [l for l in new.split("\n") if l.startswith("glint/")]
    check("glint/ lines well-formed", len(gl) == 7 and all(" = " in l for l in gl), f"{len(gl)} lines")
    # unindexed frame -> no crystal block, no drift fields
    p_ni = os.path.join(d, "ni.stream")
    k = write_stream_integrated([dict(BASE, M=None, det_shift_mm=(1.0, 2.0))], p_ni, clen_m=CLEN)
    u = open(p_ni).read()
    check("unindexed: no crystal, no drift", k == 0 and "--- Begin crystal" not in u
          and "glint/" not in u and "indexed_by = none" in u)

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
