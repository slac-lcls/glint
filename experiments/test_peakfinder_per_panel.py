"""A multi-panel slab is searched one panel at a time: no background ring, local-max window or
connected component reaches across a panel seam.

A CrystFEL geometry lays its panels out as rectangles of one 2-D data array, and two panels that touch in
the ARRAY are in general not neighbours in the lab. frames_from_cxi (the `glint --images` route) and
StreamDriver (on a multi-panel geometry, as record_stream_replay builds it) used to run ONE v4/pf9 finder
over the whole slab. Its ring, local-max window and labelling then reached into the next panel's rows:
with panel backgrounds of different level the mixed ring inflated sigma and peaks within ~4 px of the
seam were lost, and two spots either side of an unmasked seam merged into one peak between them.
Masking a 1-px panel border does not help, since the ring reaches r=4 px.

The checks:
  1  lute_bridge.slab_rects: two rects for two stacked panels; None (the old single finder) for one
     panel, a 3-D layout, a rect outside the array, overlapping rects.
  2  StreamDriver: one panel keeps a plain PeakFinderV4; two panels with backgrounds 10|100 and rows
     63/64 masked find the peaks 1-3 rows from the seam, equal to one finder per panel run by hand,
     and push() hands every planted peak to the indexer as a q-vector.
  3  spots either side of an UNMASKED seam are two peaks, one per panel.
  4  frames_from_cxi (v4 and pf9) on a two-panel .cxi: every planted peak's q recovered (needs h5py).

Plain script: prints ok/FAIL lines, exits 1 on any failure. numpy + scipy (+ h5py for 4).
"""
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", RuntimeWarning)

import glint.peakfinder_v4 as V4

FAILS = []
KEYS = ("x", "y", "intensity", "snr", "npix")


def check(name, ok, detail=""):
    print(("ok    " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def section(name, fn):
    """Run one block; an exception (e.g. an API an older tree lacks) is one FAIL, not a crash."""
    try:
        fn()
    except Exception as e:                                       # noqa: BLE001 -- report and go on
        check(f"{name}: ran", False, f"{type(e).__name__}: {e}")


def same(a, b):
    """Two peak dicts equal bit for bit."""
    for k in KEYS:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            return False
    return True


def recovered(pk, peaks, tol=1.5):
    x, y = np.asarray(pk["x"]), np.asarray(pk["y"])
    return sum(bool(np.any(np.hypot(x - px, y - py) < tol)) for (py, px) in peaks)


# ---------------------------------------------------------------- the two-panel slab
HP, WP = 64, 128                                               # two panels stacked in ss: a 128x128 slab
SLAB = (2 * HP, WP)
ROWS = (62, 61, 60, 20)                                        # panel-0 peaks, 1..3 rows from its last row


def _slab_frame(seed, bg=(10.0, 100.0), peaks=True):
    r = np.random.default_rng(seed)
    img = np.concatenate([r.poisson(bg[0], (HP, WP)), r.poisson(bg[1], (HP, WP))]).astype(np.float64)
    Y, X = np.mgrid[0:SLAB[0], 0:SLAB[1]]
    pl = [(row, 20 + 30 * i) for i, row in enumerate(ROWS)] if peaks else []
    for (py, px) in pl:
        img += 150.0 * np.exp(-((Y - py) ** 2 + (X - px) ** 2) / 2.0) * (Y < HP)
    return img.astype(np.float32), pl


def _panels():
    """Two panels adjacent in the data array, 1 m apart in the lab (as slab-stacked panels are)."""
    out = []
    for p, (s0, cy) in enumerate(((0, 30.0), (HP, -160.0))):
        out.append(dict(name=f"p{p}", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=1e4,
                        cx=-64.0, cy=cy, coffset=0.0, min_fs=0, max_fs=WP - 1, min_ss=s0, max_ss=s0 + HP - 1))
    return out


def _q_hits(q, pl, panels):
    """How many planted (ss, fs) peaks have a recovered q within the q-shift of a 1.5 px displacement."""
    from glint.lute_bridge import peaks_to_q
    fs = np.array([p[1] for p in pl], float); ss = np.array([p[0] for p in pl], float)
    plq = peaks_to_q(fs, ss, panels, 0.1, 1.3)
    tol = np.linalg.norm(peaks_to_q(fs, ss + 1.5, panels, 0.1, 1.3) - plq, axis=1)
    if len(q) == 0:
        return 0
    return int(sum(np.linalg.norm(q - t, axis=1).min() < e for t, e in zip(plq, tol)))


def _driver(panels, shape, **kw):
    """A CPU StreamDriver in blind mode, built without torch: its lazy `glint.glint_fast` import (the
    blind indexer) is stubbed for the duration, as test_cell_registry.py does. Only the finder is used."""
    import types
    import glint.stream_driver as sd
    stub = types.ModuleType("glint.glint_fast")
    stub.index_blind_nbest = lambda q, k: []
    prev = sys.modules.get("glint.glint_fast")
    sys.modules["glint.glint_fast"] = stub
    try:
        return sd.StreamDriver(None, panels, 0.1, 1.3, shape, dtype=np.float32, use_gpu=False, **kw)
    finally:
        if prev is not None:
            sys.modules["glint.glint_fast"] = prev
        else:
            sys.modules.pop("glint.glint_fast", None)


SEAM_MASK = np.ones(SLAB, bool); SEAM_MASK[HP - 1] = False; SEAM_MASK[HP] = False   # 1-px panel border


def part_1():
    from glint.lute_bridge import slab_rects
    pan = _panels()
    check("1 slab_rects: two stacked panels -> two rects", slab_rects(pan, SLAB) == [(0, HP, 0, WP), (HP, 2 * HP, 0, WP)],
          str(slab_rects(pan, SLAB)))
    check("1 slab_rects: one panel -> None (unchanged single-finder path)", slab_rects(pan[:1], SLAB) is None)
    p3 = [dict(p, dim0=float(i)) for i, p in enumerate(pan)]
    check("1 slab_rects: 3-D layout (integer dimN) -> None", slab_rects(p3, SLAB) is None)
    check("1 slab_rects: a rect outside the array -> None", slab_rects(pan, (HP + 10, WP)) is None)
    pov = [pan[0], dict(pan[1], min_ss=HP - 5)]
    check("1 slab_rects: overlapping rects -> None", slab_rects(pov, SLAB) is None)


def part_2():
    drv1 = _driver(_panels()[:1], (HP, WP))
    check("2 StreamDriver, one panel: the finder is a plain PeakFinderV4 (unchanged)",
          type(drv1.finder) is V4.PeakFinderV4)
    drv = _driver(_panels(), SLAB, mask=SEAM_MASK)
    hits = np.zeros(len(ROWS), int); found = []
    for seed in range(10):
        img, pl = _slab_frame(seed)
        pk = drv.finder.find(img)
        hits += [recovered(pk, [q]) for q in pl]
        found.append((img, pk))
    check(f"2 StreamDriver, two panels, bg 10|100, rows {HP - 1}/{HP} masked: peaks at rows {ROWS} all found (10 frames)",
          bool((hits == 10).all()), f"found per row {hits.tolist()} of 10")
    check("2 StreamDriver finder still reports its settings (r, dt, p)",
          getattr(drv.finder, "r", None) == 4 and hasattr(drv.finder, "dt") and "thr_low" in getattr(drv.finder, "p", {}))

    def by_hand():
        exact = all(same(pk, V4.merge_panel_peaks([V4.PeakFinderV4(SEAM_MASK[:HP]).find(img[:HP]),
                                                   V4.PeakFinderV4(SEAM_MASK[HP:]).find(img[HP:])],
                                                  [(0, HP, 0, WP), (HP, 2 * HP, 0, WP)])) for img, pk in found)
        check("2 StreamDriver finder == one PeakFinderV4 per panel, run by hand", exact)
    section("2 by hand", by_hand)
    # through push(): the warm-up ingest hands the peaks' q to the blind indexer (stubbed here)
    got = []
    drv2 = _driver(_panels(), SLAB, mask=SEAM_MASK, min_peaks=1)
    drv2._ingest_blind_q = lambda qq, n_peaks=None: got.append(qq)
    img, pl = _slab_frame(3)
    drv2.push(img)
    q = got[-1] if got and got[-1] is not None else np.zeros((0, 3))
    n = _q_hits(q, pl, _panels())
    check("2 StreamDriver.push: every planted peak reaches the indexer as a q-vector", n == len(pl), f"{n}/{len(pl)}")


def part_3():
    """Spots either side of an UNMASKED seam are two peaks, one per panel -- not one merged centroid."""
    drv = _driver(_panels(), SLAB)
    r = np.random.default_rng(5)
    img = r.poisson(20.0, SLAB).astype(np.float64)
    Y, X = np.mgrid[0:SLAB[0], 0:SLAB[1]]
    img += 200.0 * np.exp(-((Y - 63) ** 2 + (X - 60) ** 2) / 2.0) * (Y < HP)
    img += 200.0 * np.exp(-((Y - 64) ** 2 + (X - 61) ** 2) / 2.0) * (Y >= HP)
    pk = drv.finder.find(img.astype(np.float32))
    y, x = np.asarray(pk["y"]), np.asarray(pk["x"])
    near = (np.abs(y - 63.5) < 4) & (np.abs(x - 60.5) < 4)
    side = sorted(int(v >= HP) for v in y[near])
    check("3 unmasked seam: one peak on each panel, none straddling it", side == [0, 1],
          f"{int(near.sum())} peak(s) at y={np.round(y[near], 2).tolist()}")


def part_4():
    try:
        import h5py
    except ImportError:
        print("SKIP  F4: no h5py (CI installs it) -- frames_from_cxi not exercised")
        return
    from glint.lute_bridge import frames_from_cxi, parse_geom
    tmp = tempfile.mkdtemp(prefix="glint_pfmask_")
    try:
        geom = os.path.join(tmp, "slab2.geom")
        g = ("clen = 0.1\nres = 10000\ndata = /entry_1/data_1/data\nmask = /entry_1/data_1/mask\n"
             "mask_good = 0x0\nmask_bad = 0x1\n")
        for p in _panels():
            n = p["name"]
            g += (f"{n}/min_fs = {p['min_fs']}\n{n}/max_fs = {p['max_fs']}\n{n}/min_ss = {p['min_ss']}\n"
                  f"{n}/max_ss = {p['max_ss']}\n{n}/fs = +1.0x +0.0y\n{n}/ss = +0.0x +1.0y\n"
                  f"{n}/corner_x = {p['cx']}\n{n}/corner_y = {p['cy']}\n")
        open(geom, "w").write(g)
        frames, pls = zip(*[_slab_frame(100 + s) for s in range(4)])
        stack = np.stack(frames)
        cxi = os.path.join(tmp, "slab2.cxi")
        with h5py.File(cxi, "w") as h:
            h["/entry_1/data_1/data"] = stack
            mk = np.zeros(SLAB, np.uint16); mk[~SEAM_MASK] = 1
            h["/entry_1/data_1/mask"] = mk
        panels, _ = parse_geom(geom)
        for pf in ("v4", "pf9"):
            fr, _ = frames_from_cxi(cxi, geom, wavelength_A=1.3, min_peaks=1, peakfinder=pf)
            n = tot = 0
            for q, pl in zip(fr, pls):
                n += _q_hits(q, pl, panels); tot += len(pl)
            check(f"4 frames_from_cxi(peakfinder={pf!r}), two-panel slab: every planted peak's q recovered",
                  n == tot, f"{n}/{tot}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


for nm, fn in (("1", part_1), ("2", part_2), ("3", part_3), ("4", part_4)):
    section(nm, fn)

print()
if FAILS:
    print(f"{len(FAILS)} check(s) FAILED")
    sys.exit(1)
print("ALL PASS")
