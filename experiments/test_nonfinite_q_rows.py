"""A NaN/inf or zero-length q row must cost the frame that ROW, never the whole frame.

WHY THIS EXISTS. Two review findings, one mechanism:
  * s4-05: a peak exactly on the beam centre maps to q = [0, 0, 0] (the geometry returns that exactly when
    r_x = r_y = 0). The blind weight 1/|q| is then inf, every M3 start goes NaN, and index_blind_fast returns
    None / index_blind_nbest [] for the frame. Every front door filtered only NON-FINITE rows, so the row got in.
  * s7-05: the --images route (lute_bridge.frames_from_cxi) kept the NaN row lute_bridge.peaks_to_q returns for
    a peak on no .geom panel. hybrid_index then got nothing from either search and the frame was dropped, with
    nothing on stderr. Every other front door dropped just that peak.
The fix is one rule (glint.geom.q_rows_ok: |q|^2 finite and |q| > Q_FLOOR) applied where q enters an
indexer -- the blind and known-cell engines, CPU and torch, per-frame and batched; hybrid_index, dense_index,
index_known_fast and gate_results; the CLI loader; frames_from_cxi; geom.peaks_to_q; StreamDriver's six front
doors -- plus a zero weight below Q_FLOOR in the blind weight itself, for callers that build Q by hand.

WHAT IT CHECKS.
  numpy (always runs):
    1. the rule: which rows q_rows_ok keeps, and clean_q hands a clean frame back as the SAME object, which is
       what keeps every clean frame bit-identical through every caller; clean_frames (the batch engines' one
       check per batch) returns exactly what clean_q returns frame by frame, whatever the frames' dtype and
       memory layout, rows within a few ulps of Q_FLOOR^2 included;
    2. the --peaks route (geom.peaks_to_q) drops a beam-centre peak as it drops an off-panel one;
    3. the --images route (frames_from_cxi, peakfinder='stored'; needs h5py, skips without it): no NaN or zero
       row reaches the caller, min_peaks counts usable rows, a clean frame is unchanged, the drop is reported
       on stderr; and glint_cli._load_frames on that file;
    4. StreamDriver push_q / push_peaks / warmup_batch_q (blind mode, the blind indexer injected through its
       seam): the indexer never sees a bad row, and min_peaks counts usable rows;
    5. hybrid_index with glint_fast / replica_gpu stubbed: the indexers never see a bad row, every frame is
       registered, and gate_results judges the rows that were indexed.
  torch (skips without torch or below pyproject's 1.12 floor):
    6. the weight: 1/|q| exactly above Q_FLOOR, 0 at q = 0, and an M3 ascent over a frame with a q = 0 row has
       no NaN start;
    7. index_blind_fast, index_blind_nbest and glint_index.index_blind on simulated frames with a zero, NaN or
       inf row return EXACTLY what they return on the clean frame (and the clean frame does index);
    8. index_known_gpu_cell, index_known_gpu_cell_batch and hybrid_index(Mc_known=...) likewise, and a batch
       frame whose every row is bad is a miss (None), not a NaN or a starting-candidate matrix;
    9. glint_cli._load_frames --qframes drops the rows before --min-peaks counts them.

  PYTHONPATH=. python experiments/test_nonfinite_q_rows.py      # exit 0 = all pass
"""
import argparse
import contextlib
import importlib
import io
import os
import sys
import tempfile
import types

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np                                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

FAILS = []
NAN, INF = np.nan, np.inf


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


def guarded(title, fn):
    """Run one part; an exception inside it is a FAIL of that part, not a crash of the file."""
    print(f"\n{title}")
    try:
        fn()
    except Exception as exc:                                               # noqa: BLE001
        check(f"{title}: ran to the end", False, f"{type(exc).__name__}: {exc}")


def bad_rows(q):
    """How many rows of q an indexer must not see: |q|^2 not finite in float64 (a NaN or inf component, or a
    finite row too large to square), or |q| <= 1e-6 1/A. glint.geom.q_rows_ok's rule, written out on its own."""
    q = np.asarray(q, float).reshape(-1, 3)
    with np.errstate(over="ignore"):
        s = (q * q).sum(1)
    return int((~(np.isfinite(s) & (s > 1e-6 * 1e-6))).sum())


def frame_on(M, rng, n=40):
    """q exactly on lattice M (q @ M == hkl), no hkl = 000."""
    hkl = rng.integers(-5, 6, size=(4 * n, 3)).astype(float)
    hkl = hkl[np.abs(hkl).sum(1) > 0][:n]
    return hkl @ np.linalg.inv(M)


# One 200 x 200 panel, 100 um pixels, beam through the middle of pixel (100, 100): a peak at fs = ss = 100.0
# lands at r_x = r_y = 0, i.e. q = [0, 0, 0] exactly. fs = -0.4 floors to column -1, on no panel.
GEOM_TEXT = """\
photon_energy = 9500
clen = 0.1
res = 10000
coffset = 0.0
p0/min_fs = 0
p0/max_fs = 199
p0/min_ss = 0
p0/max_ss = 199
p0/corner_x = -100.0
p0/corner_y = -100.0
p0/fs = +1.0x
p0/ss = +1.0y
"""
CENTRE = (100.0, 100.0)
OFF_PANEL = (-0.4, 50.0)


# ------------------------------------------------------------------------------------------------ numpy
def part_rule():
    from glint.geom import Q_FLOOR, clean_q, q_rows_ok
    q = np.array([[0.05, -0.02, 0.01], [0, 0, 0], [NAN, 0.1, 0.2], [INF, 0, 0], [-INF, 0, 0], [0, NAN, 0],
                  [1e-7, 0, 0], [0, 0, 3e-6], [0.3, 0.2, -0.1]])
    want = [True, False, False, False, False, False, False, True, True]
    check("q_rows_ok keeps finite rows with |q| > Q_FLOOR; rejects NaN, +-inf, 0 and sub-floor rows",
          q_rows_ok(q).tolist() == want, q_rows_ok(q).tolist())
    check("Q_FLOOR is far below a peak one pixel from the beam (~5.8e-4 1/A)", 0 < Q_FLOOR <= 1e-5, Q_FLOOR)
    check("clean_q drops exactly the rejected rows and keeps their order",
          np.array_equal(clean_q(q), q[np.array(want)]), clean_q(q))
    good = np.random.default_rng(0).normal(size=(30, 3)) * 0.1
    check("clean_q hands a clean frame back as the SAME object (clean frames stay bit-identical)",
          clean_q(good) is good)
    lst = good.tolist()
    check("...a clean list too, unconverted", clean_q(lst) is lst)
    f32 = good.astype(np.float32)
    got = clean_q(np.vstack([f32, np.zeros((1, 3), np.float32)]))
    check("...and a filtered frame keeps its dtype", got.dtype == np.float32 and len(got) == 30, (got.dtype, len(got)))
    check("an empty frame passes through", clean_q(np.empty((0, 3))).shape == (0, 3))
    check("q_rows_ok rejects a finite row too large to square (|q|^2 overflows)",
          q_rows_ok(np.array([[1e200, 0, 0], [0.1, 0.2, 0.3]])).tolist() == [False, True])
    qq = np.vstack([q, [[1e200, 0, 0]]])
    check("this file's bad_rows counts exactly the rows q_rows_ok rejects",
          bad_rows(qq) == int((~q_rows_ok(qq)).sum()) == 7, (bad_rows(qq), int((~q_rows_ok(qq)).sum())))

    # Rows within a few ulps of Q_FLOOR^2. einsum adds the three squares in another order for a Fortran-order
    # array, so these must get one verdict whatever the frame's layout, alone or stacked in a batch. The first
    # is the review's row: |q|^2 = 1.0000000000000002e-12 from C-order rows, 1e-12 from Fortran-order (arm64).
    rng = np.random.default_rng(1)
    u = rng.normal(size=(256, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    edge = np.vstack([[4.947971892216682e-07, 7.143888878451804e-07, 4.947971892216679e-07],
                      u * (Q_FLOOR * (1 + rng.integers(-1, 2, (256, 1)) * 2.0 ** -52))])
    near = u * (Q_FLOOR * (1 + 2.0 ** -52))            # |q|^2 about two ulps above Q_FLOOR^2: most rows pass
    ok_c, ok_f = q_rows_ok(edge), q_rows_ok(np.asfortranarray(edge))
    check("q_rows_ok gives a row within a few ulps of Q_FLOOR^2 the same verdict in C- and Fortran-order frames",
          np.array_equal(ok_c, ok_f) and 0 < ok_c.sum() < len(edge), (int((ok_c != ok_f).sum()), int(ok_c.sum())))

    from glint.geom import clean_frames

    def same(a, b):        # what clean_q returned, frame by frame: the same object, or an equal new array
        return a is b or (not isinstance(b, list) and isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                          and a.dtype == b.dtype and np.array_equal(a, b))

    def agree(fs):         # clean_frames(fs) vs [clean_q(f) for f in fs]; an exception counts by its type
        try:
            want = [clean_q(f) for f in fs]
        except Exception as exc:                                           # noqa: BLE001
            want = type(exc)
        try:
            got = clean_frames(iter(fs))
        except Exception as exc:                                           # noqa: BLE001
            got = type(exc)
        if not (isinstance(want, list) and isinstance(got, list)):
            return want is got, want
        return len(got) == len(want) and all(same(g, w) for g, w in zip(got, want)), want

    fr = [good, lst, f32[:5], np.empty((0, 3))]
    out = clean_frames(fr)
    check("clean_frames hands a clean batch back as the SAME objects, frame for frame",
          len(out) == 4 and all(o is f for o, f in zip(out, fr)))
    c = np.vstack([good, edge[:1]])
    f = np.asfortranarray(c)
    res = [agree(b)[0] for b in ([good, f], [f, good], [c, f], [f], [good.astype(object), f32], [c.astype(object)])]
    check("clean_frames == clean_q with the review's row in a Fortran-order frame next to C-order ones, and on "
          "object-dtype frames of floats", all(res), res)
    bad = [np.zeros(3), [NAN, 0, 0], [0, INF, 0], [0, 0, -INF], [1e-7, 0, 0], [1e200, 0, 0]]
    odd = [None, [], np.empty((0, 3)), np.zeros((0,)), [[0.1, 0.2, 0.3]] * 7, np.ones((4, 2)) * 0.1]
    n_dirty = n_bad = n_raised = n_fort = n_obj = 0
    for t in range(400):
        fs = []
        for _ in range(int(rng.integers(1, 9))):
            f = rng.normal(size=(int(rng.integers(0, 40)), 3)) * 0.1
            if rng.random() < 0.15 and len(f):
                f[int(rng.integers(len(f)))] = bad[int(rng.integers(len(bad)))]
            if rng.random() < 0.3 and len(f):
                f[rng.integers(len(f), size=5)] = near[rng.integers(len(near), size=5)]
            with np.errstate(over="ignore"):
                f = f.astype(np.float32) if rng.random() < 0.2 else f
            r = rng.random()
            f = f.tolist() if r < 0.1 else np.asfortranarray(f) if r < 0.35 else f.astype(object) if r < 0.45 else f
            n_fort += isinstance(f, np.ndarray) and f.ndim == 2 and len(f) > 1 and not f.flags.c_contiguous
            n_obj += isinstance(f, np.ndarray) and f.dtype == object
            fs.append(f)
        if rng.random() < 0.1:
            fs.insert(int(rng.integers(len(fs) + 1)), odd[int(rng.integers(len(odd)))])
        ok, want_t = agree(fs)
        n_bad += not ok
        n_raised += not isinstance(want_t, list)
        n_dirty += isinstance(want_t, list) and any(w is not f for w, f in zip(want_t, fs))
    check("clean_frames == clean_q frame by frame, and neither raises (400 random batches: bad rows, rows within "
          "a few ulps of Q_FLOOR^2, float32, C- and Fortran-order, object dtype, lists, empty / 1-D / 2-column / None "
          "frames, an iterator)", n_bad == n_raised == 0 and n_dirty > 20 and n_fort > 100 and n_obj > 50,
          dict(bad=n_bad, raised=n_raised, dirty=n_dirty, fortran=n_fort, object=n_obj))


def part_peaks_route():
    from glint.geom import parse_geom, peaks_to_q
    with tempfile.TemporaryDirectory() as d:
        gp = os.path.join(d, "one.geom")
        open(gp, "w").write(GEOM_TEXT)
        geom = parse_geom(gp)
    pk = np.array([[150.0, 37.0], CENTRE, OFF_PANEL, [20.0, 180.0]])
    q = peaks_to_q(pk, geom)
    check("geom.peaks_to_q (--peaks route): beam-centre and off-panel peaks dropped, 2 rows left",
          len(q) == 2 and bad_rows(q) == 0, q)


def _write_cxi(path, frames_xy):
    import h5py
    npk = max(len(x) for x in frames_xy)
    X = np.zeros((len(frames_xy), npk), np.float32); Y = np.zeros_like(X)
    I = np.zeros_like(X); N = np.zeros(len(frames_xy), np.int32)
    for i, xy in enumerate(frames_xy):
        xy = np.asarray(xy, float)
        X[i, :len(xy)], Y[i, :len(xy)] = xy[:, 0], xy[:, 1]
        I[i, :len(xy)] = 500.0 + np.arange(len(xy)); N[i] = len(xy)
    with h5py.File(path, "w") as f:
        g = f.create_group("entry_1/result_1")
        g.create_dataset("peakXPosRaw", data=X); g.create_dataset("peakYPosRaw", data=Y)
        g.create_dataset("nPeaks", data=N); g.create_dataset("peakTotalIntensity", data=I)


def part_images_route():
    try:
        import h5py                                                        # noqa: F401
    except ImportError:
        print("  SKIP  no h5py: the stored-peak .cxi cannot be written here (the CPU CI job installs it)")
        return
    from glint.lute_bridge import frames_from_cxi, parse_geom as lb_parse_geom, peaks_to_q as lb_peaks_to_q
    rng = np.random.default_rng(7)
    on = lambda n: np.c_[rng.uniform(5, 195, n), rng.uniform(5, 195, n)]   # noqa: E731
    f0 = np.vstack([on(8), [OFF_PANEL], [CENTRE]])        # 8 usable + 1 off-panel + 1 beam-centre
    f1 = np.vstack([on(5), [OFF_PANEL]])                  # 6 stored, 5 usable: below min_peaks=6
    f2 = on(7)                                            # clean control
    with tempfile.TemporaryDirectory() as d:
        cxi, gp = os.path.join(d, "stack.cxi"), os.path.join(d, "one.geom")
        open(gp, "w").write(GEOM_TEXT)
        _write_cxi(cxi, [f0, f1, f2])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            frames, images = frames_from_cxi(cxi, gp, peakfinder="stored", min_peaks=6)
        check("frames_from_cxi: no NaN or zero row reaches the caller", sum(bad_rows(q) for q in frames) == 0,
              [bad_rows(q) for q in frames])
        check("frames_from_cxi: the off-panel and beam-centre peaks cost frame 0 two rows, not the frame",
              len(frames[0]) == 8, len(frames[0]))
        check("frames_from_cxi: min_peaks counts USABLE rows (6 stored, 5 usable -> empty frame)",
              len(frames[1]) == 0, len(frames[1]))
        panels, _ = lb_parse_geom(gp)
        ref = lb_peaks_to_q(f2[:, 0].astype(np.float32).astype(float), f2[:, 1].astype(np.float32).astype(float),
                            panels, 0.1, 12398.419843320026 / 9500.0)
        check("frames_from_cxi: a clean frame is exactly peaks_to_q of its peaks", np.array_equal(frames[2], ref))
        check("frames_from_cxi: the drop is reported on stderr (3 peaks in 2 frames)",
              "dropped 3 peak(s) in 2 frame(s)" in err.getvalue(), err.getvalue().strip() or "(nothing)")
        from glint.glint_cli import _load_frames
        a = argparse.Namespace(qframes=None, images=cxi, geom=gp, wavelength=None, N=0, min_peaks=6, data_path=None,
                               peakfinder="stored", top_peaks=0, ring_focus=False, cell=None, ring_qlow=0.15)
        with contextlib.redirect_stderr(io.StringIO()):
            fr, im = _load_frames(a)
        check("glint_cli._load_frames --images: frames 0 and 2 kept, no bad row, frame 1 under --min-peaks",
              [m["event"] for m in im] == [0, 2] and [len(q) for q in fr] == [8, 7]
              and sum(bad_rows(q) for q in fr) == 0, ([m["event"] for m in im], [len(q) for q in fr]))


@contextlib.contextmanager
def _stub_modules(**mods):
    """Put stub modules in sys.modules for the duration; leave the interpreter as it was found."""
    saved = {k: sys.modules.get(k) for k in mods}
    for k, m in mods.items():
        sys.modules[k] = m
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def part_stream_driver():
    from glint.stream_driver import StreamDriver
    n = 64
    panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=1e4,
                   cx=-(n / 2.0 - 0.5), cy=-(n / 2.0 - 0.5), coffset=0.0,
                   min_fs=0, max_fs=n - 1, min_ss=0, max_ss=n - 1)]
    stub = types.ModuleType("glint.glint_fast")
    stub.index_blind_nbest = lambda q, k: []
    seen = []

    def blind(q, k):                                       # records what the blind indexer is handed
        seen.append(np.array(q, float))
        return []

    def driver():
        with _stub_modules(**{"glint.glint_fast": stub}):
            d = StreamDriver(None, panels, 0.1, 1.3, (n, n), dtype=np.uint16, B=8, dmin=3.0, use_gpu=False,
                             min_peaks=6)
        d._blind_index = blind
        return d

    rng = np.random.default_rng(3)
    good = rng.normal(size=(10, 3)) * 0.05
    d = driver()
    d.push_q(np.vstack([[0.0, 0.0, 0.0], good, [NAN, 0.1, 0.1]]))
    check("StreamDriver.push_q: the blind indexer gets the 10 usable rows only",
          len(seen) == 1 and len(seen[0]) == 10 and bad_rows(seen[0]) == 0, [len(s) for s in seen])
    seen.clear()
    d.push_q(np.vstack([good[:5], [[0.0, 0.0, 0.0]]]))
    check("StreamDriver.push_q: min_peaks counts usable rows (5 usable + q = 0 is not a voting frame)",
          len(seen) == 0, [len(s) for s in seen])
    seen.clear()
    centre = n / 2.0 - 0.5                                 # lab x = y = 0: q = [0, 0, 0] exactly
    fs = np.array([centre, 3.0, 9.5, 20.0, 41.0, 50.5, 60.0])
    ss = np.array([centre, 7.0, 55.0, 12.0, 33.0, 2.5, 48.0])
    d.push_peaks(fs, ss)
    check("StreamDriver.push_peaks: a beam-centre peak is dropped like an off-panel one (6 rows reach the indexer)",
          len(seen) == 1 and len(seen[0]) == 6 and bad_rows(seen[0]) == 0, [len(s) for s in seen])
    seen.clear()
    d = driver()
    d.warmup_batch_q([np.vstack([good, [[0.0, 0.0, 0.0]]]), good[:8]])
    check("StreamDriver.warmup_batch_q: no zero row reaches the blind indexer",
          len(seen) == 2 and all(bad_rows(s) == 0 for s in seen) and sorted(len(s) for s in seen) == [8, 10],
          [len(s) for s in seen])


def part_hybrid_stubbed():
    from glint.lattice import cell_to_Ar
    cell = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
    rng = np.random.default_rng(11)
    clean = [frame_on(cell, rng) for _ in range(3)]
    dirty = [clean[0], np.vstack([clean[1], [[NAN, 0.0, 0.1]]]), np.vstack([[[0.0, 0.0, 0.0]], clean[2]])]
    seen = []

    def nbest(q, k):
        seen.append(("nbest", bad_rows(q)))
        return []                                                          # force the known-cell rescue

    def known(q, Mc, topa=8, nc=None):
        seen.append(("known", bad_rows(q)))
        return np.asarray(Mc, float).copy()

    def matched(M, q):
        r = np.asarray(q, float) @ M
        return int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())

    gf = types.ModuleType("glint.glint_fast")
    gf.GATE_FRAC, gf.GATE_MIN = 0.25, 10
    gf.index_blind_fast = lambda q: None
    gf.index_blind_nbest = nbest
    gf.load = lambda p: []
    gf.matched_strict = matched
    rg = types.ModuleType("glint.replica_gpu")
    rg.index_known_gpu_cell = known
    with _stub_modules(**{"glint.glint_fast": gf, "glint.replica_gpu": rg, "glint.hybrid_stream": None}):
        sys.modules.pop("glint.hybrid_stream", None)
        hs = importlib.import_module("glint.hybrid_stream")
        res, st = hs.hybrid_index(dirty, Mc_known=cell, warmup=False)
        check("hybrid_index: neither indexer is ever handed a bad row", all(b == 0 for _, b in seen), seen)
        check("hybrid_index: all 3 frames registered (a NaN row and a q = 0 row cost only that row)",
              st["n_idx"] == 3 and all(r["M"] is not None for r in res), st["n_idx"])
        check("hybrid_index: no bad row in what is written", all(bad_rows(r["q"]) == 0 for r in res))
        sparse = np.vstack([frame_on(cell, rng, n=12), np.full((45, 3), NAN)])     # 12 on lattice + 45 NaN rows
        r = [{"M": cell.copy(), "hkl": None, "q": sparse}]
        n_w = hs.gate_results(r, [sparse], "strict")
        check("gate_results judges the rows that were indexed (12/12 kept, not 12/57 < 25% withdrawn)",
              n_w == 0 and r[0]["M"] is not None, n_w)


# ------------------------------------------------------------------------------------------------ torch
def torch_parts():
    try:
        import torch
    except ImportError:
        print("\nSKIP torch half -- no torch: glint.glint_fast cannot import here")
        return
    ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if ver < (1, 12):
        print(f"\nSKIP torch half -- torch {torch.__version__} is below pyproject's floor (>= 1.12)")
        return
    from glint import glint_fast as gf, glint_index as gi
    from glint.simulate import simulate_shot
    rng = np.random.default_rng(3)
    shots = [simulate_shot(cell=(79.0, 79.0, 38.0, 90, 90, 90), n_target=60, dmin=2.0, pos_sigma=3e-4, rng=rng)
             for _ in range(2)]
    q0, q1 = shots[0].g, shots[1].g
    zero_end = np.vstack([q0, [[0.0, 0.0, 0.0]]])
    nan_front = np.vstack([[[NAN, 0.02, 0.03]], q1])
    inf_mid = np.vstack([q1[:20], [[0.01, INF, 0.0]], q1[20:]])
    print(f"\n(device {gf.DEV}, torch {torch.__version__}, QPOW {gf.QPOW}, M3_FUSED {gf.M3_FUSED})")

    def same(a, b):
        if a is None or b is None:
            return a is None and b is None
        return np.array_equal(np.asarray(a, float), np.asarray(b, float))

    def same_nb(a, b):
        return len(a) == len(b) and all(same(ca, cb) and sa == sb for (ca, sa), (cb, sb) in zip(a, b))

    def weight():
        Q = torch.as_tensor(zero_end, dtype=torch.float32, device=gf.DEV)
        w = gi.invq_weight(Q)
        check("invq_weight: 0 (not inf) for the q = 0 row", float(w[-1]) == 0.0, float(w[-1]))
        check("invq_weight: exactly 1/|q| on every other row", torch.equal(w[:-1], (1.0 / Q.norm(dim=1))[:-1]))
        st = gi.sample(n_dir=64).to(gf.DEV)
        T = gf._refine(st.clone(), Q, w, float(Q.norm(dim=1).max()))
        check("M3 ascent on a frame with a q = 0 row: no NaN start", int(torch.isnan(T).any(1).sum()) == 0,
              int(torch.isnan(T).any(1).sum()))

    def blind():
        a, b = gf.index_blind_fast(q0), gf.index_blind_fast(zero_end)
        check("index_blind_fast: clean frame indexes (else the next check is vacuous)", a is not None)
        check("index_blind_fast: + one q = 0 row -> EXACTLY the clean answer", same(a, b), b)
        a, b = gf.index_blind_fast(q1), gf.index_blind_fast(nan_front)
        check("index_blind_fast: + one NaN row -> EXACTLY the clean answer", a is not None and same(a, b), b)
        a, b = gf.index_blind_nbest(q0, 3), gf.index_blind_nbest(zero_end, 3)
        check("index_blind_nbest: clean frame has hypotheses", len(a) >= 1, len(a))
        check("index_blind_nbest: + one q = 0 row -> EXACTLY the clean hypotheses and scores", same_nb(a, b), len(b))
        a, b = gf.index_blind_nbest(q1, 3), gf.index_blind_nbest(inf_mid, 3)
        check("index_blind_nbest: + one inf row -> EXACTLY the clean hypotheses and scores",
              len(a) >= 1 and same_nb(a, b), len(b))
        st = gi.sample(n_dir=200).to(gf.DEV)                    # a smaller start grid: the scalar path is slow
        a, b = gi.index_blind(q0, starts=st), gi.index_blind(zero_end, starts=st)
        check("glint_index.index_blind (scalar): + one q = 0 row -> EXACTLY the clean answer",
              a is not None and same(a, b), b)

    def known():
        from glint.hybrid_stream import hybrid_index
        import glint.hybrid_stream as hs
        from glint.lattice import cell_to_Ar
        from glint.replica_gpu import index_known_gpu_cell
        from glint.replica_gpu_batch import index_known_gpu_cell_batch
        Mc = cell_to_Ar(79.0, 79.0, 38.0, 90, 90, 90)
        a, b = index_known_gpu_cell(q1, Mc), index_known_gpu_cell(nan_front, Mc)
        check("index_known_gpu_cell: + one NaN row -> EXACTLY the clean answer", a is not None and same(a, b), b)
        ref = index_known_gpu_cell_batch([q0, q1], Mc)
        got = index_known_gpu_cell_batch([zero_end, nan_front, np.full((7, 3), NAN)], Mc)
        check("index_known_gpu_cell_batch: frames with a q = 0 / NaN row -> EXACTLY the clean batch's answers",
              ref[0] is not None and same(ref[0], got[0]) and same(ref[1], got[1]), got[:2])
        check("index_known_gpu_cell_batch: a frame whose every row is NaN is a miss (None)", got[2] is None, got[2])
        real_nb = hs.index_blind_nbest
        hs.index_blind_nbest = lambda q, k: []      # the blind engine is covered above; this is the rescue path
        try:
            rc, sc = hybrid_index([q0, q1], Mc_known=Mc, warmup=False)
            rd, sd_ = hybrid_index([zero_end, nan_front], Mc_known=Mc, warmup=False)
        finally:
            hs.index_blind_nbest = real_nb
        check("hybrid_index(Mc_known): frames with a q = 0 / NaN row register EXACTLY as the clean ones",
              sc["n_idx"] == sd_["n_idx"] == 2 and all(same(x["M"], y["M"]) for x, y in zip(rc, rd)),
              (sc["n_idx"], sd_["n_idx"]))

    def qframes():
        from glint.glint_cli import _load_frames
        rows = [np.vstack([q0, [[0.0, 0.0, 0.0]], [[NAN, NAN, NAN]]]), np.vstack([q1[:5], [[NAN, 0.0, 0.0]]])]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "frames.txt")
            with open(p, "w") as fh:
                for i, q in enumerate(rows):
                    fh.write(f"FRAME {i} {len(q)}\n" + "".join(f"{x!r} {y!r} {z!r}\n" for x, y, z in q))
            a = argparse.Namespace(qframes=p, images=None, geom=None, wavelength=None, N=0, min_peaks=6)
            fr, im = _load_frames(a)
        check("glint_cli._load_frames --qframes: bad rows dropped, then --min-peaks (frame 1: 5 usable) applied",
              [m["event"] for m in im] == [0] and len(fr[0]) == len(q0) and bad_rows(fr[0]) == 0,
              ([m["event"] for m in im], [len(q) for q in fr]))

    guarded("6. the blind weight", weight)
    guarded("7. blind engines (simulated frames, CPU torch)", blind)
    guarded("8. known-cell engines and hybrid_index", known)
    guarded("9. CLI --qframes loader", qframes)


def main():
    guarded("1. the row rule (glint.geom)", part_rule)
    guarded("2. --peaks route: geom.peaks_to_q", part_peaks_route)
    guarded("3. --images route: lute_bridge.frames_from_cxi + glint_cli._load_frames", part_images_route)
    guarded("4. StreamDriver front doors (blind indexer injected)", part_stream_driver)
    guarded("5. hybrid_index + gate_results (indexers stubbed)", part_hybrid_stubbed)
    torch_parts()
    print()
    print("ALL PASS" if not FAILS else f"FAIL: {len(FAILS)} check(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
