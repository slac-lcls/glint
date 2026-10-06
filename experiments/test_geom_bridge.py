"""Self-contained correctness test for the CrystFEL .geom -> q bridge (no real data / no GPU):
lysozyme cell + random orientation -> Ewald spots q -> forward-project to detector pixels via a
synthetic single-panel geom -> write a real .geom -> parse_geom + peaks_to_q -> q' -> index.
Passes if (a) q' reproduces q and (b) the bridged peaks index back to lysozyme.

It also pins the .geom grammar the bridge has to read the way CrystFEL does (ok/FAIL lines):
  * bad regions (glint review s3-04). geometry(5) bad-region blocks (`badregionA/min_fs ...`,
    `bad_beamstop/min_x ...`) are not panels. parse_geom used to register them as panels, and the
    --peaks route then died with a bare KeyError 'corner_x'. They must parse to the same panels and
    the same q as the file without them, lute_bridge and the stream writer must agree, and the
    fact that GLINT does not apply them must be said, not silent.
  * fs/ss directions (glint review s3-05). A term may omit its coefficient (`fs = x`, `ss = -y`,
    which GLINT's own stream header writes). Both parsers dropped such terms and returned a zero
    basis, collapsing every peak on the panel onto its corner with no warning. geom and lute_bridge
    now share one parser, which reads those forms and raises on anything it cannot read.

  python test_geom_bridge.py
"""
import argparse, os, sys, tempfile, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.spatial.transform import Rotation
from glint import index_shot
from glint.lattice import cell_to_Ar
from glint.multishot import same_lattice
from glint.geom import parse_geom, peaks_to_q

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
LAM = 1.322                                              # A (cxidb)
RES, CLEN = 10000.0, 0.10                                # px/m, m
CX, CY, NPX = -750.0, -750.0, 1500                       # beam-centred panel corner (px), size

GEOM = f"""photon_energy = {12398.419843320026 / LAM:.4f}
clen = {CLEN}
res = {RES}
coffset = 0.0
adu_per_eV = 0.001
p0/min_fs = 0
p0/max_fs = {NPX - 1}
p0/min_ss = 0
p0/max_ss = {NPX - 1}
p0/corner_x = {CX}
p0/corner_y = {CY}
p0/fs = +1.0x +0.0y
p0/ss = +0.0x +1.0y
"""


def ewald_spots(seed):
    R = Rotation.random(random_state=seed).as_matrix()
    B = np.linalg.inv(LYSO).T
    hk = np.array([[h, k, l] for h in range(-7, 8) for k in range(-7, 8) for l in range(-13, 14)])
    g = (R @ B @ hk.T).T
    s0 = np.array([0, 0, 1 / LAM])
    res = np.abs(np.linalg.norm(s0 + g, axis=1) - 1 / LAM)
    return g[(res < 0.0012) & (np.linalg.norm(g, axis=1) > 0.02)]


def project(q):
    """q (near Ewald) -> (detector pixels (fs,ss), exact-Ewald q_ref). Snaps the scattered ray to
    the sphere (shat normalized), so q_ref is what real detector pixels encode and the bridge must
    reproduce exactly; the input q's small off-Ewald slack is the only difference."""
    shat = np.array([0, 0, 1.0]) + LAM * q
    shat = shat / np.linalg.norm(shat, axis=1, keepdims=True)   # exact unit (on sphere)
    t = CLEN / shat[:, 2]
    px, py = t * shat[:, 0], t * shat[:, 1]                 # metres
    fs = px * RES - CX; ss = py * RES - CY                  # pixels
    q_ref = (shat - np.array([0, 0, 1.0])) / LAM
    return np.stack([fs, ss], 1), q_ref


# ---------------------------------------------------------------------------------------------
# .geom grammar checks. Each check catches its own exception, so a pre-fix crash prints a FAIL
# line instead of a traceback that hides the checks after it.
FAILS = []


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as exc:                       # noqa: BLE001 - the exception IS the detail
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)[:200]}")
    if not ok:
        FAILS.append(name)
    return ok


def _write(d, name, text):
    path = os.path.join(d, name)
    with open(path, "w") as f:
        f.write(text)
    return path


def _parse_quiet(path):
    """parse_geom with its warnings captured: (geom, [warning messages])."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        g = parse_geom(path)
    return g, [str(x.message) for x in w if issubclass(x.category, UserWarning)]


# geometry(5)'s two bad-region forms: panel-local fs/ss (needs `panel`), and lab-frame x/y
BAD_FS = ("badregionA/min_fs = 0\nbadregionA/max_fs = 10\nbadregionA/min_ss = 0\n"
          "badregionA/max_ss = 10\nbadregionA/panel = p0\n")
BAD_XY = ("bad_beamstop/min_x = -10\nbad_beamstop/max_x = 10\nbad_beamstop/min_y = -10\n"
          "bad_beamstop/max_y = 10\n")


def bad_region_checks(peaks):
    """s3-04: bad-region blocks are bad regions, not panels."""
    from glint import lute_bridge
    from glint.predict import panels_from_geom
    from glint.stream import panel_bounds, panel_at, _origin_panel
    from glint.glint_cli import _load_frames
    print("\n.geom bad regions (s3-04):")
    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")            # the warning is checked via _parse_quiet below
        g0, w0 = _parse_quiet(_write(d, "plain.geom", GEOM))
        q0 = peaks_to_q(peaks, g0)
        pf0, clen0 = panels_from_geom(g0)
        check("a .geom without bad regions parses with no warning and an empty bad_regions",
              lambda: (not w0 and g0.get("bad_regions") == {}, (w0, g0.get("bad_regions"))))
        stream = _write(d, "peaks.stream", "CrystFEL stream format 2.3\n----- Begin chunk -----\n"
                        "Image filename: synth.cxi\nEvent: //0\nPeaks from peak search\n"
                        "  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel\n"
                        + "".join(f"  {f:.2f}  {s:.2f}  0.0  1000.0  p0\n" for f, s in peaks)
                        + "End of peak list\n----- End chunk -----\n")

        def load(gpath):
            ns = argparse.Namespace(qframes=None, images=None, peaks=stream, geom=gpath,
                                    wavelength=None, min_peaks=6, N=0)
            return _load_frames(ns)[0]
        fr0 = load(_write(d, "plain_cli.geom", GEOM))

        cases = (("badregionA, panel fs/ss form", GEOM + BAD_FS, "badregionA",
                  {"min_fs": 0.0, "max_fs": 10.0, "min_ss": 0.0, "max_ss": 10.0, "panel": "p0"}),
                 ("bad_beamstop, lab x/y form", GEOM + BAD_XY, "bad_beamstop",
                  {"min_x": -10.0, "max_x": 10.0, "min_y": -10.0, "max_y": 10.0}),
                 ("badregionA block BEFORE the panel", BAD_FS + GEOM, "badregionA",
                  {"min_fs": 0.0, "max_fs": 10.0, "min_ss": 0.0, "max_ss": 10.0, "panel": "p0"}))
        for i, (label, text, bname, bkeys) in enumerate(cases):
            gpath = _write(d, f"bad{i}.geom", text)
            g, w = _parse_quiet(gpath)
            check(f"[{label}] parse_geom panels are ['p0'] (the bad region is not a panel)",
                  lambda: (list(g["panels"]) == ["p0"], list(g["panels"])))
            check(f"[{label}] ...and the region is kept in geom['bad_regions'] with its keys",
                  lambda: (g["bad_regions"] == {bname: bkeys}, g.get("bad_regions")))
            check(f"[{label}] peaks_to_q gives the SAME q as the file without it",
                  lambda: (lambda q: (q.shape == q0.shape and np.array_equal(q, q0), q.shape))(
                      peaks_to_q(peaks, g)))
            check(f"[{label}] predict.panels_from_geom gives the same panels",
                  lambda: (lambda pc: (pc[1] == clen0 and len(pc[0]) == len(pf0) and all(
                      a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
                      for a, b in zip(pc[0], pf0)), [p["name"] for p in pc[0]]))(panels_from_geom(g)))
            check(f"[{label}] lute_bridge.parse_geom agrees on the panel list",
                  lambda: (lambda lb: ([p["name"] for p in lb] == list(g["panels"]),
                                       ([p["name"] for p in lb], list(g["panels"]))))(
                      lute_bridge.parse_geom(gpath)[0]))
            check(f"[{label}] a warning says the region is read but not applied",
                  lambda: (any(bname in m and "not applied" in m for m in w), w))
            check(f"[{label}] the --peaks loader (glint_cli._load_frames) gives the same frames",
                  lambda: (lambda fr: (len(fr) == len(fr0) and all(
                      np.array_equal(a, b) for a, b in zip(fr, fr0)), [len(x) for x in fr]))(load(gpath)))

        # the stream writer labels peak rows by the panel bounds it reads off the same .geom text
        b = panel_bounds(BAD_FS + GEOM)
        check("stream.panel_bounds skips a bad region listed before the panel",
              lambda: ([n for n, *_ in b] == ["p0"], b))
        check("...so a peak inside it is labelled with its PANEL (p0), not the region",
              lambda: (panel_at(b, 5, 5) == "p0", panel_at(b, 5, 5)))
        check("stream._origin_panel does not pick the bad region either",
              lambda: (_origin_panel(BAD_FS + GEOM) == "p0", _origin_panel(BAD_FS + GEOM)))

        # anything else that reaches a panel consumer without a corner must be NAMED, not KeyError
        gnc, _ = _parse_quiet(_write(d, "nocorner.geom", GEOM + "p1/min_fs = 0\np1/max_fs = 9\n"))
        def named(fn):
            try:
                fn()
            except ValueError as exc:
                return "p1" in str(exc) and "corner_x" in str(exc), repr(exc)
            except Exception as exc:                   # noqa: BLE001
                return False, repr(exc)
            return False, "no exception"
        check("a panel block with no corner_x raises a ValueError naming it (peaks_to_q)",
              lambda: named(lambda: peaks_to_q(peaks, gnc)))
        check("...and in predict.panels_from_geom",
              lambda: named(lambda: panels_from_geom(gnc)))


def _raises_value_error(fn, *must_contain):
    try:
        fn()
    except ValueError as exc:
        return all(t in str(exc) for t in must_contain), repr(exc)
    except Exception as exc:                       # noqa: BLE001
        return False, repr(exc)
    return False, "no exception"


def direction_checks(peaks):
    """s3-05: CrystFEL direction terms without a coefficient, through ONE shared parser."""
    from glint import geom as G, lute_bridge
    print("\n.geom fs/ss directions (s3-05):")
    cases = {"x": (1, 0, 0), "+x": (1, 0, 0), "-x": (-1, 0, 0), "y": (0, 1, 0), "-y": (0, -1, 0),
             "z": (0, 0, 1), "-x +0.5y": (-1, 0.5, 0), "+1.0x +0.0y": (1, 0, 0),
             "-0.005723x +0.999984y": (-0.005723, 0.999984, 0), "+1.0e-03x -1y": (1e-3, -1, 0),
             "-0.999882x -0.000169y -0.015358z": (-0.999882, -0.000169, -0.015358)}
    for txt, want in cases.items():
        check(f"geom._vec({txt!r}) == {list(want)}",
              lambda: (lambda v: (np.array_equal(v, np.array(want, float)), v.tolist()))(G._vec(txt)))
    check("lute_bridge reads directions with the SAME parser as glint.geom",
          lambda: (lute_bridge._vec is G._vec, lute_bridge._vec))
    check("...so the two agree on every case above",
          lambda: (all(np.array_equal(lute_bridge._vec(t), G._vec(t)) for t in cases),
                   {t: np.asarray(lute_bridge._vec(t)).tolist() for t in cases
                    if not np.array_equal(lute_bridge._vec(t), G._vec(t))}))
    for txt in ("", "q", "1.0", "+1.0x junk", "1x 1x", "X"):
        check(f"an unreadable direction {txt!r} raises ValueError instead of a silent zero",
              lambda: _raises_value_error(lambda: G._vec(txt), repr(txt)))

    with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q0 = peaks_to_q(peaks, parse_geom(_write(d, "plain.geom", GEOM)))   # round-trip reference
        DIRS = "p0/fs = +1.0x +0.0y\np0/ss = +0.0x +1.0y\n"
        flip = GEOM.replace(f"p0/corner_y = {CY}", f"p0/corner_y = {-CY}")   # ss = -y needs it
        variants = (("fs = x, ss = y", GEOM, "p0/fs = x\np0/ss = y\n", "p0/fs = +1.0x\np0/ss = +1.0y\n"),
                    ("fs = +x, ss = +y", GEOM, "p0/fs = +x\np0/ss = +y\n", "p0/fs = +1.0x\np0/ss = +1.0y\n"),
                    ("fs = x, ss = -y", flip, "p0/fs = x\np0/ss = -y\n", "p0/fs = +1.0x\np0/ss = -1.0y\n"))
        for i, (label, base, bare_dirs, num_dirs) in enumerate(variants):
            bare = _write(d, f"bare{i}.geom", base.replace(DIRS, bare_dirs))
            numeric = _write(d, f"num{i}.geom", base.replace(DIRS, num_dirs))
            qn = peaks_to_q(peaks, parse_geom(numeric))
            if i == 0:
                check("(control) the explicit `+1.0x`/`+1.0y` file reproduces the round-trip reference q",
                      lambda: (np.array_equal(qn, q0), "differs"))
            check(f"[{label}] peaks_to_q equals the explicit-coefficient file",
                  lambda: (lambda qb: (qb.shape == qn.shape and np.array_equal(qb, qn),
                                       f"{len(np.unique(np.round(qb, 9), axis=0))} distinct q of {len(qb)}"))(
                      peaks_to_q(peaks, parse_geom(bare))))
            qln = lute_bridge.peaks_to_q(peaks[:, 0], peaks[:, 1], lute_bridge.parse_geom(numeric)[0], CLEN, LAM)
            check(f"[{label}] lute_bridge.peaks_to_q equals the explicit-coefficient file",
                  lambda: (lambda qb: (np.array_equal(qb, qln, equal_nan=True),
                                       f"{len(np.unique(np.round(qb, 9), axis=0))} distinct q"))(
                      lute_bridge.peaks_to_q(peaks[:, 0], peaks[:, 1], lute_bridge.parse_geom(bare)[0],
                                             CLEN, LAM)))

        from glint.stream import _FALLBACK_GEOM
        gf = parse_geom(_write(d, "fallback.geom", _FALLBACK_GEOM))["panels"]["p0"]
        check("GLINT's own stream-header fallback geometry (fs = x, ss = y) parses to a unit basis",
              lambda: ((gf["fsx"], gf["fsy"], gf["ssx"], gf["ssy"]) == (1.0, 0.0, 0.0, 1.0),
                       (gf["fsx"], gf["fsy"], gf["ssx"], gf["ssy"])))

        degen = GEOM.replace("p0/fs = +1.0x +0.0y", "p0/fs = +0.0x +0.0y")
        gd = _write(d, "degen.geom", degen)
        check("a panel whose fs/ss span no area raises ValueError naming it (geom.parse_geom)",
              lambda: _raises_value_error(lambda: parse_geom(gd), "p0"))
        check("...and lute_bridge.parse_geom",
              lambda: _raises_value_error(lambda: lute_bridge.parse_geom(gd), "p0"))


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile("w", suffix=".geom", delete=False) as f:
        f.write(GEOM); gpath = f.name
    geom = parse_geom(gpath)
    print(f"parsed geom: {len(geom['panels'])} panel(s), wavelength={geom['wavelength_A']:.4f} A "
          f"(expect {LAM})")

    nidx = nlyso = qmax_err = 0; trials = 12
    worst = 0.0
    for s in range(trials):
        q = ewald_spots(s)
        if len(q) < 10:
            continue
        peaks, q_ref = project(q)
        on = (peaks[:, 0] >= 0) & (peaks[:, 0] < NPX) & (peaks[:, 1] >= 0) & (peaks[:, 1] < NPX)
        peaks = peaks[on]; q_ref = q_ref[on]
        qb = peaks_to_q(peaks, geom)                        # the bridge under test
        err = np.linalg.norm(qb - q_ref, axis=1).max()      # pixel<->q round-trip (must be exact)
        worst = max(worst, err)
        r = index_shot(qb, float(np.linalg.norm(qb, axis=1).max()))
        ok = r.M is not None and same_lattice(r.M, LYSO)
        nidx += r.M is not None; nlyso += ok
        if s < 4:
            cell = np.round(np.sort(np.linalg.norm(r.M, axis=0)), 1) if r.M is not None else None
            print(f"  trial {s}: {len(qb):2d} spots  bridge_err={err:.2e} 1/A  cell={cell}  lyso?={ok}")
    ok_roundtrip = worst < 1e-9
    ok_index = nlyso >= trials - 1
    print(f"\nbridge round-trip max error: {worst:.2e} 1/A  ({'PASS' if ok_roundtrip else 'CHECK'})")
    print(f"indexed {nidx}/{trials}, correct lysozyme {nlyso}/{trials}  "
          f"({'PASS' if ok_index else 'CHECK'})")

    # --- CrystFEL peak-search stream input path (what LUTE/peakfinder8 emits) ---
    from glint.geom import read_crystfel_peaks
    buf = ["CrystFEL stream format 2.3\n"]
    expect = []
    for s in range(3):
        peaks, _ = project(ewald_spots(s + 100))
        on = (peaks[:, 0] >= 0) & (peaks[:, 0] < NPX) & (peaks[:, 1] >= 0) & (peaks[:, 1] < NPX)
        peaks = peaks[on]
        buf.append("----- Begin chunk -----\nImage filename: synth.cxi\nEvent: //%d\n" % s)
        buf.append("num_peaks = %d\nPeaks from peak search\n  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel\n" % len(peaks))
        for fs, ss in peaks:
            buf.append(f"  {fs:.2f}  {ss:.2f}  0.0  1000.0  p0\n")
        buf.append("End of peak list\n----- End chunk -----\n")
        expect.append(len(peaks))
    with tempfile.NamedTemporaryFile("w", suffix=".stream", delete=False) as f:
        f.write("".join(buf)); spath = f.name
    chunks = read_crystfel_peaks(spath); os.unlink(spath); os.unlink(gpath)
    ok_stream = 0
    for ch, npk in zip(chunks, expect):
        assert len(ch["peaks"]) == npk, f"peak count {len(ch['peaks'])} != {npk}"
        qb = peaks_to_q(ch["peaks"], geom)
        r = index_shot(qb, float(np.linalg.norm(qb, axis=1).max()))
        ok_stream += r.M is not None and same_lattice(r.M, LYSO)
    ok_peakstream = ok_stream == len(chunks)
    print(f"CrystFEL peak-stream input: {len(chunks)} chunks read, "
          f"{ok_stream}/{len(chunks)} indexed to lysozyme  "
          f"({'PASS' if ok_peakstream else 'CHECK'})")

    pk, _ = project(ewald_spots(0))
    pk = pk[(pk[:, 0] >= 0) & (pk[:, 0] < NPX) & (pk[:, 1] >= 0) & (pk[:, 1] < NPX)]
    bad_region_checks(pk)
    direction_checks(pk)
    print(f"\n.geom grammar FAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + "; ".join(FAILS)))
    raise SystemExit(0 if (ok_roundtrip and ok_index and ok_peakstream and not FAILS) else 1)
