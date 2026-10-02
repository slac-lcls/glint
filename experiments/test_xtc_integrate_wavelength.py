"""glint_xtc --integrate must predict each event at the wavelength pass 1 indexed it at (review s7-03).

THE DEFECT. `--wavelength` defaults to 0.0, meaning "use the per-event photon energy". Pass 1 did:
it built each event's q at that event's EBeam wavelength. But the reader kept none of them, and pass 2
(`integrate_and_write`) predicted with `args.wavelength`, i.e. 0.0. predict_spots then gated on a plane
through the origin instead of the Ewald sphere and projected every reflection onto the beam centre:
a central-hole detector got a header-only stream, a beam-covering one got every row on the direct
beam, and the run exited 0 either way. The header said photon_energy_eV = 9392.70, a constant.

WHAT RUNS. The real xtc route end to end, through `glint_xtc.main`:
  real   xtc_qreader_psana1.run_to_qframes_psana1 (pass 1) and frames_for_events (pass 2),
         geom_coords.coords_from_geom, xtc_core.prep_geometry / frame_q, glint_xtc.read_qframes /
         index_and_write / integrate_and_write, glint.predict.predict_spots / integrate_spots /
         write_stream_integrated, and the .stream they write, which is what is checked.
  mocked psana (DataSource, the area detector, EBeam with a per-event photon energy), cupy (numpy
         aliases; this runner has none), PeakFinderV4 (a local-maximum finder over the synthetic
         frames), and glint.hybrid_stream (torch): a perfect indexer that returns the event's true
         lattice in whichever axis-sign basis fits the q pass 1 actually built.
Ground truth is an independent forward model in CrystFEL's frame (k_in along +z, detector plane at
z = +clen, pixel = (corner + fs*fs_vec + ss*ss_vec)/res), at each event's own wavelength.

Plain script: prints ok/FAIL lines and exits non-zero on any failure (CI runs it as a script).
"""
import contextlib
import io
import itertools
import os
import re
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
XB = ROOT / "experiments" / "xtc_bridge"
sys.path.insert(0, str(XB))
sys.path.insert(0, str(ROOT))
# psana-metadata diagnostic with its own suite (xtc_bridge/test_geom_manifest.py); nothing to check
# against a mock DataSource.
os.environ["GLINT_GEOM_MANIFEST"] = "off"

# glint modules first, so nothing in the package sees the cupy stub below as a real GPU.
import glint.predict as gp                      # noqa: E402
from glint.lattice import cell_to_Ar            # noqa: E402
import glint.lute_bridge                        # noqa: E402,F401
import glint.stream                             # noqa: E402,F401

# cupy -> numpy aliases: prep_geometry and _PanelFinders only move masks/frames through it.
_cp = types.ModuleType("cupy")
_cp.asarray = lambda a, *k, **kw: np.asarray(a)
_cp.asnumpy = lambda a: np.asarray(a)
_cp.float32 = np.float32
_cp.zeros = np.zeros
sys.modules["cupy"] = _cp

import xtc_core                                 # noqa: E402
import glint_xtc                                # noqa: E402
from scipy.ndimage import maximum_filter        # noqa: E402

HC = xtc_core.HC_EV_A
FAILS = []


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAILS.append(msg)


# ---------------------------------------------------------------- synthetic experiment
NSEG, H, W = 4, 256, 256           # four quadrants around a central hole (Jungfrau/Epix-like)
GAP = 10                           # px from the beam axis to each quadrant's inner edge
RES = 1.0 / 100e-6                 # px/m
ZDIST = 0.08                       # m
DMIN, TOL = 3.0, 0.002             # pass-2 prediction limits (--int-dmin / --int-tol)
CELL = (61.2, 74.5, 41.3, 88.0, 96.0, 91.0)
CORNERS = [(-W - GAP, -H - GAP), (GAP, -H - GAP), (-W - GAP, GAP), (GAP, GAP)]   # (corner_x, corner_y) px


def write_geom(path):
    lines = ["; synthetic 4-quadrant geometry for test_xtc_integrate_wavelength", f"res = {RES}",
             f"clen = {ZDIST}", "coffset = 0.0", "adu_per_photon = 1", ""]
    for i, (cx, cy) in enumerate(CORNERS):
        n = f"q{i}"
        lines += [f"{n}/min_fs = 0", f"{n}/max_fs = {W - 1}",
                  f"{n}/min_ss = {i * H}", f"{n}/max_ss = {i * H + H - 1}",
                  f"{n}/fs = +1.0x +0.0y", f"{n}/ss = +0.0x +1.0y",
                  f"{n}/corner_x = {cx}", f"{n}/corner_y = {cy}", ""]
    Path(path).write_text("\n".join(lines))


def lab_to_slab(x, y):
    """Lab (x, y) m on the z = +ZDIST plane -> slab (fs, ss); NaN off the panels. CrystFEL's
    convention, written out here rather than taken from glint's parser."""
    fs = np.full(len(x), np.nan); ss = np.full(len(x), np.nan)
    for i, (cx, cy) in enumerate(CORNERS):
        lf, ls = x * RES - cx, y * RES - cy
        on = (lf >= 0) & (lf <= W - 1) & (ls >= 0) & (ls <= H - 1)
        fs[on], ss[on] = lf[on], i * H + ls[on]
    return fs, ss


def forward(M, lam, tol):
    """Independent CrystFEL-frame forward model -> (hkl (n,3) int, fs (n,), ss (n,)) on the panels."""
    A = np.asarray(M, float)
    R = np.linalg.inv(A)                                   # rows a*, b*, c*: q = h @ R
    qmax = 1.0 / DMIN
    n = [int(np.ceil(qmax * np.linalg.norm(A[:, i]))) + 1 for i in range(3)]   # |h_i| <= qmax |a_i|
    g = np.stack(np.meshgrid(*[np.arange(-k, k + 1) for k in n], indexing="ij"), -1).reshape(-1, 3)
    g = g[np.any(g != 0, axis=1)]
    q = g @ R
    kin = np.array([0.0, 0.0, 1.0 / lam])
    kout = kin + q
    exc = np.linalg.norm(kout, axis=1) - 1.0 / lam
    ok = (np.linalg.norm(q, axis=1) <= qmax) & (np.abs(exc) < tol) & (kout[:, 2] > 0)
    g, kout = g[ok], kout[ok]
    t = ZDIST / kout[:, 2]
    fs, ss = lab_to_slab(kout[:, 0] * t, kout[:, 1] * t)
    on = np.isfinite(fs)
    return g[on], fs[on], ss[on]


def rot(rng):
    a, b, c, d = (lambda v: v / np.linalg.norm(v))(rng.normal(size=4))
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def make_run(energies_eV, render_eV, seed):
    """One synthetic run. energies_eV: what EBeam reports per event (inf = unusable, as on
    mfxx49820 r0016). render_eV: the energy each event's pattern is actually drawn at."""
    rng = np.random.default_rng(seed)
    Ar = cell_to_Ar(*CELL)
    events = []
    yy, xx = np.mgrid[0:NSEG * H, 0:W]
    for e, eV in enumerate(render_eV):
        M = rot(rng) @ Ar
        lam = HC / eV
        # Spots drawn a little wider than pass 2's gate, so every reflection pass 2 predicts has one.
        hkl, fs, ss = forward(M, lam, 1.5 * TOL)
        img = rng.poisson(2.0, size=(NSEG * H, W)).astype(np.float32)
        for f, s in zip(fs, ss):
            sl = (slice(max(int(s) - 5, 0), int(s) + 6), slice(max(int(f) - 5, 0), int(f) + 6))
            gauss = np.exp(-((xx[sl] - f) ** 2 + (yy[sl] - s) ** 2) / 2.0) / (2 * np.pi)
            img[sl] += rng.poisson(2000.0 * gauss).astype(np.float32)
        events.append(dict(M=M, lam=lam, hkl=hkl, fs=fs, ss=ss, img=img, ebeam_eV=energies_eV[e]))
    return events


# ---------------------------------------------------------------- mocks: psana, finder, indexer
RUN = []                 # the synthetic run the mocks serve (set per case)
HAND = [1.0]             # which of the two equally good bases the mock indexer returns (det sign)
SPY = {"lam": [], "pred_calls": 0, "reread": 0}


class _Evt:
    def __init__(self, i):
        self.i = i


class _AreaDet:
    def calib(self, evt):
        return RUN[evt.i]["img"].reshape(NSEG, H, W).copy()

    def raw(self, evt):
        return None

    def coords_x(self, evt):
        return COORDS[0]

    def coords_y(self, evt):
        return COORDS[1]

    def coords_z(self, evt):
        return COORDS[2]

    def mask(self, evt, **kw):
        return np.ones((NSEG, H, W), int)


class _EBeamData:
    def __init__(self, eV):
        self.eV = eV

    def ebeamPhotonEnergy(self):
        return self.eV


class _EBeam:
    def get(self, evt):
        return _EBeamData(RUN[evt.i]["ebeam_eV"])


class _DataSource:
    def __init__(self, spec):
        self.spec = spec

    def events(self):
        return (_Evt(i) for i in range(len(RUN)))

    def env(self):
        raise AssertionError("env() is only reached on a Detector() KeyError")


def _Detector(name):
    return _EBeam() if name == "EBeam" else _AreaDet()


_psana = types.ModuleType("psana")
_psana.DataSource = _DataSource
_psana.Detector = _Detector
_psana.setOption = lambda *a, **k: None
sys.modules["psana"] = _psana

import xtc_qreader_psana1 as rd               # noqa: E402  (imports psana lazily, inside functions)

_frames_for_events = rd.frames_for_events


def _counting_frames_for_events(*a, **kw):
    SPY["reread"] += 1
    yield from _frames_for_events(*a, **kw)


rd.frames_for_events = _counting_frames_for_events


class _LocalMaxFinder:
    """Stands in for PeakFinderV4 (cupy kernels): every local maximum above a flat threshold. The
    synthetic spots peak at ~300 photons on a background of 2, so this is a perfect finder."""

    def __init__(self, *a, **k):
        pass

    def find(self, panel):
        a = np.asarray(panel, float)
        y, x = np.nonzero((a == maximum_filter(a, size=5)) & (a > 60.0))
        return {"x": x.astype(float), "y": y.astype(float)}


xtc_core.load_peakfinder_v4 = lambda: types.SimpleNamespace(PeakFinderV4=_LocalMaxFinder)

_SIGNS = [np.diag(s) for s in itertools.product((1.0, -1.0), repeat=3)]
CHOSEN = {}


def _perfect_hybrid_index(frames, images, Mc_known=None, nbest=3, **kw):
    """The event's true lattice in the basis S @ M_true (S an axis-sign matrix) that best indexes the
    q pass 1 actually built: the frame pass 1 works in is MEASURED here, not assumed. S and -S fit
    equally (a lattice is centrosymmetric); HAND picks between them by determinant sign, standing in
    for whichever handedness a real indexer returns."""
    out = []
    for q, im in zip(frames, images):
        Mt = RUN[int(im["event"])]["M"]
        best = None
        for S in _SIGNS:
            M = S @ Mt
            h = np.asarray(q, float) @ M
            frac = float(np.mean(np.all(np.abs(h - np.rint(h)) < 0.15, axis=1)))
            key = (round(frac, 6), HAND[0] * np.sign(np.linalg.det(M)))
            if best is None or key > best[0]:
                best = (key, M, S)
        CHOSEN[int(im["event"])] = (np.diag(best[2]).tolist(), best[0][0])
        out.append({"image": im["image"], "event": im["event"],
                    "M": best[1] if best[0][0] > 0.5 else None})
    return out, {}


_hs = types.ModuleType("glint.hybrid_stream")
_hs.hybrid_index = _perfect_hybrid_index
_hs._report = lambda stats, path: None
sys.modules["glint.hybrid_stream"] = _hs

_predict_spots = gp.predict_spots


def _spy_predict(M, panels, clen_m, wavelength_A, **kw):
    SPY["pred_calls"] += 1
    SPY["lam"].append(wavelength_A)
    return _predict_spots(M, panels, clen_m, wavelength_A, **kw)


gp.predict_spots = _spy_predict            # integrate_and_write imports it from glint.predict at call time

_read_qframes = glint_xtc.read_qframes
LAST_OUT = {}


def _keep_out(*a, **kw):
    out = _read_qframes(*a, **kw)
    LAST_OUT.clear(); LAST_OUT.update(out)
    return out


glint_xtc.read_qframes = _keep_out


# ---------------------------------------------------------------- running + reading the stream
def read_stream(path):
    """-> {event: {photon_eV, recip (3,3) 1/A or None, rows (n,9) h k l I sig peak bg fs ss}}."""
    chunks, cur, inref = {}, None, False
    for line in open(path):
        if line.startswith("----- Begin chunk"):
            cur = {"photon_eV": None, "recip": [], "rows": []}
        elif line.startswith("Event:") and cur is not None:
            chunks[int(re.findall(r"\d+", line)[-1])] = cur
        elif line.startswith("photon_energy_eV") and cur is not None:
            cur["photon_eV"] = float(line.split("=")[1])
        elif re.match(r"[abc]star = ", line) and cur is not None:
            cur["recip"].append([float(v) / 10.0 for v in line.split("=")[1].split()[:3]])   # nm^-1 -> 1/A
        elif line.startswith("   h    k    l"):
            inref = True
        elif line.startswith("End of reflections"):
            inref = False
        elif inref:
            cur["rows"].append([float(v) for v in line.split()[:9]])
    for c in chunks.values():
        c["rows"] = np.array(c["rows"]).reshape(-1, 9)
        c["recip"] = np.array(c["recip"]) if len(c["recip"]) == 3 else None
    return chunks


def run_main(tmp, tag, extra_argv=()):
    gpath = os.path.join(tmp, "quad.geom")
    out = os.path.join(tmp, f"{tag}.stream")
    SPY.update(lam=[], pred_calls=0, reread=0)
    argv = ["--exp", "synthx", "--run", "1", "--zdist", str(ZDIST), "--psana", "1",
            "--geom", gpath, "--integrate", "--int-dmin", str(DMIN), "--int-tol", str(TOL),
            "-o", out, *extra_argv]
    log = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(log):
        try:
            glint_xtc.main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
    return code, (read_stream(out) if os.path.exists(out) else {}), log.getvalue()


def indexed_events():
    return [e for e, r in enumerate(RUN) if np.isfinite(r["ebeam_eV"])]


# ---------------------------------------------------------------- cases
def case_per_event_wavelength(tmp):
    """No --wavelength: every event at its own EBeam energy, one of them unusable (inf)."""
    print("\n[per-event wavelength] no --wavelength; EBeam 9300..9900 eV, event 2 = inf")
    energies = [9300.0, 9450.0, np.inf, 9600.0, 9750.0, 9900.0]
    RUN[:] = make_run(energies, [e if np.isfinite(e) else 9500.0 for e in energies], seed=301)
    HAND[0] = 1.0
    CHOSEN.clear()
    code, chunks, log = run_main(tmp, "per_event")
    want = indexed_events()
    lam_true = {e: RUN[e]["lam"] for e in want}
    check(code == 0, f"glint_xtc.main exits 0 (got {code})")
    check(sorted(CHOSEN) == want and all(f > 0.5 for _, f in CHOSEN.values()),
          f"(setup) the mock indexer fits every indexed event's pass-1 q: "
          f"{ {e: (s, round(f, 2)) for e, (s, f) in sorted(CHOSEN.items())} }")
    lams = LAST_OUT.get("lams")
    check(lams is not None and LAST_OUT.get("events") == want
          and np.allclose(lams, [lam_true[e] for e in want], rtol=0, atol=1e-12),
          f"pass 1 returns each indexed event's wavelength (events {LAST_OUT.get('events')}, "
          f"lams {None if lams is None else np.round(lams, 5).tolist()}; "
          f"skipped no-wavelength {LAST_OUT.get('n_skipped_wl')})")
    used = SPY["lam"]
    check(len(used) == len(want) and np.allclose(used, [lam_true[e] for e in want], rtol=0, atol=1e-12),
          f"pass 2 predicts each event at its own wavelength (used {np.round(used, 5).tolist()})")
    check(sorted(chunks) == want and all(len(chunks[e]["rows"]) for e in want),
          f"one integrated chunk per indexed event (events {sorted(chunks)}, want {want})")
    ph = {e: chunks[e]["photon_eV"] for e in chunks}
    check(len(ph) == len(want) and all(e in ph and abs(ph[e] - RUN[e]["ebeam_eV"]) < 0.006 for e in want),
          f"each chunk's photon_energy_eV is its event's energy ({ph})")
    return chunks


def case_explicit_wavelength(tmp):
    """--wavelength given: unchanged behaviour, and EBeam is never needed (all inf here)."""
    print("\n[explicit --wavelength] EBeam inf on every event; patterns drawn at 9500 eV")
    lam0 = HC / 9500.0
    RUN[:] = make_run([np.inf] * 4, [9500.0] * 4, seed=302)
    HAND[0] = 1.0
    code, chunks, log = run_main(tmp, "explicit", ["--wavelength", repr(lam0)])
    want = list(range(len(RUN)))
    check(code == 0, f"glint_xtc.main exits 0 (got {code})")
    check(len(SPY["lam"]) == len(want) and all(lam == lam0 for lam in SPY["lam"]),
          f"pass 2 predicts at --wavelength exactly (used {np.round(SPY['lam'], 6).tolist()})")
    check(sorted(chunks) == want, f"one integrated chunk per event (events {sorted(chunks)})")
    check(all(f"{chunks[e]['photon_eV']:.2f}" == f"{HC / lam0:.2f}" for e in chunks),
          f"photon_energy_eV = 12398.42/--wavelength in every chunk "
          f"({sorted({c['photon_eV'] for c in chunks.values()})})")
    return chunks


def case_refusals(tmp):
    """A reader dict without per-event wavelengths (older reader, a mock, psana2's) and no
    --wavelength: pass 2 must refuse before re-reading the run, never predict at 0."""
    print("\n[refusal] no --wavelength and no per-event wavelengths reach pass 2")
    RUN[:] = make_run([9500.0] * 3, [9500.0] * 3, seed=303)
    HAND[0] = 1.0
    gpath = os.path.join(tmp, "quad.geom")
    args = glint_xtc.build_parser().parse_args(
        ["--exp", "synthx", "--run", "1", "--zdist", str(ZDIST), "--geom", gpath, "--integrate",
         "--int-dmin", str(DMIN), "--int-tol", str(TOL), "-o", os.path.join(tmp, "refuse.stream")])
    with contextlib.redirect_stdout(io.StringIO()):
        out = _read_qframes(args)
    out.pop("lams", None)
    SPY.update(lam=[], pred_calls=0, reread=0)
    msg = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            glint_xtc.index_and_write(out, args, args.out, report=False)
    except SystemExit as e:
        msg = str(e.code)
    check(msg is not None and "wavelength" in msg,
          f"index_and_write refuses with a message naming the wavelength ({(msg or 'no refusal')[:70]}...)")
    check(SPY["pred_calls"] == 0 and SPY["reread"] == 0,
          f"...before re-reading the run or predicting anything (re-reads {SPY['reread']}, "
          f"predict_spots calls {SPY['pred_calls']}, wavelengths {SPY['lam']})")

    out["lams"] = [HC / 9500.0] * (len(out["events"]) - 1)
    err = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            glint_xtc.index_and_write(out, args, args.out, report=False)
    except (ValueError, SystemExit) as e:
        err = e
    check(isinstance(err, ValueError), f"a wavelength list that does not align with events is an error ({err!r})")


def case_predict_guard():
    print("\n[predict_spots] a wavelength that is not a finite positive number is an error")
    panels = [{"min_fs": 0, "max_fs": 1023, "min_ss": 0, "max_ss": 1023, "fs": (1.0, 0.0, 0.0),
               "ss": (0.0, 1.0, 0.0), "cx": -512.0, "cy": -512.0, "res": RES, "coffset": 0.0}]
    M = cell_to_Ar(*CELL)
    for bad in (0.0, -1.3, float("nan"), float("inf"), None):
        try:
            _predict_spots(M, panels, ZDIST, bad, dmin=DMIN, tol=TOL)
            raised = "returned"
        except ValueError:
            raised = "ValueError"
        except Exception as e:                       # noqa: BLE001  (record what it did instead)
            raised = type(e).__name__
        check(raised == "ValueError", f"predict_spots(wavelength_A={bad!r}) raises ValueError ({raised})")
    check(len(_predict_spots(M, panels, ZDIST, 1.3, dmin=DMIN, tol=TOL)) > 0,
          "predict_spots at 1.3 A still predicts")


def main():
    global COORDS
    with tempfile.TemporaryDirectory() as tmp:
        write_geom(os.path.join(tmp, "quad.geom"))
        import geom_coords
        X, Y, _ = geom_coords.coords_from_geom(os.path.join(tmp, "quad.geom"), (NSEG, H, W), ZDIST)
        COORDS = (X.reshape(NSEG, H, W), Y.reshape(NSEG, H, W), np.full((NSEG, H, W), -0.1e6))
        case_per_event_wavelength(tmp)
        case_explicit_wavelength(tmp)
        case_refusals(tmp)
        case_predict_guard()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED")
        sys.exit(1)
    print("ALL PASS")


COORDS = None

if __name__ == "__main__":
    main()
