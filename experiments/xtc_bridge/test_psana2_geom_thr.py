"""--geom on the psana2 reader, and --pf8-thr-adu on both readers and the CLI.

Runs without psana, cupy or a GPU: psana2's DataSource and GeometryAccess are stubs, and
xtc_core.prep_geometry / frame_q are replaced by recorders, so what is under test is only which
coordinates and which ADU floor reach the peak-finder setup. Checks:

  * psana2 + --geom: X/Y come from the .geom (geom_coords.coords_from_geom), and psana's calibconst
    geometry is never loaded. Before this change the CLI exited on --psana 2 --geom, so a run without
    a deployed refinement silently used calibconst's placeholder.
  * psana2 without --geom: calibconst's coordinates, exactly as before.
  * pf8_thr_adu reaches prep_geometry from both readers, and None stays the default.
  * the CLI: --pf8-thr-adu parses, defaults to None, and travels with --geom over the bridge as the
    two LAST positional arguments, so older callers of run_to_qframes are unaffected; --calib-dir is
    still refused on psana2.
"""
from __future__ import annotations

import os, sys, tempfile, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        FAILS.append(name)


NSEG, H, W = 2, 4, 6
PSANA_X = np.full(NSEG * H * W, 777.0)          # a value no .geom below can produce


class FakeGeometryAccess:
    loaded = 0

    def load_pars_from_str(self, s):
        FakeGeometryAccess.loaded += 1

    def get_pixel_coords(self):
        return PSANA_X.copy(), PSANA_X.copy(), np.full(PSANA_X.size, -100000.0)


class FakeRaw:
    def calib(self, evt):
        return np.zeros((NSEG, H, W), np.float32)


class FakeEnergyRaw:
    def ebeamPhotonEnergy(self, evt):
        return (10000.0, 11000.0, 12000.0)[evt]


class FakeDetector:
    calibconst = {"geometry": ["# EXPERIMENT any (placeholder)"]}
    raw = FakeRaw()


class FakeEnergyDetector:
    raw = FakeEnergyRaw()


class FakeRun:
    def Detector(self, name):
        if name == "ebeamh":
            return FakeEnergyDetector()
        return FakeDetector()

    def events(self):
        return iter(range(3))


class FakeDataSource:
    def __init__(self, **kw):
        pass

    def runs(self):
        return iter([FakeRun()])


def install_fake_psana2():
    mods = {"psana": types.ModuleType("psana"),
            "psana.pscalib": types.ModuleType("psana.pscalib"),
            "psana.pscalib.geometry": types.ModuleType("psana.pscalib.geometry"),
            "psana.pscalib.geometry.GeometryAccess": types.ModuleType("psana.pscalib.geometry.GeometryAccess")}
    mods["psana"].DataSource = FakeDataSource
    mods["psana.pscalib.geometry.GeometryAccess"].GeometryAccess = FakeGeometryAccess
    sys.modules.update(mods)


def write_geom(path):
    """Two 4x6 panels stacked in ss, 100 um pixels, panel 1 offset by 50 px in x."""
    with open(path, "w") as f:
        f.write("res = 10000\n")
        for i in range(NSEG):
            f.write(f"p{i}/min_fs = 0\np{i}/max_fs = {W - 1}\np{i}/min_ss = {i * H}\np{i}/max_ss = {i * H + H - 1}\n"
                    f"p{i}/fs = +1.0x +0.0y\np{i}/ss = +0.0x +1.0y\n"
                    f"p{i}/corner_x = {-10 + 50 * i}\np{i}/corner_y = {-20 + 10 * i}\n")


install_fake_psana2()
import xtc_core
import xtc_qreader
import geom_coords

seen = []


def fake_prep(Xf, Yf, Zf, shape, good, zdist, **kw):
    seen.append({"Xf": np.asarray(Xf).ravel().copy(), "kw": kw})
    return np.zeros(1), np.zeros(1), -0.1, np.array([0, 0, -1.0]), []    # X not None: built once


xtc_core.prep_geometry = fake_prep
xtc_core.frame_q = lambda *a, return_px=False, **k: (
    (np.ones((1, 3)), np.array([[0, 1, 2]]), np.array([5.0]))
    if return_px else np.ones((1, 3)))

tmp = tempfile.mkdtemp()
geom = os.path.join(tmp, "tiny.geom")
write_geom(geom)
want_X, _, _ = geom_coords.coords_from_geom(geom, (NSEG, H, W), 0.1)

print("psana2 reader: geometry source")
seen.clear(); FakeGeometryAccess.loaded = 0
fixed_wl = xtc_qreader.run_to_qframes("exp", 1, zdist=0.1, wavelength=1.3, geom=geom)
check("--geom: X comes from the .geom", len(seen) == 1 and np.array_equal(seen[0]["Xf"], want_X),
      f"got {seen[0]['Xf'][:3] if seen else None}")
check("--geom: calibconst geometry not loaded", FakeGeometryAccess.loaded == 0,
      f"loaded {FakeGeometryAccess.loaded}x")
check("psana2 returns q-frame wavelengths", fixed_wl["lams"] == [1.3] * 3, fixed_wl["lams"])

event_wl = xtc_qreader.run_to_qframes("exp", 1, zdist=0.1)
want_lams = xtc_core.HC_EV_A / np.array([10000.0, 11000.0, 12000.0])
check("psana2 returns each event's wavelength", np.allclose(event_wl["lams"], want_lams),
      event_wl["lams"])

seen.clear(); FakeGeometryAccess.loaded = 0
xtc_qreader.run_to_qframes("exp", 1, zdist=0.1, wavelength=1.3)
check("no --geom: X comes from calibconst", len(seen) == 1 and np.array_equal(seen[0]["Xf"], PSANA_X))
check("no --geom: calibconst loaded once", FakeGeometryAccess.loaded == 1)

print("pf8_thr_adu reaches prep_geometry")
check("psana2 default is None", seen[0]["kw"].get("pf8_thr_adu", "MISSING") is None,
      repr(seen[0]["kw"].get("pf8_thr_adu", "MISSING")))
seen.clear()
xtc_qreader.run_to_qframes("exp", 1, zdist=0.1, wavelength=1.3, peakfinder="pf8", pf8_thr_adu=5000.0)
check("psana2 forwards 5000", seen and seen[0]["kw"].get("pf8_thr_adu") == 5000.0)

import inspect
import xtc_qreader_psana1
sig = inspect.signature(xtc_qreader_psana1.run_to_qframes_psana1)
check("psana1 reader accepts pf8_thr_adu (default None)",
      "pf8_thr_adu" in sig.parameters and sig.parameters["pf8_thr_adu"].default is None)
src = inspect.getsource(xtc_qreader_psana1.run_to_qframes_psana1)
check("psana1 reader passes it to prep_geometry", "pf8_thr_adu=pf8_thr_adu" in src)

print("CLI")
import glint_xtc
ap = glint_xtc.build_parser()
base = ["--exp", "x", "--run", "1", "--zdist", "0.1", "-o", os.path.join(tmp, "o.stream")]
a = ap.parse_args(base)
check("--pf8-thr-adu defaults to None", a.pf8_thr_adu is None)
a = ap.parse_args(base + ["--pf8-thr-adu", "5000"])
check("--pf8-thr-adu 5000 parses", a.pf8_thr_adu == 5000.0)

calls = []
eb = types.ModuleType("envbridge")


class _Env:
    @staticmethod
    def conda(env, stack=None, pythonpath=None):
        return ("env", env)


worker_timeouts = []


class _Worker:
    def __init__(self, env, timeout=120.0, call_timeout="same"):
        worker_timeouts.append(call_timeout)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def call(self, fn, *args):
        calls.append((fn, args))
        return {"qframes": [], "events": []}


eb.Env = _Env
eb.Worker = _Worker
sys.modules["envbridge"] = eb

a = ap.parse_args(base + ["--psana", "2", "--geom", geom, "--peakfinder", "pf8", "--pf8-thr-adu", "5000"])
try:
    glint_xtc.read_qframes(a, verbose=False)
    exited = None
except SystemExit as e:
    exited = str(e)
check("--psana 2 --geom is accepted", exited is None, exited or "")
check("bridge gets geom (absolute) and the floor as the last two args",
      calls and calls[-1][1][-2:] == (os.path.realpath(geom), 5000.0), repr(calls[-1][1][-2:] if calls else None))
a = ap.parse_args(base + ["--psana", "2"])
calls.clear(); glint_xtc.read_qframes(a, verbose=False)
check("no --geom: bridge gets (None, None)", calls and calls[-1][1][-2:] == (None, None))
check("--bridge-timeout defaults to no limit on the reader call", worker_timeouts[-1] is None,
      repr(worker_timeouts[-1]))
a = ap.parse_args(base + ["--psana", "2", "--bridge-timeout", "21600"])
glint_xtc.read_qframes(a, verbose=False)
check("--bridge-timeout 21600 reaches the worker", worker_timeouts[-1] == 21600.0, repr(worker_timeouts[-1]))
a = ap.parse_args(base + ["--psana", "2", "--calib-dir", tmp])
try:
    glint_xtc.read_qframes(a, verbose=False)
    refused = False
except SystemExit:
    refused = True
check("--psana 2 --calib-dir still refused", refused)

print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
sys.exit(1 if FAILS else 0)
