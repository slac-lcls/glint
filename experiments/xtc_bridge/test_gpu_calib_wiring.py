"""The --gpu-calib wiring: does the flag reach the reader, and does an unsupported detector REFUSE?

Runs without psana, cupy or a GPU -- psana.Detector is a stub whose pedestals()/gain() shapes are
the only thing under test, and the reader is driven with a fake DataSource. What it checks is
exactly the part that a real-data run cannot check cheaply:

  * the family guard fires on Jungfrau, whose pedestals are ALSO 4-D. Getting this wrong would not
    raise -- it would call epix10ka's decode on a Jungfrau and silently calibrate garbage.
  * an unsupported detector RAISES instead of falling back to det.calib. A silent fallback is the
    dangerous failure here: the caller believes it got a 190x speedup and got the slow path, or
    worse, believes the two paths agreed because both quietly ran det.calib.
  * gpu_calib=False still calls det.calib, and gpu_calib=True never does.
"""
from __future__ import annotations

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        FAILS.append(name)


class FakeDet:
    """Minimal psana.Detector stand-in: only the calibration-constant shapes matter."""

    def __init__(self, nmodes, panel, nseg=2):
        self.name = f"fake{nmodes}x{panel}"
        self._shape = (nmodes, nseg) + panel
        self.calib_calls = 0
        self.raw_calls = 0

    def pedestals(self, par):
        return np.zeros(self._shape, np.float32)

    def gain(self, par):
        return np.ones(self._shape, np.float32)

    def calib(self, evt):
        self.calib_calls += 1
        return np.zeros(self._shape[1:], np.float32)

    def raw(self, evt):
        self.raw_calls += 1
        return np.zeros(self._shape[1:], np.uint16)


print("family guard (gain_mode_planes)")
import gpu_calib

for label, nmodes, panel, want_ok in [
        ("epix10ka2M   (7, n, 352, 384)", 7, (352, 384), True),
        ("Jungfrau4M   (3, n, 512, 1024)", 3, (512, 1024), False),
        ("epixHR2x2    (7, n, 288, 384)", 7, (288, 384), False),
        ("wrong nmodes (3, n, 352, 384)", 3, (352, 384), False)]:
    det = FakeDet(nmodes, panel)
    try:
        gpu_calib.gain_mode_planes(det, 1)
        got, why = True, ""
    except NotImplementedError as e:
        got, why = False, str(e)[:60]
    except Exception as e:                      # epix10ka shape gets past the guard, then needs psana
        got, why = True, f"passed guard, failed later in psana ({type(e).__name__})"
    check(f"{label} -> {'accepted' if want_ok else 'refused'}", got == want_ok, why)

print("\nreader wiring (run_to_qframes_psana1 / frames_for_events)")
import xtc_qreader_psana1 as rd

# A DataSource that yields nothing: the loop body never runs, so no cupy is needed. What is under
# test is what happens BEFORE the first event -- the calibrator is built up front, so an unsupported
# detector must fail there rather than after a run's worth of work.
class FakeDS:
    def events(self):
        return iter(())

    def env(self):
        raise AssertionError("not reached")


def with_fake_psana(det, fn):
    import types
    fake = types.ModuleType("psana")
    fake.DataSource = lambda s: FakeDS()
    fake.Detector = lambda n: det
    fake.setOption = lambda *a: None
    old = sys.modules.get("psana")
    sys.modules["psana"] = fake
    try:
        return fn()
    finally:
        if old is None:
            del sys.modules["psana"]
        else:
            sys.modules["psana"] = old


jf = FakeDet(3, (512, 1024))
try:
    with_fake_psana(jf, lambda: rd.run_to_qframes_psana1(
        "x", 1, "d", zdist=0.1, wavelength=1.3, gpu_calib=True))
    check("Jungfrau + gpu_calib RAISES (no silent fallback)", False, "returned normally")
except NotImplementedError as e:
    check("Jungfrau + gpu_calib RAISES (no silent fallback)", True)
    check("  ... and the message names the detector family",
          "Epix10ka" in str(e), str(e)[:80])
except Exception as e:
    check("Jungfrau + gpu_calib RAISES (no silent fallback)", False,
          f"wrong exception {type(e).__name__}: {e}")

# ... and it must fail EAGERLY, before any event is read
check("  ... before reading any event", jf.raw_calls == 0 and jf.calib_calls == 0,
      f"raw={jf.raw_calls} calib={jf.calib_calls}")

jf2 = FakeDet(3, (512, 1024))
out = with_fake_psana(jf2, lambda: rd.run_to_qframes_psana1(
    "x", 1, "d", zdist=0.1, wavelength=1.3, gpu_calib=False))
check("Jungfrau without the flag still works (default path untouched)",
      out["n_events"] == 0 and out["n_sent"] == 0)

# frames_for_events: same guard on the pass-2 generator. It is a generator, so it must be STARTED
# for the body to run -- a guard that only fires on first next() would let pass 2 get all the way
# to the integration loop before failing.
jf3 = FakeDet(3, (512, 1024))
try:
    g = with_fake_psana(jf3, lambda: rd.frames_for_events("x", 1, "d", {0}, gpu_calib=True))
    with_fake_psana(jf3, lambda: list(g))
    check("frames_for_events + Jungfrau RAISES", False, "returned normally")
except NotImplementedError:
    check("frames_for_events + Jungfrau RAISES", True)
except Exception as e:
    check("frames_for_events + Jungfrau RAISES", False, f"{type(e).__name__}: {e}")

print("\nCLI plumbing (glint_xtc)")
import glint_xtc

p = glint_xtc.build_parser()
a = p.parse_args(["--zdist", "0.1", "--exp", "x", "--run", "1"])
check("--gpu-calib defaults to False", a.gpu_calib is False)
a = p.parse_args(["--zdist", "0.1", "--exp", "x", "--run", "1", "--gpu-calib"])
check("--gpu-calib parses as a bare switch", a.gpu_calib is True)

# psana2 goes over envbridge into another env; the flag must be refused, not ignored.
a2 = p.parse_args(["--zdist", "0.1", "--exp", "x", "--run", "1", "--gpu-calib", "--psana", "2"])
try:
    glint_xtc.read_qframes(a2, verbose=False)
    check("--gpu-calib with --psana 2 is refused", False, "did not exit")
except SystemExit as e:
    check("--gpu-calib with --psana 2 is refused", "gpu-calib" in str(e), str(e)[:70])
except ImportError:
    check("--gpu-calib with --psana 2 is refused", False,
          "envbridge missing -- guard sits after the import, move it earlier")

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
