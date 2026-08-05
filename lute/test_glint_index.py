"""Tests for the IndexGLINT task model's validators and the launcher's flag whitelist.

STATUS.md item 1. These are the pieces that decide, at config time, whether a LUTE run does what its
YAML says -- and none of them had a test. Two defects found by hand on 2026-08-05 were both in
exactly this layer, and both were invisible until the entry point was actually executed: the launcher
could not make `glint` importable, and no peak-finder threshold could reach the program at all. This
file exists so the next one is caught by `pytest` rather than by a wasted GPU job.

RUNNING WITHOUT LUTE. `IndexGLINTParameters` inherits `lute.io.models.base.ThirdPartyParameters`,
which is absent outside a LUTE install, and LUTE pins pydantic v1 while a modern dev box has v2. So
the model is loaded with the real base if it is importable and against a minimal stand-in otherwise.
The stand-in is a plain pydantic BaseModel: it exercises the SAME validator functions, which is what
is under test here, and it deliberately does NOT reproduce LUTE's flag rendering -- that is covered
instead by driving the real `glint_launch.sh` in the whitelist tests below.

    pytest lute/test_glint_index.py -q
"""
from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

pydantic = pytest.importorskip("pydantic")
V1 = pydantic.VERSION.startswith("1")


def _load_model():
    """Import glint_index with a stand-in base class when LUTE is not installed.

    Uses pydantic's v1 compatibility shim under v2 so the model's `@validator`/`@root_validator`
    decorators resolve; skips outright if neither is usable, rather than passing vacuously.
    """
    have_lute = True
    try:
        import lute.io.models.base  # noqa: F401
    except ImportError:
        have_lute = False

    saved = sys.modules.get("pydantic")
    try:
        if not have_lute:
            # glint_index.py is pydantic V1 code: two of its validators take the `field` argument,
            # which v2's deprecation shim rejects outright ("`field` and `config` are not available
            # in Pydantic V2"). LUTE pins v1, so run the module against the v1 API here too -- on a
            # v2 box that means pointing `pydantic` at the bundled `pydantic.v1` for the duration of
            # the import. Anything less faithful either fails to import or, worse, attaches no
            # validators and lets every negative test pass vacuously.
            if not V1:
                try:
                    import pydantic.v1 as p1
                except ImportError:
                    pytest.skip("pydantic v1 API unavailable; the task model needs it",
                                allow_module_level=True)
                sys.modules["pydantic"] = p1
            from pydantic import BaseModel

            class ThirdPartyParameters(BaseModel):          # minimal stand-in
                # NESTED class, not `Config = _Cfg`: a plain class attribute is read as a field.
                class Config:
                    extra = "allow"
                    arbitrary_types_allowed = True

            base = type(sys)("lute.io.models.base")
            base.ThirdPartyParameters = ThirdPartyParameters
            for name in ("lute", "lute.io", "lute.io.models"):
                sys.modules.setdefault(name, type(sys)(name))
            sys.modules["lute.io.models.base"] = base

        spec = importlib.util.spec_from_file_location("_glint_index_under_test",
                                                      os.path.join(HERE, "glint_index.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.IndexGLINTParameters
    finally:
        if saved is not None:
            sys.modules["pydantic"] = saved


P = _load_model()
XTC = dict(exp="mfxx49820", run=16, zdist=0.1027, out="o.stream")   # a minimal valid xtc config


def bad(**kw):
    """Assert the config is REJECTED, and hand back the message so the test can check which rule."""
    with pytest.raises(Exception) as e:                     # ValidationError wraps our ValueError
        P(**kw)
    return str(e.value)


# --------------------------------------------------------------------- exactly one frame source
def test_no_source_rejected():
    assert "frame source is required" in bad(out="o.stream")


@pytest.mark.parametrize("a,b", [
    ({"peaks": "p.stream"}, {"images": "i.cxi"}),
    ({"peaks": "p.stream"}, {"exp": "mfxx49820", "run": 16, "zdist": 0.1}),
    ({"images": "i.cxi"}, {"exp": "mfxx49820", "run": 16, "zdist": 0.1}),
])
def test_two_sources_rejected(a, b):
    assert "exactly ONE frame source" in bad(out="o.stream", **a, **b)


@pytest.mark.parametrize("src", [
    {"peaks": "p.stream"},
    {"images": "i.cxi"},
    {"exp": "mfxx49820", "run": 16, "zdist": 0.1027},
])
def test_one_source_accepted(src):
    assert P(out="o.stream", **src) is not None


# ------------------------------------------------------------------- xtc needs run and zdist
@pytest.mark.parametrize("missing", ["run", "zdist"])
def test_xtc_requires_run_and_zdist(missing):
    kw = dict(XTC)
    kw.pop(missing)
    assert f"`{missing}` is required with `exp`" in bad(**kw)


# ------------------------------------------------------------- peakfinder validity is per source
def test_peakfinder_defaults_per_source():
    assert P(**XTC).peakfinder == "v4"                        # raw xtc: no stored list
    assert P(images="i.cxi", out="o.stream").peakfinder == "stored"   # .cxi: reuse its own peaks


@pytest.mark.parametrize("pf", ["v4", "pf8", "pf8-panel"])
def test_peakfinder_valid_on_xtc(pf):
    assert P(peakfinder=pf, **XTC).peakfinder == pf


@pytest.mark.parametrize("pf", ["stored", "pf9"])
def test_peakfinder_rejected_on_xtc(pf):
    assert "not available on the `exp`" in bad(peakfinder=pf, **XTC)


@pytest.mark.parametrize("pf", ["pf8", "pf8-panel"])
def test_pf8_rejected_on_images(pf):
    assert "only wired on the `exp`" in bad(peakfinder=pf, images="i.cxi", out="o.stream")


# ----------------------------------------------------------------- xtc-only knobs on other sources
@pytest.mark.parametrize("field,value", [
    ("det", "MfxEndstation.0:Epix10ka2M.0"), ("psana", "1"), ("calib_dir", "/tmp/calib")])
def test_xtc_only_knobs_rejected_elsewhere(field, value):
    assert "applies only to the `exp`" in bad(peaks="p.stream", out="o.stream", **{field: value})


@pytest.mark.parametrize("field,value", [
    ("det", "MfxEndstation.0:Epix10ka2M.0"), ("psana", "1"), ("calib_dir", "/tmp/calib")])
def test_xtc_only_knobs_accepted_on_xtc(field, value):
    assert P(**XTC, **{field: value}) is not None


def test_wavelength_is_not_xtc_only():
    """Deliberately excluded from the xtc-only list -- it is meaningful on all three sources."""
    assert P(peaks="p.stream", out="o.stream", wavelength=1.29) is not None


# --------------------------------------------------------------------------- the remaining rules
def test_top_peaks_rejected_with_peaks():
    assert "applies only to `images`" in bad(peaks="p.stream", out="o.stream", top_peaks=100)


def test_top_peaks_accepted_with_images():
    assert P(images="i.cxi", out="o.stream", top_peaks=100).top_peaks == 100


def test_out_is_required():
    assert "`out` is required" in bad(peaks="p.stream", out="")


def test_integrate_with_peaks_needs_image_dir():
    assert "requires `image_dir`" in bad(peaks="p.stream", out="o.stream", integrate=True)
    assert P(peaks="p.stream", out="o.stream", integrate=True, image_dir="/data") is not None


def test_legacy_fromfile_is_renamed_not_dropped():
    """Silently dropping it would turn a best-merge config into a placeholder-intensity stream."""
    assert P(peaks="p.stream", out="o.stream", fromfile="sol.txt").tofile == "sol.txt"
    assert "not both" in bad(peaks="p.stream", out="o.stream", fromfile="a.txt", tofile="b.txt")


# ============================================================ the launcher's per-destination filter
# Driven by RUNNING glint_launch.sh with a fake `python` on PATH that records argv, because the bug
# this guards against -- a flag reaching argparse that should have been filtered, or a real flag
# being dropped -- lives in the shell, not in Python.

def _run_launcher(tmp_path, args):
    """-> (argv the launcher would have exec'd, stderr). Never runs the real indexer."""
    shim = tmp_path / "python"
    shim.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@"\n')
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    p = subprocess.run(["bash", os.path.join(HERE, "glint_launch.sh")] + args,
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=120)
    return p.stdout.split("\n"), p.stderr


def test_launcher_routes_exp_to_the_xtc_program(tmp_path):
    argv, _ = _run_launcher(tmp_path, ["--exp", "mfxx49820", "--run", "16"])
    assert any("glint_xtc.py" in a for a in argv), argv


def test_launcher_routes_peaks_to_the_cli(tmp_path):
    argv, _ = _run_launcher(tmp_path, ["--peaks", "p.stream"])
    assert any("glint.glint_cli" in a for a in argv), argv


def test_launcher_drops_cli_only_flags_and_says_so(tmp_path):
    """A dropped flag must be REPORTED: silence would let a run look as though it had honoured a
    setting the indexer never received."""
    argv, err = _run_launcher(tmp_path, ["--exp", "e", "--run", "1", "--top-peaks", "100"])
    assert "--top-peaks" not in argv
    assert "--top-peaks" in err and "dropped" in err


@pytest.mark.parametrize("flag,value", [
    ("--zdist", "0.1027"), ("--geom", "r.geom"), ("--calib-dir", "/c"),
    ("--peakfinder", "pf8"), ("--pf8-min-snr", "15"), ("--min-pix", "3"),
    ("--son-min", "15"), ("--thr-high", "10"), ("--thr-low", "5"),
])
def test_launcher_forwards_every_xtc_flag(flag, value):
    """Regression for the gap found on 2026-08-05: --pf8-min-snr existed in argparse and in the task
    model but was missing from the whitelist, so setting it was silently a no-op."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        import pathlib
        argv, err = _run_launcher(pathlib.Path(td), ["--exp", "e", "--run", "1", flag, value])
    assert flag in argv, f"{flag} was dropped: {err}"
    assert value in argv
