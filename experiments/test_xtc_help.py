"""glint_xtc.py --help must exit 0. argparse %-formats every help= string (that is how %(default)s
works), so a bare "%" in one of them is a format directive, not a character.

THE DEFECT. --geom's help said "median 3.16% in |q|". "% i" is the %i directive with the space flag,
so format_help() raised `TypeError: %i format: a number is required, not dict` on every --help, at
exit 1. Nothing noticed because lute/glint_launch.sh forwarded a whitelist of flags and dropped
--help; LUTE's IndexGLINT (slac-lcls/lute#147) now runs the file directly, so --help reaches
argparse. Seen on S3DF, ana-4.0.58-py3-minipytorch, 9 Oct 2026.

WHAT RUNS (numpy only: --help needs argparse, numpy and xtc_core, nothing heavy):
  1. the instrument: a parser with a bare "% i" in a help string DOES raise on this interpreter,
     so the two checks below passing means the strings are clean, not that argparse is lenient;
  2. the real entry point as LUTE runs it, `python <root>/experiments/xtc_bridge/glint_xtc.py --help`
     in a subprocess: exit 0, usage on stdout, and the escaped figures render as "3.16%" and
     "97.7%", with no "%%" left in the text;
  3. build_parser().format_help() in-process, which expands every help string and group description
     at once, so any future bare "%" in glint_xtc.py fails here rather than at the beamline.
glint_xtc_mpi.py builds its parser inside main() after `from mpi4py import MPI`, so its two extra
options are not reachable without MPI; the one with a "%" is already escaped ("%% N").

Plain script: prints ok/FAIL lines and exits non-zero on any failure (CI runs it as a script).
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE = os.path.join(ROOT, "experiments", "xtc_bridge")
SCRIPT = os.path.join(BRIDGE, "glint_xtc.py")
fails = []


def check(name, cond, msg=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name:18s} {msg}")
    if not cond:
        fails.append(name)


# 1. The instrument can fail: argparse on THIS interpreter %-formats help= strings.
bad = argparse.ArgumentParser(prog="x", add_help=False)
bad.add_argument("--geom", help="median 3.16% in |q|")
try:
    bad.format_help()
    check("instrument", False, "a bare '% i' in help= did not raise, so this test cannot see the defect")
except (TypeError, ValueError) as e:
    check("instrument", True, f"bare '% i' in help= raises {type(e).__name__}: {e}")

# 2. The entry point exactly as LUTE invokes it: the file by absolute path, from the repo root.
r = subprocess.run([sys.executable, SCRIPT, "--help"], cwd=ROOT, capture_output=True, text=True)
err_tail = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
check("--help exit", r.returncode == 0, f"exit {r.returncode}" + (f"; stderr: {err_tail}" if r.returncode else ""))
check("--help usage", r.stdout.startswith("usage:"), repr(r.stdout[:50]))
check("--help renders %", "3.16%" in r.stdout and "97.7%" in r.stdout and "%%" not in r.stdout,
      "3.16% and 97.7% appear once-escaped, no '%%' survives")

# 3. Every help string and every group description, expanded in one go.
sys.path.insert(0, BRIDGE)
import glint_xtc                                          # numpy + xtc_core only at import

try:
    text = glint_xtc.build_parser().format_help()
except (TypeError, ValueError) as e:
    text = ""
    check("format_help", False, f"{type(e).__name__}: {e}")
else:
    check("format_help", "%%" not in text, f"{len(text)} chars, no '%%' survives")
    # The group descriptions are NOT %-formatted by argparse (only help= is, and descriptions only
    # when they contain %(prog)), so "100% of frames" must stay single-%: an over-escape would print
    # "100%%". That is the other way this file can regress.
    check("group desc", "100% of frames" in text, "'100% of frames' prints as written")
    for opt in ("--geom", "--gpu-calib", "--peakfinder", "--pf8-min-snr", "--integrate", "--cell"):
        check(f"lists {opt}", opt in text)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
sys.exit(1 if fails else 0)
