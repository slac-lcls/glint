"""The device ladder must never select MPS -- it cannot run this package (glint#161).

WHY THIS EXISTS. `glint.replica_gpu` and `glint.glint_index` chose their device as
cuda -> mps -> cpu. Apple's MPS backend does not implement float64, and both modules compute in
float64, so on any Apple-silicon machine the mps rung raised

    TypeError: Cannot convert a MPS Tensor to float64 dtype ...

at IMPORT time, before a single frame could be indexed -- and `glint --device cpu` did NOT rescue
it, because that flag only clears `CUDA_VISIBLE_DEVICES`, which says nothing about MPS. So the one
documented escape hatch was inert on exactly the machines that needed it, and the mps rung was
never an acceleration path at all: only a way for the working cpu fallback to be skipped.

Found while verifying the commands in REPRODUCING.md actually run -- a referee reading the paper on
a Mac would have hit it on their first GLINT command.

THE SOURCE CHECK RUNS WITHOUT TORCH, DELIBERATELY. The obvious way to write this test is to import
the two modules and read their `DEV`, but the environments that run it -- the CPU CI job, and
`run_ci_locally.py`, which blocks torch at the import system on purpose -- have no torch, so a
test built that way skips in every automated context and would report success with the defect
restored (Copilot review of glint#161). So the regression that CI must catch is checked by reading
the SOURCE, which needs nothing; the runtime assertions below are a bonus wherever torch exists.

  PYTHONPATH=. python experiments/test_device_selection.py     # exit 0 = all pass
"""
from __future__ import annotations

import os
import pathlib
import re
import sys

ROOT = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


MODULES = ("glint/replica_gpu.py", "glint/glint_index.py")

# --- the half that must run everywhere, torch or no torch ---------------------------------------
for rel in MODULES:
    src = (ROOT / rel).read_text(encoding="utf-8")
    # Comments are stripped first: the fix's own comment explains that torch.backends.mps is no
    # longer consulted, which a naive substring search reads as the very thing it promises is gone.
    code = "\n".join(l.split("#", 1)[0] for l in src.split("\n"))
    # The two forms that can actually select the backend -- NOT a bare "mps" substring, which
    # matches inside "jumps" and "ramps" in the prose of docstrings (comment-stripping does not
    # reach those). Being precise here is the difference between a rule and a nuisance.
    MPS_CODE = re.compile(r"backends\s*\.\s*mps|[\"']mps[\"']")
    offenders = [l.strip() for l in code.split("\n") if MPS_CODE.search(l)]
    check(f"{rel} does not select mps in code", not offenders, offenders[:2])

    dev_lines = [l.strip() for l in code.split("\n") if re.match(r"\s*DEV\s*=", l)]
    check(f"{rel} assigns DEV exactly once", len(dev_lines) == 1, dev_lines)
    if len(dev_lines) == 1:
        check(f"{rel} ladder is cuda -> cpu",
              "cuda" in dev_lines[0] and "cpu" in dev_lines[0] and "mps" not in dev_lines[0],
              dev_lines[0])

# --- the runtime half, wherever torch exists ----------------------------------------------------
try:
    import torch
except ImportError:
    print("\n  (runtime checks skipped: no torch here -- the source checks above are the "
          "regression CI relies on)")
else:
    import glint.glint_index as gi
    import glint.replica_gpu as rg

    for mod in (gi, rg):
        name = mod.__name__
        check(f"{name}.DEV is one of cuda/cpu", mod.DEV in ("cuda", "cpu"), mod.DEV)
        try:                                  # whatever was chosen must accept the module's dtype
            torch.zeros(2, dtype=torch.float64, device=mod.DEV)
            ok, why = True, ""
        except Exception as exc:              # noqa: BLE001 - the message is the point
            ok, why = False, repr(exc)
        check(f"{name}.DEV accepts float64 (what the module actually uses)", ok, why)
    # the original failure was at IMPORT, so getting here is itself part of the contract
    check("both modules import on this host", True)

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
