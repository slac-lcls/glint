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
a Mac would have hit it on their first command.

What is pinned:
  * neither module offers an `mps` device, under any torch, on any host
  * the ladder is exactly cuda -> cpu, and resolves to a device torch will accept float64 on
  * the modules import at all (the crash was at import, so this is not redundant)

  PYTHONPATH=. python experiments/test_device_selection.py     # exit 0 = all pass
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


try:
    import torch
except ImportError:                      # pragma: no cover - the CPU CI job has no torch
    print("SKIP: torch not available (the device ladder needs it to be meaningful)")
    sys.exit(0)

import glint.glint_index as gi
import glint.replica_gpu as rg

for mod in (gi, rg):
    name = mod.__name__
    check(f"{name}.DEV is never 'mps'", mod.DEV != "mps", mod.DEV)
    check(f"{name}.DEV is one of cuda/cpu", mod.DEV in ("cuda", "cpu"), mod.DEV)
    # the whole point: whatever was chosen must accept the dtype the module computes in
    try:
        torch.zeros(2, dtype=torch.float64, device=mod.DEV)
        ok, why = True, ""
    except Exception as exc:             # noqa: BLE001 - the message is the point
        ok, why = False, repr(exc)
    check(f"{name}.DEV accepts float64 (what the module actually uses)", ok, why)

# the crash was at IMPORT, so reaching here at all is part of the contract
check("both modules import on this host", True)

# ...and the CODE must not have grown the rung back. Comments are stripped first: the fix's own
# explanatory comment says "torch.backends.mps is no longer consulted", which a naive substring
# search reads as the very thing it is promising is absent.
import inspect                                                        # noqa: E402
for mod in (gi, rg):
    code = "\n".join(l.split("#", 1)[0] for l in inspect.getsource(mod).split("\n"))
    check(f"{mod.__name__} does not consult torch.backends.mps in code",
          "backends.mps" not in code and '"mps"' not in code,
          [l.strip() for l in code.split("\n") if "mps" in l][:2])

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
