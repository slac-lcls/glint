"""pin_gpu must NARROW the launcher's CUDA_VISIBLE_DEVICES, never replace it (glint_xtc_mpi).

WHY THIS EXISTS. A rank that inherits a multi-device mask (``CUDA_VISIBLE_DEVICES=2,3`` from
``srun --gpus-per-task=2``, or two UUIDs) and is run with ``--gpus-per-node 2`` used to have that mask
OVERWRITTEN with the bare ordinal ``local_rank % 2`` -- "0" or "1", devices the launcher never granted.
And an explicitly EMPTY mask (a CPU-only rank) was read as "unset", so the same option re-enabled a GPU.
psana2's MPI GPU path is being fixed for the same pattern (lcls2#155: honour the launcher's visibility,
choose within the allowed set, never guess physical ordinals), and the GLINT rank and the psana worker
share a device by inheriting this one variable, so GLINT's side has to obey the same rule.

Torch-free on purpose: ``pin_gpu`` touches only ``os.environ``, so this runs in the CPU CI job and in
``run_ci_locally.py``, where a torch-gated test would skip and stay green with the defect restored.

  PYTHONPATH=. python experiments/test_pin_gpu.py     # exit 0 = all pass
"""
from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("GLINT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, str(ROOT / "experiments" / "xtc_bridge"))

from glint_xtc_mpi import pin_gpu  # noqa: E402  (module-level imports are os/sys/pathlib only)

FAILS = 0


def check(name, got, want, env_want):
    """One case: the return value and the variable left in the environment, both compared."""
    global FAILS
    env_got = os.environ.get("CUDA_VISIBLE_DEVICES")
    ok = (got == want) and (env_got == env_want)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: returned {got!r}, env {env_got!r}"
          + ("" if ok else f"  (wanted {want!r}, env {env_want!r})"))
    if not ok:
        FAILS += 1


def case(name, mask, local_rank, gpus_per_node, want, env_want):
    if mask is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = mask
    got = pin_gpu(local_rank, gpus_per_node)
    check(name, got, want, env_want)


# the launcher pinned one device: trust it, whatever the option says
case("single ordinal, no option",        "3",            1, 0, "3", "3")
case("single ordinal, option set",       "3",            1, 4, "3", "3")
case("single UUID, option set",          "GPU-aaaa",     1, 4, "GPU-aaaa", "GPU-aaaa")

# an empty mask is a CPU-only rank: keep it empty (used to become "1")
case("empty mask, option set",           "",             1, 2, "", "")
case("empty mask, no option",            "",             1, 0, "", "")

# several devices allowed: choose WITHIN them (used to become the bare ordinal "1")
case("mask 2,3 rank 1 gpn 2",            "2,3",          1, 2, "3", "3")
case("mask 2,3 rank 0 gpn 2",            "2,3",          0, 2, "2", "2")
case("mask 2,3 rank 3 gpn 4 (wraps)",    "2,3",          3, 4, "3", "3")
case("mask 2,3 rank None gpn 2",         "2,3",       None, 2, "2", "2")
case("two UUIDs rank 1 gpn 2",           "GPU-a,GPU-b",  1, 2, "GPU-b", "GPU-b")
case("mask with spaces ' 2, 3' rank 1",  " 2, 3",        1, 2, "3", "3")

# several visible but no pin requested: trust the launcher, leave all visible
case("mask 2,3 no option",               "2,3",          1, 0, "2,3", "2,3")

# nothing set by the launcher: the old behaviour, unchanged
case("unset, no option",                 None,           1, 0, None, None)
case("unset, gpn 4 rank 6",              None,           6, 4, "2", "2")
case("unset, gpn 4 rank None",           None,        None, 4, "0", "0")

print(f"\n{'OK' if FAILS == 0 else 'FAIL'}: {FAILS} failing case(s)")
sys.exit(1 if FAILS else 0)
