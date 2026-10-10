"""glint_xtc_mpi never writes CUDA_VISIBLE_DEVICES; it refuses a launch that leaves a rank's GPU undecided.

WHY THIS EXISTS. GLINT used to pin each MPI rank by rewriting CUDA_VISIBLE_DEVICES (``pin_gpu``). In
review of glint#256, @monarin asked that the launcher own placement instead, for two reasons:
- the wrapper imports mpi4py before it could rewrite the variable, and importing mpi4py can initialize
  CUDA (lcls2#175 measured that under psana's MPI). A rewrite after that moves only the bridge worker the
  rank spawns, not the rank itself, so the torch indexer and the worker can land on different GPUs;
- lcls2#175 keeps the launcher's mask and selects within it, and does not adopt GLINT's narrowing.

So ``check_gpu_binding`` only reads the mask. It accepts one device or an empty mask (the launcher
decided), and several devices or an unset mask when this rank is alone on its node. With several ranks
on a node it refuses, with the launch fix, unless ``--share-gpu`` says they share one GPU on purpose.
The checks below compare the WHOLE environment before and after every call, and scan the module's
source for any write to the variable, so restoring the old pinning fails here.

Torch- and MPI-free on purpose, so it really runs in the CPU CI job and in ``run_ci_locally.py``.

  PYTHONPATH=. python experiments/test_gpu_binding.py     # exit 0 = all pass
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("GLINT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODULE = ROOT / "experiments" / "xtc_bridge" / "glint_xtc_mpi.py"
sys.path.insert(0, str(MODULE.parent))

from glint_xtc_mpi import GpuBindingError, check_gpu_binding, retired_gpus_per_node  # noqa: E402

FAILS = 0


def report(name, ok, detail):
    global FAILS
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILS += 1


def case(name, mask, local_size, share, accept, says=None):
    """One launch: the inherited mask, the ranks on the node, --share-gpu; must it run, and say what."""
    if mask is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = mask
    before = dict(os.environ)
    try:
        got, refused = check_gpu_binding(os.environ.get("CUDA_VISIBLE_DEVICES"), local_size, share), None
    except GpuBindingError as e:
        got, refused = None, str(e)
    unchanged = dict(os.environ) == before
    ok = unchanged and (refused is None) == accept
    if accept and says is not None:
        ok = ok and says in got
    if not accept:
        ok = ok and "--gpus-per-task=1" in refused and "--share-gpu" in refused
    detail = (f"{'ran: ' + repr(got) if refused is None else 'refused: ' + refused[:60] + '...'}; "
              f"environment {'unchanged' if unchanged else 'CHANGED'}")
    report(name, ok, detail)


# the launcher gave this rank one device: use it, alone on the node or not
case("single ordinal, 4 ranks on node",      "3",            4, False, True, "GPU 3")
case("single UUID, 4 ranks on node",         "GPU-aaaa",     4, False, True, "GPU GPU-aaaa")
case("single MIG id, 2 ranks on node",       "MIG-1234",     2, False, True, "MIG-1234")
case("single ordinal with spaces ' 2 '",     " 2 ",          2, False, True, "GPU 2")

# an empty mask: the launcher gave this rank no GPU; it stays that way
case("empty mask, 4 ranks on node",          "",             4, False, True, "no GPU")
case("empty mask, alone",                    "",             1, False, True, "no GPU")

# several devices or unset, alone on the node: the rank and its worker both use the first device
case("mask 2,3, alone",                      "2,3",          1, False, True, "first visible GPU")
case("unset, alone",                         None,           1, False, True, "first visible GPU")

# several ranks on the node and no per-rank GPU: refuse with the fix (they would all use one GPU)
case("mask 2,3, 2 ranks on node",            "2,3",          2, False, False)
case("two UUIDs, 2 ranks on node",           "GPU-a,GPU-b",  2, False, False)
case("unset, 4 ranks on node",               None,           4, False, False)

# ...unless --share-gpu says that is intended
case("mask 2,3, 2 ranks, --share-gpu",       "2,3",          2, True,  True, "shared by the 2 ranks")
case("unset, 4 ranks, --share-gpu",          None,           4, True,  True, "shared by the 4 ranks")

# the retired option: refused with the fix whenever it asks for pinning; unset or 0 is accepted
for name, value, env, refuse in [
        ("--gpus-per-node unset, env unset", None, {}, False),
        ("--gpus-per-node 0",                0,    {}, False),
        ("GLINT_GPUS_PER_NODE=0",            None, {"GLINT_GPUS_PER_NODE": "0"}, False),
        ("--gpus-per-node 2",                2,    {}, True),
        ("GLINT_GPUS_PER_NODE=4",            None, {"GLINT_GPUS_PER_NODE": "4"}, True),
        ("--gpus-per-node 0, env 4",         0,    {"GLINT_GPUS_PER_NODE": "4"}, True)]:
    msg = retired_gpus_per_node(value, env)
    ok = (msg is not None) == refuse and (msg is None or "--gpus-per-task=1" in msg)
    report(name, ok, "refused" if msg else "accepted")

# no code path in the module writes the variable: assignment, del, or an environ/putenv call naming it
writes = []
for node in ast.walk(ast.parse(MODULE.read_text())):
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
        if "CUDA_VISIBLE_DEVICES" in ast.unparse(node.slice):
            writes.append(f"line {node.lineno}: {ast.unparse(node)}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in ("putenv", "unsetenv", "update", "setdefault", "pop", "__setitem__") \
                and "CUDA_VISIBLE_DEVICES" in ast.unparse(node):
            writes.append(f"line {node.lineno}: {ast.unparse(node)}")
report("glint_xtc_mpi.py source writes CUDA_VISIBLE_DEVICES nowhere", not writes,
       "; ".join(writes) or "no writes found")

print(f"\n{'OK' if FAILS == 0 else 'FAIL'}: {FAILS} failing case(s)")
sys.exit(1 if FAILS else 0)
