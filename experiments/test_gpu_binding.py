"""glint_xtc_mpi never writes CUDA_VISIBLE_DEVICES, and stops a launch whose GPU placement is undecided.

WHY THIS EXISTS. GLINT used to pin each MPI rank by rewriting CUDA_VISIBLE_DEVICES (``pin_gpu``). In
review of glint#256, @monarin asked that the launcher own placement instead, and that the wrapper fail
with an actionable error when the assignment is missing or ambiguous. Two reasons:
- the wrapper imports mpi4py before it could rewrite the variable, and importing mpi4py can initialize
  CUDA (lcls2#175 measured that under psana's MPI). A rewrite after that moves only the bridge worker the
  rank spawns, not the rank itself, so the torch indexer and the worker can land on different GPUs;
- matching mask strings do not establish matching GPUs. So every rank now reports the physical UUID of
  the device it will use, and ranks are compared by that.

WHAT IS CHECKED (torch-, GPU- and MPI-free; a stand-in mpi4py and stub GPU identities):
  1. decide_placement's rules, on hand-built per-rank records;
  2. the retired --gpus-per-node / GLINT_GPUS_PER_NODE;
  3. main() itself, as ONE rank of a simulated job whose other ranks' reports are injected: it must stop
     (return 2, reader never called) when any rank's report forbids the run, and reach the reader
     otherwise. The whole environment is compared before main(), at the reader, and after;
  4. a source scan of glint_xtc_mpi.py and glint_xtc.py (the per-rank body that spawns the reader) for
     direct writes to the variable and the common indirect forms: an alias constant, environ |=,
     update/setdefault/pop/putenv, a CUDA_VISIBLE_DEVICES= keyword, or an env dict literal with that key.
     It is a guard, not a proof; check 3 is the behavioural one;
  5. the wrapper's --help renders (argparse %-formats help strings; glint#257).

  PYTHONPATH=. python experiments/test_gpu_binding.py     # exit 0 = all pass
"""
from __future__ import annotations

import argparse
import ast
import os
import pathlib
import sys
import types

ROOT = pathlib.Path(os.environ.get("GLINT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BRIDGE = ROOT / "experiments" / "xtc_bridge"
sys.path.insert(0, str(BRIDGE))

import glint_xtc_mpi as W  # noqa: E402  (module-level imports are os/sys/pathlib only)

FAILS = 0
CVD = "CUDA_VISIBLE_DEVICES"


def report(name, ok, detail=""):
    global FAILS
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILS += 1


def rec(rank, host="n1", cvd="0", uuid="GPU-a", note="A100", retired=None):
    return {"rank": rank, "host": host, "cvd": cvd, "uuid": uuid, "note": note, "retired": retired}


NOUUID = "no uuid: torch 1.11 does not report device UUIDs"

# ---- 1. the rules ---------------------------------------------------------------------------------
for name, records, share, run in [
    ("one GPU per rank, distinct UUIDs",          [rec(0, uuid="GPU-a"), rec(1, uuid="GPU-b")], False, True),
    ("same mask string '0', distinct UUIDs (per-task cgroups)",
                                                   [rec(0, cvd="0", uuid="GPU-a"), rec(1, cvd="0", uuid="GPU-b")], False, True),
    ("disjoint multi-GPU masks, distinct UUIDs",  [rec(0, cvd="0,1", uuid="GPU-a"), rec(1, cvd="2,3", uuid="GPU-c")], False, True),
    ("two ranks resolve to one UUID",             [rec(0, uuid="GPU-a"), rec(1, uuid="GPU-a")], False, False),
    ("two ranks, one UUID, --share-gpu",          [rec(0, uuid="GPU-a"), rec(1, uuid="GPU-a")], True, True),
    ("same UUID on DIFFERENT hosts is fine",      [rec(0, host="n1"), rec(1, host="n2")], False, True),
    ("a rank sees no GPU",                         [rec(0), rec(1, cvd="", uuid=None, note="no GPU visible")], False, False),
    ("a rank sees no GPU, even with --share-gpu",  [rec(0), rec(1, cvd="", uuid=None, note="no GPU visible")], True, False),
    ("lone rank, mask 0,1",                        [rec(0, cvd="0,1")], False, True),
    ("no-uuid fallback: single masks",             [rec(0, cvd="0", uuid=None, note=NOUUID), rec(1, cvd="1", uuid=None, note=NOUUID)], False, True),
    ("no-uuid fallback: unset mask, 2 on host",    [rec(0, cvd=None, uuid=None, note=NOUUID), rec(1, cvd=None, uuid=None, note=NOUUID)], False, False),
    ("no-uuid fallback: mask 0,1, 2 on host",      [rec(0, cvd="0,1", uuid=None, note=NOUUID), rec(1, cvd="0,1", uuid=None, note=NOUUID)], False, False),
    ("no-uuid fallback: unset, --share-gpu",       [rec(0, cvd=None, uuid=None, note=NOUUID), rec(1, cvd=None, uuid=None, note=NOUUID)], True, True),
    ("a retired option on one rank stops all",     [rec(0), rec(1, uuid="GPU-b", retired="GLINT_GPUS_PER_NODE=2 was removed")], False, False),
]:
    probs = W.decide_placement(records, share)
    ok = (not probs) == run and all("srun --gpus-per-task=1" in p or "removed" in p for p in probs)
    report(f"rule: {name}", ok, "runs" if not probs else "stops: " + probs[0][:70] + "...")

# ---- 2. the retired option ------------------------------------------------------------------------
for name, value, env, refuse, says in [
        ("--gpus-per-node unset, env unset", None, {}, False, None),
        ("--gpus-per-node 0",                0,    {}, False, None),
        ("GLINT_GPUS_PER_NODE=0",            None, {"GLINT_GPUS_PER_NODE": "0"}, False, None),
        ("--gpus-per-node 2",                2,    {}, True, "Drop it"),
        ("GLINT_GPUS_PER_NODE=4",            None, {"GLINT_GPUS_PER_NODE": "4"}, True, "Unset it"),
        ("--gpus-per-node 0, env 4",         0,    {"GLINT_GPUS_PER_NODE": "4"}, True, "Unset it")]:
    msg = W.retired_gpus_per_node(value, env)
    ok = (msg is not None) == refuse and (msg is None or (says in msg and "--gpus-per-task=1" in msg))
    report(f"retired: {name}", ok, "refused" if msg else "accepted")

# ---- 3. main() as one rank of a simulated job -------------------------------------------------------


class FakeComm:
    """This process is rank `rank`; allgather returns the injected reports of the other ranks."""

    def __init__(self, rank, others):
        self.rank, self.others, self.gathers = rank, others, 0

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return 1 + len(self.others)

    def allgather(self, x):
        self.gathers += 1
        it = iter(self.others)
        return [x if r == self.rank else next(it) for r in range(self.Get_size())]

    def gather(self, x, root=0):
        raise AssertionError("main() went past placement into the reduction")

    def Barrier(self):
        pass


class ReachedReader(Exception):
    pass


import glint_xtc  # noqa: E402  (numpy + xtc_core only)

seen = {}


def stub_read(args, rank=0, nranks=1, verbose=True):
    seen["env"] = dict(os.environ)
    raise ReachedReader


def run_main(name, others, my_uuid="GPU-a", argv_extra=(), env_extra=None, expect_reader=False):
    comm = FakeComm(0, others)
    mpi = types.ModuleType("mpi4py.MPI"); mpi.COMM_WORLD = comm
    pkg = types.ModuleType("mpi4py"); pkg.MPI = mpi
    saved = {k: sys.modules.get(k) for k in ("mpi4py", "mpi4py.MPI")}
    sys.modules["mpi4py"], sys.modules["mpi4py.MPI"] = pkg, mpi
    real_identity, real_read = W.gpu_identity, glint_xtc.read_qframes
    W.gpu_identity = (lambda: (my_uuid, "stub")) if my_uuid else (lambda: (None, "no GPU visible"))
    glint_xtc.read_qframes = stub_read
    old_env = dict(os.environ)
    os.environ.update(env_extra or {})
    os.environ[CVD] = "3"   # unusual on purpose: a stray write of "0" must show as a change
    seen.clear()
    before, reached, rc = dict(os.environ), False, None
    try:
        rc = W.main(["--exp", "sim", "--run", "1", "--zdist", "0.1", "--psana", "1", "-o", "/dev/null"]
                    + list(argv_extra))
    except ReachedReader:
        reached = True
    finally:
        after = dict(os.environ)
        os.environ.clear(); os.environ.update(old_env)
        W.gpu_identity, glint_xtc.read_qframes = real_identity, real_read
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    env_ok = after == before and (not reached or seen.get("env") == before)
    ok = env_ok and comm.gathers == 1 and (reached if expect_reader else (rc == 2 and not reached))
    report(f"main(): {name}", ok, f"{'reached the reader' if reached else f'returned {rc}'}; "
           f"allgather x{comm.gathers}; environment {'unchanged' if env_ok else 'CHANGED'}")


host = __import__("socket").gethostname()
run_main("alone on its host", [], expect_reader=True)
run_main("another rank, other GPU", [rec(1, host=host, uuid="GPU-b")], expect_reader=True)
run_main("another rank on this host, same GPU", [rec(1, host=host, uuid="GPU-a")])
run_main("same GPU, --share-gpu", [rec(1, host=host, uuid="GPU-a")], argv_extra=["--share-gpu"], expect_reader=True)
run_main("this rank sees no GPU", [], my_uuid=None)
run_main("another rank sees no GPU", [rec(1, host=host, cvd="", uuid=None, note="no GPU visible")])
run_main("another rank has the retired variable", [rec(1, host=host, uuid="GPU-b", retired="GLINT_GPUS_PER_NODE=2 was removed")])
run_main("--gpus-per-node 2 here", [], argv_extra=["--gpus-per-node", "2"])
run_main("GLINT_GPUS_PER_NODE=2 here", [], env_extra={"GLINT_GPUS_PER_NODE": "2"})

# ---- 4. source scan --------------------------------------------------------------------------------


def writes_in(path):
    tree = ast.parse(path.read_text())
    alias = {CVD}
    for n in ast.walk(tree):                       # names bound to the string, e.g. _K = "CUDA_..."
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and n.value.value == CVD:
            alias |= {t.id for t in n.targets if isinstance(t, ast.Name)}

    def names_it(node):
        for m in ast.walk(node):
            if isinstance(m, ast.Constant) and m.value == CVD:
                return True
            if isinstance(m, ast.Name) and m.id in alias:
                return True
            if isinstance(m, ast.keyword) and m.arg == CVD:
                return True
        return False

    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript) and isinstance(n.ctx, (ast.Store, ast.Del)) and names_it(n.slice):
            hits.append(n)
        elif isinstance(n, ast.AugAssign) and names_it(n):
            hits.append(n)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in (
                "putenv", "unsetenv", "update", "setdefault", "pop", "__setitem__", "__delitem__") and names_it(n):
            hits.append(n)
        elif isinstance(n, ast.keyword) and n.arg == CVD:
            hits.append(n)
        elif isinstance(n, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == CVD for k in n.keys):
            hits.append(n)
    return [f"{path.name}:{getattr(h, 'lineno', '?')}: {ast.unparse(h)[:70]}" for h in hits]


for f in (BRIDGE / "glint_xtc_mpi.py", BRIDGE / "glint_xtc.py"):
    w = writes_in(f)
    report(f"scan: {f.name} writes CUDA_VISIBLE_DEVICES nowhere", not w, "; ".join(w) or "none of the forms found")

# ---- 5. --help ---------------------------------------------------------------------------------------
try:
    text = W.build_parser().format_help()
    ok = all(o in text for o in ("--share-gpu", "--gpus-per-node", "--keep-parts")) and "removed:" in text
    report("glint_xtc_mpi --help renders", ok, f"{len(text)} chars")
except (TypeError, ValueError) as e:
    report("glint_xtc_mpi --help renders", False, f"{type(e).__name__}: {e}")

print(f"\n{'OK' if FAILS == 0 else 'FAIL'}: {FAILS} failing case(s)")
sys.exit(1 if FAILS else 0)
