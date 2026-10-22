"""glint_xtc_mpi never writes CUDA_VISIBLE_DEVICES, and stops a launch whose GPU placement is undecided.

WHY THIS EXISTS. GLINT used to pin each MPI rank by rewriting CUDA_VISIBLE_DEVICES (``pin_gpu``). In
review of glint#256, @monarin asked that the launcher own placement instead, and that the wrapper fail
with an actionable error when the assignment is missing or ambiguous. Two reasons:
- the wrapper imports mpi4py before it could rewrite the variable, and a CUDA-aware MPI can initialize
  CUDA when mpi4py loads (lcls2#175 measured CUDA already initialized after ``import psana``). A rewrite
  after that moves only the reader the rank spawns, not the rank, so the two can land on different GPUs;
- matching mask strings do not establish matching GPUs. So every rank identifies its device physically
  (CuPy's PCI bus id; torch's UUID as a fallback), and ranks are compared by that.

WHAT IS CHECKED (no torch, GPU or MPI needed: stand-in modules throughout):
  1. decide_placement's rules, on hand-built per-rank records;
  2. gpu_identity() itself, against stand-in cupy and torch modules, and that what it returns is what
     decide_placement acts on;
  3. the retired --gpus-per-node / GLINT_GPUS_PER_NODE;
  4. main() run as rank 0 AND as rank 1 of a simulated job, with the other ranks' reports injected:
     it must stop (return 2, reader never reached, the fix printed) when any report forbids the run,
     and otherwise reach the reader through the REAL glint_xtc.read_qframes, on both routes (psana1
     in-process; psana2 at the envbridge boundary). The environment is compared before main(), at the
     reader (in Python and as a child process sees it, which catches os.putenv) and after;
  5. a source scan of the wrapper and reader modules for direct and common indirect writes. It is a
     guard, not a proof; check 4 is the behavioural one;
  6. the wrapper's --help renders (argparse %-formats help strings; glint#257).

  PYTHONPATH=. python experiments/test_gpu_binding.py     # exit 0 = all pass
"""
from __future__ import annotations

import ast
import contextlib
import io
import os
import pathlib
import subprocess
import sys
import types

ROOT = pathlib.Path(os.environ.get("GLINT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BRIDGE = ROOT / "experiments" / "xtc_bridge"
sys.path.insert(0, str(BRIDGE))

import glint_xtc_mpi as W  # noqa: E402  (module-level imports are os/sys/pathlib only)
import glint_xtc  # noqa: E402  (numpy + xtc_core only)

FAILS = 0
CVD = "CUDA_VISIBLE_DEVICES"
FIX = "srun --gpus-per-task=1"


def report(name, ok, detail=""):
    global FAILS
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILS += 1


def rec(rank, host="n1", cvd="0", status="ok", key="PCI 0000:41:00.0", detail="stub", retired=None):
    return {"rank": rank, "host": host, "cvd": cvd, "status": status, "key": key, "detail": detail,
            "retired": retired}


A, B = "PCI 0000:41:00.0", "PCI 0000:81:00.0"
UNK = dict(status="unknown", key=None, detail="no CuPy, and torch 1.11.0 reports no device UUID")

# ---- 1. the rules ---------------------------------------------------------------------------------
for name, records, share, run in [
    ("one GPU per rank",                          [rec(0, key=A), rec(1, key=B)], False, True),
    ("same mask '0', different GPUs (per-task cgroups)", [rec(0, cvd="0", key=A), rec(1, cvd="0", key=B)], False, True),
    ("disjoint multi-GPU masks",                  [rec(0, cvd="0,1", key=A), rec(1, cvd="2,3", key=B)], False, True),
    ("two ranks, one GPU",                        [rec(0, key=A), rec(1, key=A)], False, False),
    ("two ranks, one GPU, --share-gpu",           [rec(0, key=A), rec(1, key=A)], True, True),
    ("same key on different hosts",               [rec(0, host="n1"), rec(1, host="n2")], False, True),
    ("a rank sees no GPU",                        [rec(0), rec(1, cvd="", status="no-gpu", key=None)], False, False),
    ("a rank sees no GPU, --share-gpu",           [rec(0), rec(1, cvd="", status="no-gpu", key=None)], True, False),
    ("a rank's GPU check raised",                 [rec(0), rec(1, status="error", key=None, detail="RuntimeError: x")], False, False),
    ("lone rank, mask 0,1",                       [rec(0, cvd="0,1")], False, True),
    ("unknown: masks '0' and '1'",                [rec(0, cvd="0", **UNK), rec(1, cvd="1", **UNK)], False, True),
    ("unknown: disjoint masks '0,1' and '2,3'",   [rec(0, cvd="0,1", **UNK), rec(1, cvd="2,3", **UNK)], False, True),
    ("unknown: both mask '0'",                    [rec(0, cvd="0", **UNK), rec(1, cvd="0", **UNK)], False, False),
    ("unknown: both unset",                       [rec(0, cvd=None, **UNK), rec(1, cvd=None, **UNK)], False, False),
    ("unknown: both unset, --share-gpu",          [rec(0, cvd=None, **UNK), rec(1, cvd=None, **UNK)], True, True),
    ("unknown: empty mask",                       [rec(0, cvd="", **UNK)], False, False),
    ("a retired option on one rank stops all",    [rec(0), rec(1, key=B, retired="GLINT_GPUS_PER_NODE=2 was removed")], False, False),
    # incomparable reports (Copilot, 18 Oct): separate buckets are not proof of separate GPUs
    ("unknown: unset and '0' on one host",        [rec(0, cvd=None, **UNK), rec(1, cvd="0", **UNK)], False, False),
    ("unknown: unset and '0', --share-gpu",       [rec(0, cvd=None, **UNK), rec(1, cvd="0", **UNK)], True, True),
    ("identified and unidentified on one host",   [rec(0, key=A), rec(1, cvd="0", **UNK)], False, False),
    ("identified and unidentified, --share-gpu",  [rec(0, key=A), rec(1, cvd="0", **UNK)], True, True),
    ("identified and unidentified, two hosts",    [rec(0, key=A), rec(1, host="n2", cvd="0", **UNK)], False, True),
    ("PCI id and UUID on one host",               [rec(0, key=A), rec(1, key="GPU-aaaa-1111-0")], False, False),
    ("PCI id and UUID, --share-gpu",              [rec(0, key=A), rec(1, key="GPU-aaaa-1111-0")], True, True),
    ("PCI id and UUID on different hosts",        [rec(0, key=A), rec(1, host="n2", key="GPU-aaaa-1111-0")], False, True),
]:
    probs = W.decide_placement(records, share)
    ok = (not probs) == run and all(FIX in p or "removed" in p for p in probs)
    if any(r["status"] == "no-gpu" for r in records):   # the specific reason, not just any refusal
        ok = ok and any("sees no GPU" in p for p in probs)
    report(f"rule: {name}", ok, "runs" if not probs else "stops: " + probs[0][:70] + "...")

# CUDA_VISIBLE_DEVICES=-1 hides every GPU: an unidentified rank with it must stop, with or without --share-gpu
for share in (False, True):
    probs = W.decide_placement([rec(0, cvd="-1", **UNK), rec(1, cvd="0", key=A, host="n2")], share)
    ok = bool(probs) and any("sees no GPU" in p and "'-1'" in p for p in probs)
    report(f"rule: mask '-1' hides every GPU (unknown report){', --share-gpu' if share else ''}", ok,
           "stops: " + probs[0][:70] + "..." if probs else "runs")
report("rule: _first_device treats -1 and '' and None alike", all(W._first_device(c) is None for c in ("-1", "-1,0", "", None, " "))
       and W._first_device("2,3") == "2")

summary = W.placement_summary([rec(r, key=f"PCI {r}") for r in range(20)])
report("summary: one line per rank, no cap", len(summary) == 20 and all(f"rank {r}:" in summary[r] for r in range(20)))

# ---- 2. gpu_identity against stand-in cupy / torch ---------------------------------------------


def fake_cupy(count=None, raises=None, buses=("0000:41:00.0", "0000:81:00.0")):
    m = types.ModuleType("cupy")

    def get_count():
        if raises:
            raise RuntimeError(raises)
        return count
    m.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=get_count),
                                   Device=lambda i: types.SimpleNamespace(pci_bus_id=buses[i]))
    return m


def fake_torch(available=True, cuda="12.1", uuid="aaaa-1111", version="2.5.1", raises=None):
    m = types.ModuleType("torch")
    m.__version__ = version
    m.version = types.SimpleNamespace(cuda=cuda)

    def props(i):
        if raises:
            raise RuntimeError(raises)
        d = {"name": f"A100 #{i}"}
        if uuid is not None:
            d["uuid"] = f"{uuid}-{i}"
        return types.SimpleNamespace(**d)
    m.cuda = types.SimpleNamespace(is_available=lambda: available, get_device_properties=props)
    return m


def identity_with(cupy, torch):
    saved = {k: sys.modules.get(k) for k in ("cupy", "torch")}
    sys.modules["cupy"], sys.modules["torch"] = cupy, torch   # None makes the import raise ImportError
    try:
        return W.gpu_identity()
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


for name, cupy, torch, want_status, want_key in [
    ("CuPy sees 2: device 0's bus id",       fake_cupy(count=2), None, "ok", "PCI 0000:41:00.0"),
    ("CuPy sees 0",                           fake_cupy(count=0), None, "no-gpu", None),
    ("CuPy raises (no driver/device)",        fake_cupy(raises="cudaErrorNoDevice"), None, "no-gpu", None),
    ("no CuPy, torch 2.5 with UUID",          None, fake_torch(), "ok", "GPU-aaaa-1111-0"),
    ("no CuPy, torch 1.11 without UUID",      None, fake_torch(uuid=None, version="1.11.0"), "unknown", None),
    ("no CuPy, torch built without CUDA",     None, fake_torch(available=False, cuda=None), "unknown", None),
    ("no CuPy, CUDA torch sees no device",    None, fake_torch(available=False), "no-gpu", None),
    ("neither CuPy nor torch",                None, None, "unknown", None),
]:
    st, key, detail = identity_with(cupy, torch)
    r = W.placement_record(0, "n1", {CVD: "0"}, None, (st, key, detail))
    lone_ok = (not W.decide_placement([r])) == (st != "no-gpu")
    report(f"identity: {name}", st == want_status and key == want_key and lone_ok, f"{st} {key} ({detail})")

# ---- 3. the retired option ------------------------------------------------------------------------
for name, value, env, refuse, says in [
        ("--gpus-per-node unset, env unset", None, {}, False, None),
        ("--gpus-per-node 0",                0,    {}, False, None),
        ("GLINT_GPUS_PER_NODE=0",            None, {"GLINT_GPUS_PER_NODE": "0"}, False, None),
        ("--gpus-per-node 2",                2,    {}, True, "Drop it"),
        ("GLINT_GPUS_PER_NODE=4",            None, {"GLINT_GPUS_PER_NODE": "4"}, True, "Unset it"),
        ("--gpus-per-node 0, env 4",         0,    {"GLINT_GPUS_PER_NODE": "4"}, True, "Unset it")]:
    msg = W.retired_gpus_per_node(value, env)
    ok = (msg is not None) == refuse and (msg is None or (says in msg and FIX in msg))
    report(f"retired: {name}", ok, "refused" if msg else "accepted")

# ---- 4. main() as one rank of a simulated job, through the real read_qframes ------------------------


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
        raise AssertionError("main() went past the reader into the reduction")

    def Barrier(self):
        pass


class ReachedReader(Exception):
    pass


seen = {}


def child_view():
    """CUDA_VISIBLE_DEVICES as a child process sees it (the C environment, so os.putenv shows)."""
    r = subprocess.run([sys.executable, "-c", f"import os; print(os.environ.get({CVD!r}))"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def at_reader(where, env_obj=None):
    seen.update(where=where, env=dict(os.environ), child=child_view(),
                env_obj=repr(vars(env_obj)) if env_obj is not None else "")
    raise ReachedReader


fake_psana1 = types.ModuleType("xtc_qreader_psana1")
fake_psana1.run_to_qframes_psana1 = lambda *a, **k: at_reader("psana1")
fake_envbridge = types.ModuleType("envbridge")
fake_envbridge.Env = types.SimpleNamespace(conda=lambda *a, **k: types.SimpleNamespace(args=a, **k))
fake_envbridge.call = lambda env, fn, *a, **k: at_reader("psana2", env)

MASK = "3"   # unusual on purpose: a stray write of "0" must show as a change
host = __import__("socket").gethostname()


def run_main(name, me, others, argv_extra=(), env_extra=None, expect_reader=False, psana="1", my_identity=None):
    comm = FakeComm(me, others)
    mpi = types.ModuleType("mpi4py.MPI"); mpi.COMM_WORLD = comm
    pkg = types.ModuleType("mpi4py"); pkg.MPI = mpi
    stubs = {"mpi4py": pkg, "mpi4py.MPI": mpi, "xtc_qreader_psana1": fake_psana1, "envbridge": fake_envbridge}
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    real_identity = W.gpu_identity
    W.gpu_identity = my_identity or (lambda: ("ok", A, "stub"))
    old_env = dict(os.environ)
    os.environ.update(env_extra or {})
    os.environ[CVD] = MASK
    seen.clear()
    before, reached, rc = dict(os.environ), False, None
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = W.main(["--exp", "sim", "--run", "1", "--zdist", "0.1", "--psana", psana, "-o", "/dev/null"]
                        + list(argv_extra))
    except ReachedReader:
        reached = True
    except Exception as e:   # anything else escaping main() is a failure of this case, not of the test
        rc = f"{type(e).__name__}: {e}"
    finally:
        after = dict(os.environ)
        os.environ.clear(); os.environ.update(old_env)
        W.gpu_identity = real_identity
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    env_ok = after == before and (not reached or (seen["env"] == before and seen["child"] == MASK
                                                   and CVD not in seen["env_obj"]))
    size = 1 + len(others)
    if expect_reader:
        ok = reached and seen.get("where") == f"psana{psana}"
        logged = me != 0 or all(f"rank {r}:" in out.getvalue() for r in range(size))
    else:
        ok = rc == 2 and not reached
        logged = me != 0 or FIX in err.getvalue() or "removed" in err.getvalue()
    ok = ok and env_ok and logged and comm.gathers == 1
    what = f"reached the {seen.get('where')} reader" if reached else f"returned {rc}"
    report(f"main() rank {me}: {name}", ok, f"{what}; allgather x{comm.gathers}; environment "
           f"{'unchanged' if env_ok else 'CHANGED'}{'' if logged else '; NOT LOGGED'}")


def raising_identity():
    raise RuntimeError("CUDA error: unknown error")


run_main("alone on its host", 0, [], expect_reader=True)
for me in (0, 1):
    peer = 1 - me
    run_main("another rank, other GPU", me, [rec(peer, host=host, key=B)], expect_reader=True)
    run_main("another rank, other GPU, psana2 route", me, [rec(peer, host=host, key=B)], psana="2", expect_reader=True)
    run_main("another rank on this host, same GPU", me, [rec(peer, host=host, key=A)])
    run_main("same GPU, --share-gpu", me, [rec(peer, host=host, key=A)], argv_extra=["--share-gpu"], expect_reader=True)
    run_main("this rank sees no GPU", me, [rec(peer, host=host, key=B)], my_identity=lambda: ("no-gpu", None, "stub"))
    run_main("this rank's GPU check raises", me, [rec(peer, host=host, key=B)], my_identity=raising_identity)
    run_main("another rank sees no GPU", me, [rec(peer, host=host, cvd="", status="no-gpu", key=None)])
    run_main("another rank has the retired variable", me, [rec(peer, host=host, key=B, retired="GLINT_GPUS_PER_NODE=2 was removed")])
    run_main("--gpus-per-node 2 here", me, [rec(peer, host=host, key=B)], argv_extra=["--gpus-per-node", "2"])
    run_main("GLINT_GPUS_PER_NODE=2 here", me, [rec(peer, host=host, key=B)], env_extra={"GLINT_GPUS_PER_NODE": "2"})

# ---- 5. source scan --------------------------------------------------------------------------------
SHELL = ("export CUDA_VISIBLE_DEVICES", "env CUDA_VISIBLE_DEVICES=")   # a mask set in a shell prelude


def writes_in(path):
    tree = ast.parse(path.read_text())
    alias = {CVD}
    for n in ast.walk(tree):                       # names bound to the string, e.g. _K = "CUDA_..."
        val = n.value if isinstance(n, (ast.Assign, ast.AnnAssign)) else None
        if isinstance(val, ast.Constant) and val.value in (CVD, CVD.encode()):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            alias |= {t.id for t in targets if isinstance(t, ast.Name)}

    def names_it(node):
        for m in ast.walk(node):
            if isinstance(m, ast.Constant) and m.value in (CVD, CVD.encode()):
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
        elif isinstance(n, ast.Dict) and any(isinstance(k, ast.Constant) and k.value in (CVD, CVD.encode()) for k in n.keys):
            hits.append(n)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and any(t in n.value for t in SHELL):
            hits.append(n)
        elif isinstance(n, ast.Constant) and isinstance(n.value, bytes) and CVD.encode() in n.value:
            hits.append(n)
    return [f"{path.name}:{getattr(h, 'lineno', '?')}: {ast.unparse(h)[:70]}" for h in hits]


for f in ("glint_xtc_mpi.py", "glint_xtc.py", "xtc_qreader.py", "xtc_qreader_psana1.py", "xtc_core.py", "gpu_calib.py"):
    w = writes_in(BRIDGE / f)
    report(f"scan: {f} writes CUDA_VISIBLE_DEVICES nowhere", not w, "; ".join(w) or "none of the forms found")

# ---- 6. --help ---------------------------------------------------------------------------------------
try:
    text = W.build_parser().format_help()
    ok = all(o in text for o in ("--share-gpu", "--gpus-per-node", "--keep-parts")) and "removed:" in text
    report("glint_xtc_mpi --help renders", ok, f"{len(text)} chars")
except (TypeError, ValueError) as e:
    report("glint_xtc_mpi --help renders", False, f"{type(e).__name__}: {e}")

print(f"\n{'OK' if FAILS == 0 else 'FAIL'}: {FAILS} failing case(s)")
sys.exit(1 if FAILS else 0)
