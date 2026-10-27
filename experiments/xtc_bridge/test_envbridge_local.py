"""Tests for the vendored envbridge (experiments/xtc_bridge/envbridge/, from slac-lcls/drp-benchmarks
envbridge @ 5cd09f8, plus Worker.call_timeout). Run from this directory:

    python test_envbridge_local.py

Self-test runnable in ONE env (no conda/spack needed): Env.shell('true') spawns the worker in the
same interpreter's environment, so this exercises the codec, the socket transport, the worker loop and
error propagation end to end. The cross-ENV behaviour (conda1/conda2/spack activation) is verified
separately on a machine that has those stacks -- see README.

"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

from envbridge import Env, Worker, call, remote, RemoteError

SAME = Env.shell("true")            # no-op activation: worker runs in this very env

ok = True


def check(name, cond):
    global ok
    ok = ok and cond
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


# --- one-shot call, JSON round-trip -----------------------------------------------------------------
check("call add(2,3)==5", call(SAME, "envbridge._selftest:add", 2, 3) == 5)

# --- numpy array round-trip must be BIT-EXACT (the pickle-free codec's whole job) -------------------
a = np.random.RandomState(0).standard_normal((17, 3))
back = call(SAME, "envbridge._selftest:echo", a)
check("ndarray bit-exact round-trip", isinstance(back, np.ndarray) and np.array_equal(back, a))
check("dtype preserved", back.dtype == a.dtype)

# --- a few dtypes, including the ones that matter across the numpy boundary --------------------------
for dt in ("float32", "float64", "int32", "int64", "uint16"):
    v = (np.arange(12).reshape(3, 4)).astype(dt)
    r = call(SAME, "envbridge._selftest:echo", v)
    check(f"dtype {dt} exact", r.dtype == np.dtype(dt) and np.array_equal(r, v))

# --- nested structure: dict mixing arrays and JSON scalars -------------------------------------------
res = call(SAME, "envbridge._selftest:array_stats", a)
check("nested dict: scalar", abs(res["sum"] - float(a.sum())) < 1e-9)
check("nested dict: array", np.array_equal(res["doubled"], a * 2))
check("nested dict: json list", res["shape"] == [17, 3])

# --- persistent worker: many calls over one activation ----------------------------------------------
with Worker(SAME) as w:
    check("persistent worker reuse", all(w.call("envbridge._selftest:add", i, 1) == i + 1
                                         for i in range(50)))

# --- remote exception surfaces as RemoteError with the far-side traceback ---------------------------
try:
    call(SAME, "envbridge._selftest:boom")
    check("remote error raises", False)
except RemoteError as e:
    check("remote error raises RemoteError", "intentional failure" in str(e))
    check("remote traceback carried", "ValueError" in str(e))

# --- sending an un-sendable object fails LOUDLY at encode, not silently ------------------------------
try:
    call(SAME, "envbridge._selftest:echo", object())
    check("unsendable rejected", False)
except TypeError as e:
    check("unsendable rejected at encode", "cannot send" in str(e))

# --- the decorator form -----------------------------------------------------------------------------
_wrapped = remote(SAME)(__import__("envbridge._selftest", fromlist=["add"]).add)
check("decorator runs in target env", _wrapped(4, 5) == 9)

# --- PYTHONPATH inheritance: a prelude's PYTHONPATH must not shadow the target env ------------------
# REGRESSION TEST for the bug that made `glint_xtc.py --psana 2` fail out of the box. LCLS's
# conda2 psconda.sh exports PYTHONPATH=.../lcls2_100726/install/lib/python3.9/site-packages, a psana
# built for Python 3.9; the worker command appended `:$PYTHONPATH` unconditionally, so that python3.9
# psana landed in front of the python3.11 psana inside xpp_drp_gpu_311 and `from psana import dgram`
# died with "cannot import name 'dgram' from partially initialized module 'psana'".
#
# Reproduced here WITHOUT any conda stack: a prelude that exports a PYTHONPATH holding a shadowing
# module, and a package directory that holds the real one. The worker must import the real module.
# ⚠ This fails on the pre-fix code -- that is the point of it. Both halves matter: the conda case
# must NOT inherit (or psana breaks), and spack MUST inherit (that is how `spack load` delivers
# packages), so asserting only one direction would let the other regress silently.
_tmp = tempfile.mkdtemp(prefix="envbridge-pptest-")
_shadow = os.path.join(_tmp, "shadow"); os.makedirs(_shadow)
# ⚠ THE SHADOWED MODULE MUST LIVE IN SITE-PACKAGES, not on `pythonpath=`. Inherited entries are
# appended LAST, so they can never outrank an explicit `pythonpath=` dir or envbridge's own -- the
# only thing they can shadow is the env's site-packages, and that is precisely what psana is. An
# earlier version of this test put the real module on `pythonpath=` and PASSED on the broken code.
# numpy stands in for psana: same situation (installed in the env, imported by the worker), and the
# codec needs the real one to move an array.
with open(os.path.join(_shadow, "numpy.py"), "w") as fh:
    fh.write("raise ImportError('shadowed numpy: the envbridge PYTHONPATH regression test')\n")
with open(os.path.join(_shadow, "ebprobe.py"), "w") as fh:
    fh.write("def which():\n    return 'shadow'\n")

_prelude = f"export PYTHONPATH={_shadow}"      # the prelude plays the part of psconda.sh
_noinherit = Env(prelude=_prelude, label="pp-noinherit", inherit_pythonpath=False)
_inherit   = Env(prelude=_prelude, label="pp-inherit",   inherit_pythonpath=True)

_arr = np.arange(6, dtype=np.float64)
try:
    _got = call(_noinherit, "envbridge._selftest:echo", _arr)
    _round = isinstance(_got, np.ndarray) and np.array_equal(_got, _arr)
except (RemoteError, Exception):
    _round = False
check("a prelude PYTHONPATH cannot shadow the env's site-packages (numpy still imports)", _round)

# the opposite direction, so "fixing" this cannot silently break spack, where `spack load` is
# exactly how python packages reach PYTHONPATH
check("prelude PYTHONPATH IS still honoured when inherit_pythonpath=True",
      call(_inherit, "ebprobe:which") == "shadow")

check("Env.conda does not inherit (the psana shadowing case)",
      Env.conda("x", stack="conda2").inherit_pythonpath is False)
check("Env.spack still inherits (spack load delivers packages on PYTHONPATH)",
      Env.spack(env="x").inherit_pythonpath is True)
check("Env.shell still inherits", Env.shell("true").inherit_pythonpath is True)
check("conda worker PYTHONPATH carries no :$PYTHONPATH",
      "$PYTHONPATH" not in Env.conda("x", stack="conda2")._worker_pythonpath("/eb"))
check("spack worker PYTHONPATH keeps :$PYTHONPATH",
      Env.spack(env="x")._worker_pythonpath("/eb").endswith(":$PYTHONPATH"))
shutil.rmtree(_tmp, ignore_errors=True)

# --- call_timeout: a long call must not be cut off by the activation timeout ------------------------
# REGRESSION TEST: the activation timeout (120 s) also bounded every call, and glint_xtc's psana2 reader
# reads the whole run in ONE call, so any `--psana 2` run longer than 2 min died on a socket timeout.
# Scaled down: a 1 s activation timeout, a 2.5 s call.
import socket as _socket
try:
    with Worker(SAME, timeout=1.0) as w:
        w.call("envbridge._selftest:sleep", 2.5)
    _cut = False
except _socket.timeout:
    _cut = True
check("default call_timeout is the activation timeout (unchanged behaviour)", _cut)
with Worker(SAME, timeout=1.0, call_timeout=None) as w:
    check("call_timeout=None lets a call outlive the activation timeout",
          w.call("envbridge._selftest:sleep", 2.5) == 2.5)
with Worker(SAME, timeout=1.0, call_timeout=5.0) as w:
    check("an explicit call_timeout replaces it", w.call("envbridge._selftest:sleep", 2.5) == 2.5)

# with no call limit, a worker that dies mid-call must still raise, not hang
import threading as _threading
_res = {}


def _crash():
    try:
        with Worker(SAME, call_timeout=None) as w:
            w.call("envbridge._selftest:die")
        _res["r"] = "returned"
    except Exception as e:                                  # noqa: BLE001 -- any error is the pass
        _res["r"] = type(e).__name__


_t = _threading.Thread(target=_crash, daemon=True)
_t.start(); _t.join(30)
check("a worker dying mid-call raises even with call_timeout=None",
      not _t.is_alive() and _res.get("r") not in (None, "returned"))
print(f"        (raised {_res.get('r')})")

print("ALL PASS" if ok else "FAILURES ABOVE")
raise SystemExit(0 if ok else 1)
