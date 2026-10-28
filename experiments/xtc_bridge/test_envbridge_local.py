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
import shlex
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
      Env.spack(env="x")._worker_pythonpath("/eb").endswith(':"$PYTHONPATH"'))
shutil.rmtree(_tmp, ignore_errors=True)

# --- review fixes (glint#275) -----------------------------------------------------------------------
import glob as _glob
import time as _time


def _try(fn):
    """fn() or the exception it raised -- so one broken check cannot hide the ones after it."""
    try:
        return fn()
    except Exception as e:                                  # noqa: BLE001
        return e

# a directory with a space in a pythonpath= entry, and envbridge itself under such a path
_sp = tempfile.mkdtemp(prefix="envbridge space ")
with open(os.path.join(_sp, "ebspace.py"), "w") as fh:
    fh.write("def f():\n    return 'ok'\n")
check("a pythonpath dir with a space still starts the worker",
      _try(lambda: call(Env.shell("true", pythonpath=[_sp]), "ebspace:f")) == "ok")
_eb_copy = os.path.join(_sp, "eb parent")
shutil.copytree(os.path.join(os.path.dirname(os.path.abspath(__file__)), "envbridge"),
                os.path.join(_eb_copy, "envbridge"))
_r = _try(lambda: __import__("subprocess").run(
    [__import__("sys").executable, "-c",
     "import envbridge as e; print(e.call(e.Env.shell('true'), 'envbridge._selftest:add', 2, 2))"],
    cwd=_eb_copy, capture_output=True, text=True, timeout=60))
check("envbridge itself under a path with a space starts the worker",
      (getattr(_r, "stdout", None) or "").strip() == "4")
shutil.rmtree(_sp, ignore_errors=True)

# the socket file (and its directory) is removed on close
_socks = lambda: set(_glob.glob(os.path.join(tempfile.gettempdir(), "envbridge-*.sock"))
                     + _glob.glob(os.path.join(tempfile.gettempdir(), "envbridge-sock-*")))
_before = _socks()
_try(lambda: call(SAME, "envbridge._selftest:add", 1, 1))
check("no envbridge socket or socket directory left behind after close", _socks() - _before == set())

# the socket sits in a private directory, not straight in the shared temp dir: there it took the
# umask's mode, and under umask 002 a group member could connect first and play the worker
_um = os.umask(0o002)
_w = Worker(SAME)
try:
    _try(_w.start)
    _d = os.path.dirname(_w._path or "")
    check("the socket is not bound directly in the shared temp dir", bool(_d) and _d != tempfile.gettempdir())
    check("...its directory is private (no group/other bits) under umask 002",
          bool(_d) and os.stat(_d).st_mode & 0o077 == 0)
    check("...and the worker still works", _try(lambda: _w.call("envbridge._selftest:add", 2, 2)) == 4)
finally:
    os.umask(_um)
    _w.close()

# on Linux, a connection from another user is refused (SO_PEERCRED); faked here by changing what the
# bridge believes its own uid is
import socket as _socket
import envbridge as _eb
if hasattr(_socket, "SO_PEERCRED"):
    _real_getuid = _eb.os.getuid
    _eb.os.getuid = lambda: _real_getuid() + 1
    try:
        _e = _try(Worker(SAME, timeout=10.0).start)
    finally:
        _eb.os.getuid = _real_getuid
    check("a connection from another uid is refused", isinstance(_e, RemoteError) and "another user" in str(_e))
    check("...and leaves no socket directory", _socks() - _before == set())
else:
    print("  (not run here: no SO_PEERCRED on this platform; the Linux CI runs the peer-uid check)")

# a hung activation is torn down: the subprocess is reaped and the socket file removed
_w = Worker(Env.shell("sleep 30", label="hung"), timeout=1.0)
_t0 = _time.time()
try:
    _w.start()
    _raised = False
except RemoteError:
    _raised = True
check("a hung activation raises RemoteError", _raised)
check("...and its worker process is gone", _w._proc is not None and _w._proc.poll() is not None)
check("...and its socket file is removed", _socks() - _before == set())
check("...promptly (not after the 30 s sleep)", _time.time() - _t0 < 15)

# ...including what the prelude started: killing only the shell left the prelude's `sleep` running
_tag = "41.25"                                             # a sleep no other process is running
_w = Worker(Env.shell(f"sleep {_tag}", label="hung-child"), timeout=1.0)
_try(_w.start)
_time.sleep(0.5)
_left = __import__("subprocess").run(["pgrep", "-f", f"sleep {_tag}"], capture_output=True, text=True).stdout.split()
check("a hung activation's child process is killed too", _left == [])
for _pid in _left:
    _try(lambda: os.kill(int(_pid), 9))

# a worker that connects but never sends its handshake must still time out (the accepted socket had
# no timeout, so start() waited forever). A fake `envbridge` package earlier on the worker's
# PYTHONPATH plays the stalled worker.
_fd = tempfile.mkdtemp(prefix="envbridge-stall-")
os.makedirs(os.path.join(_fd, "envbridge"))
open(os.path.join(_fd, "envbridge", "__init__.py"), "w").close()
with open(os.path.join(_fd, "envbridge", "_worker.py"), "w") as fh:
    fh.write("import socket, sys, time\ns = socket.socket(socket.AF_UNIX)\ns.connect(sys.argv[1])\n"
             "time.sleep(30)\n")
_res = {}


def _stall():
    _res["e"] = _try(Worker(Env.shell("true", label="stall", pythonpath=[_fd]), timeout=1.0).start)


# `python -m` puts the CURRENT directory first on sys.path, and this directory holds the real envbridge,
# so run the stalled worker from the fake's directory
_cwd = os.getcwd()
os.chdir(_fd)


import threading as _thr
_t = _thr.Thread(target=_stall, daemon=True)
_t.start(); _t.join(15)
check("a worker that connects but never hand-shakes times out",
      not _t.is_alive() and isinstance(_res.get("e"), (RemoteError, OSError)))
os.chdir(_cwd)
shutil.rmtree(_fd, ignore_errors=True)

# arrays: 0-d shape kept; object and structured dtypes refused at encode
_z = _try(lambda: call(SAME, "envbridge._selftest:echo", np.array(7.0)))
check("0-d array keeps shape ()", isinstance(_z, np.ndarray) and _z.shape == () and _z == 7.0)
for _bad, _what in ((np.zeros(2, dtype=[("a", "<i4"), ("b", "<f8")]), "structured"),
                    (np.array([1, "x"], dtype=object), "object")):
    _e = _try(lambda: call(SAME, "envbridge._selftest:echo", _bad))
    check(f"{_what} dtype refused at encode",
          isinstance(_e, TypeError) and "cannot send an array of dtype" in str(_e))

# @remote used as a decorator: the worker resolves module:qualname to the WRAPPER, which must run the
# body there instead of calling call() again (that recursed, one new worker per level). Run in a child
# process group with a time limit, so the broken version is killed rather than left spawning workers.
import signal as _signal
import subprocess as _subprocess
import sys as _sys
_dd = tempfile.mkdtemp(prefix="envbridge-deco-")
with open(os.path.join(_dd, "ebdeco.py"), "w") as fh:
    fh.write("import os\nfrom envbridge import Env, remote\n\n\n"
             "@remote(Env.shell('true'))\ndef pid_plus(x):\n    return [os.getpid(), x + 1]\n")
_p = _subprocess.Popen(
    [_sys.executable, "-c", "import os, ebdeco; pid, y = ebdeco.pid_plus(1); "
                            "print(y, pid != os.getpid())"],
    cwd=_dd, env={**os.environ, "PYTHONPATH": os.path.dirname(os.path.abspath(__file__)) + ":" + _dd},
    stdout=_subprocess.PIPE, stderr=_subprocess.DEVNULL, text=True, start_new_session=True)
try:
    _out, _ = _p.communicate(timeout=30)
except _subprocess.TimeoutExpired:
    os.killpg(_p.pid, _signal.SIGKILL)
    _out = "timeout"
check("@remote as a decorator runs the body in the worker (no recursion)", _out.strip() == "2 True")
shutil.rmtree(_dd, ignore_errors=True)

# @remote stacked on @remote: each worker unwrapped one layer, got the inner wrapper and started another
# worker, without end. Refused at decoration time (never called here, so the broken version cannot spawn).
def _plain(x):
    return x


_e = _try(lambda: remote(SAME)(remote(SAME)(_plain)))
check("@remote applied twice is refused at decoration", isinstance(_e, TypeError) and "twice" in str(_e))

# an ordinary functools.wraps decorator ON TOP of @remote: the worker unwrapped one layer, landed on the
# @remote wrapper and started another worker, without end (~130 workers in 8 s). Checked by resolving
# the name the way the worker does, without calling it, so the broken version spawns nothing.
from envbridge import _worker as _wk
_do = tempfile.mkdtemp(prefix="envbridge-outer-")
with open(os.path.join(_do, "ebouter.py"), "w") as fh:
    fh.write("import functools\nfrom envbridge import Env, remote\n\n\n"
             "def logged(f):\n    @functools.wraps(f)\n    def w(*a, **k):\n        return f(*a, **k)\n"
             "    return w\n\n\n"
             "@logged\n@remote(Env.shell('true'))\ndef plus_one(x):\n    return x + 1\n")
_sys.path.insert(0, _do)
_fn = _try(lambda: _wk._resolve("ebouter:plus_one"))
_sys.path.remove(_do)
check("a decorator on top of @remote: the worker resolves to the body, not to a wrapper",
      callable(_fn) and not hasattr(_fn, "_envbridge_spec") and _fn.__name__ == "plus_one")
_is_body = callable(_fn) and not hasattr(_fn, "_envbridge_spec")   # never call a wrapper: it would spawn
check("...and the body runs locally there", _is_body and _try(lambda: _fn(1)) == 2)
shutil.rmtree(_do, ignore_errors=True)

# conda / spack activation with spaces in the paths (they went into the shell unquoted). Fake
# conda.sh / setup-env.sh define `conda` / `spack` functions that succeed only if they get the env
# path as ONE argument; the worker then runs in this env.
_fk = tempfile.mkdtemp(prefix="envbridge base ")
os.makedirs(os.path.join(_fk, "etc", "profile.d"))
_envp = os.path.join(_fk, "my env")
with open(os.path.join(_fk, "etc", "profile.d", "conda.sh"), "w") as fh:
    fh.write(f'conda() {{ [ "$1" = activate ] && [ "$2" = {shlex.quote(_envp)} ] && [ $# -eq 2 ]; }}\n')
with open(os.path.join(_fk, "setup env.sh"), "w") as fh:
    fh.write(f'spack() {{ [ "$1 $2" = "env activate" ] && [ "$3" = {shlex.quote(_envp)} ] && [ $# -eq 3 ]; }}\n')
check("Env.conda with spaces in base and env path activates",
      _try(lambda: call(Env.conda(_envp, base=_fk), "envbridge._selftest:add", 1, 1)) == 2)
check("Env.spack with spaces in setup and env path activates",
      _try(lambda: call(Env.spack(env=_envp, setup=os.path.join(_fk, "setup env.sh")),
                        "envbridge._selftest:add", 2, 2)) == 4)
check("Env.spack leaves a multi-word spec unquoted (spack syntax)",
      Env.spack(spec="py-numpy %gcc", setup="/s").prelude.endswith("spack load py-numpy %gcc"))
shutil.rmtree(_fk, ignore_errors=True)

# a user dict holding the array marker key is refused at encode (the decoder would misread it)
with Worker(SAME) as w:
    _e = _try(lambda: w.call("envbridge._selftest:echo", {"__nd__": 1}))
    check("a dict with the reserved '__nd__' key is refused at encode",
          isinstance(_e, TypeError) and "reserved" in str(_e))
    check("...and the same worker still serves the next call",
          _try(lambda: w.call("envbridge._selftest:add", 1, 2)) == 3)

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

# after a call times out, its reply is still pending on the socket: the worker must not be reused, or
# the NEXT call would return this one's late reply (3d6661e, Copilot Autofix on #275)
with Worker(SAME, timeout=1.0, call_timeout=1.0) as w:
    _first = _try(lambda: w.call("envbridge._selftest:sleep", 2.5))
    _time.sleep(2.0)                                       # let the late reply arrive
    _second = _try(lambda: w.call("envbridge._selftest:add", 1, 2))
check("a timed-out call raises", isinstance(_first, OSError))
check("...and the worker is not reused (no stale reply for the next call)",
      isinstance(_second, RuntimeError) and "not started" in str(_second))

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

# a call interrupted by something other than a socket error (here a signal handler's exception, as
# Ctrl-C would be) left its reply in flight, and the next call on the same Worker received it
class _Interrupt(Exception):
    pass


def _raise_interrupt(*_):
    raise _Interrupt()


_old_h = _signal.signal(_signal.SIGALRM, _raise_interrupt)
_w = Worker(SAME, call_timeout=None)
try:
    _try(_w.start)
    _signal.setitimer(_signal.ITIMER_REAL, 0.5)
    _first = _try(lambda: _w.call("envbridge._selftest:sleep", 1.5))
    _signal.setitimer(_signal.ITIMER_REAL, 0)
    _time.sleep(1.5)                                     # the stale reply has arrived by now
    _second = _try(lambda: _w.call("envbridge._selftest:add", 1, 2))
finally:
    _signal.setitimer(_signal.ITIMER_REAL, 0)
    _signal.signal(_signal.SIGALRM, _old_h)
    _w.close(kill=True)
check("an interrupted call raises the interruption", isinstance(_first, _Interrupt))
check("...and the next call on that Worker gets no stale reply",
      isinstance(_second, RuntimeError) and "not started" in str(_second))

print("ALL PASS" if ok else "FAILURES ABOVE")
raise SystemExit(0 if ok else 1)
