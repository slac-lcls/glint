"""envbridge -- run a function in ANOTHER conda/spack environment on the same node, and pass numpy
arrays across, when the two environments cannot share one interpreter (incompatible numpy ABIs, or
packages that will not co-import like psana + torch).

    from envbridge import Env, Worker, remote

    # one call in the other env
    from envbridge import call
    n = call(Env.conda("xpp_drp_gpu_311", stack="conda2"), "numpy:add", a, b)

    # persistent worker (amortise the seconds-long conda activation over many calls)
    with Worker(Env.conda("ana-4.0.58-py3-minipytorch", stack="conda1")) as w:
        for x in xs:
            q = w.call("mypkg.reader:frame_to_q", x)

    # decorator: this function's body runs in the OTHER env
    @remote(Env.spack(env="psana2"))
    def read(run, evt): ...

WHY THIS EXISTS. General cross-interpreter RPC (RPyC, execnet, Pyro5) bridges different Python
*versions* but bakes in no conda/spack activation and moves no arrays efficiently; the LCLS-native
LCLStreamer is a network/DAQ scale-out system, not a same-node call. The one thing none of them do is
activate a spack (or a specific conda-stack) env for you. That is the whole novelty here; everything
else is stdlib.

The transport is stdlib-only (subprocess + AF_UNIX socket) so it imports in BOTH hostile envs, and the
wire is pickle-free (see _codec) so it is safe across incompatible numpy ABIs. Data crosses by copy
through the socket -- fine to ~GB; for larger same-node payloads the documented upgrade is
multiprocessing.shared_memory (CPU) or CUDA IPC (GPU), not built in here.
"""
from __future__ import annotations

import os
import shlex
import socket
import subprocess
import tempfile
import uuid
from dataclasses import dataclass

from . import _codec

__all__ = ["Env", "Worker", "call", "remote", "RemoteError"]

# LCLS psana conda stacks. The activation prelude nobody else bakes in.
_LCLS_STACKS = {
    "conda1": "/sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh",
    "conda2": "/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh",
}


class RemoteError(RuntimeError):
    """Raised in the caller when the function raised in the target env; carries the remote traceback."""


@dataclass
class Env:
    """How to activate the target environment: a shell prelude run before the worker starts.

    `pythonpath` are directories prepended to the worker's PYTHONPATH (before envbridge's own), so the
    target `module:qualname` -- which lives in YOUR package, not in envbridge -- is importable there.

    `inherit_pythonpath` decides whether the PYTHONPATH the PRELUDE leaves behind is also passed to
    the worker. It must be False for a conda env and True for spack, and the difference is not a
    style choice:

      * conda ships packages in the env's own site-packages, so a conda env never needs PYTHONPATH --
        but LCLS's `psconda.sh` exports one anyway, and it points at a release built for a DIFFERENT
        interpreter. MEASURED on sdfiana027, 2026-10-19:
          conda1                     -> PYTHONPATH unset; psana1 imports from the env either way.
          conda2/xpp_drp_gpu_311     -> PYTHONPATH=.../lcls2_100726/install/lib/python3.9/site-packages
                                        while the interpreter is 3.11.14. That python3.9 psana shadows
                                        the env's own and `from psana import dgram` fails with
                                        "cannot import name 'dgram' from partially initialized
                                        module 'psana'". With PYTHONPATH unset, psana imports from
                                        the env's python3.11 site-packages and the read works.
        So inheriting is useless in one stack and actively broken in the other.
      * spack commonly DOES deliver python packages by putting them on PYTHONPATH (`spack load`), so
        dropping it there would break the env instead of fixing it. Spack and `shell` keep inheriting.
    """
    prelude: str
    label: str = ""
    pythonpath: tuple = ()
    inherit_pythonpath: bool = True

    def _worker_pythonpath(self, eb_parent: str) -> str:
        """The PYTHONPATH assignment for the worker command, as a shell fragment.

        Factored out so it can be asserted without a conda stack (see test_local.py) -- the bug this
        guards against is invisible in the composed command unless you look for the `:$PYTHONPATH`.
        """
        # the literal dirs are shell-quoted (a space in a checkout path would otherwise split the
        # command); the inherited `$PYTHONPATH` stays expandable, inside double quotes.
        pp = shlex.quote(":".join([*self.pythonpath, eb_parent]))
        return f'{pp}:"$PYTHONPATH"' if self.inherit_pythonpath else pp

    @classmethod
    def conda(cls, env: str, stack: str | None = None, base: str | None = None,
              pythonpath=()) -> "Env":
        """A conda env. `stack='conda1'|'conda2'` picks an LCLS psana stack; else `base` is a conda
        install whose etc/profile.d/conda.sh is sourced (defaults to $CONDA_PREFIX_1 / `conda info`)."""
        if stack:
            src = _LCLS_STACKS.get(stack)
            if not src:
                raise ValueError(f"unknown LCLS stack {stack!r}; use one of {sorted(_LCLS_STACKS)}")
            pre = f"source {src}"
        else:
            base = base or os.environ.get("CONDA_PREFIX_1") or os.environ.get("CONDA_PREFIX", "")
            pre = f"source {base}/etc/profile.d/conda.sh"
        # inherit_pythonpath=False: see the Env docstring. A conda env serves its packages from its own
        # site-packages, and the LCLS stacks' psconda.sh leaves behind a PYTHONPATH for the wrong
        # interpreter, which is what broke `glint_xtc.py --psana 2` over this bridge.
        return cls(f"{pre} && conda activate {env}", label=f"conda:{env}",
                   pythonpath=tuple(pythonpath), inherit_pythonpath=False)

    @classmethod
    def spack(cls, spec: str | None = None, env: str | None = None, setup: str | None = None,
              pythonpath=()) -> "Env":
        """A spack environment (`env=`) or a loaded spec (`spec=`). `setup` is spack's setup-env.sh;
        it defaults to $SPACK_ROOT/share/spack/setup-env.sh so `spack` is on PATH in the subprocess."""
        setup = setup or (os.environ.get("SPACK_ROOT", "") + "/share/spack/setup-env.sh")
        pp = tuple(pythonpath)
        if env:
            return cls(f"source {setup} && spack env activate {env}", label=f"spack-env:{env}", pythonpath=pp)
        if spec:
            return cls(f"source {setup} && spack load {spec}", label=f"spack:{spec}", pythonpath=pp)
        raise ValueError("Env.spack needs either env= or spec=")

    @classmethod
    def shell(cls, prelude: str, label: str = "shell", pythonpath=()) -> "Env":
        """Any activation you write yourself (source a setup.sh, module load, etc.)."""
        return cls(prelude, label=label, pythonpath=tuple(pythonpath))


class Worker:
    """A persistent subprocess in the target env. Reuse it to amortise the (seconds-long) activation."""

    def __init__(self, env: Env, timeout: float = 120.0, call_timeout="same"):
        """`timeout` bounds the env ACTIVATION (the worker connecting back). `call_timeout` bounds each
        `.call()`: "same" (the default) reuses `timeout`, None waits for as long as the call runs.

        They are separate because a call can legitimately run for hours (GLINT's psana2 reader reads a
        whole run in one call), while a hung activation should still fail fast. A worker that dies
        mid-call closes its socket, so the caller gets an error rather than waiting forever.
        """
        self.env = env
        self.timeout = timeout
        self.call_timeout = timeout if call_timeout == "same" else call_timeout
        self._sock = self._conn = self._proc = None
        self._path = None

    def start(self) -> "Worker":
        path = self._path = os.path.join(tempfile.gettempdir(), f"envbridge-{uuid.uuid4().hex}.sock")
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(path)
        self._sock.listen(1)
        self._sock.settimeout(self.timeout)
        # activate, then set PYTHONPATH *after* (conda activate resets it) so the worker can import
        # both envbridge and the caller's target module, even in an env where neither is installed;
        # the env's own numpy stays first because these dirs carry no numpy.
        # ⚠ Whether the prelude's own PYTHONPATH is appended is Env.inherit_pythonpath, NOT a constant:
        # appending it unconditionally put a python3.9 psana in front of a python3.11 env. See Env.
        eb_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmd = (f"{self.env.prelude} && "
               f"PYTHONPATH={self.env._worker_pythonpath(eb_parent)} "
               f"exec python -m envbridge._worker {shlex.quote(path)}")
        # `bash -c`, not `-lc`: a login shell would re-source profiles and reset to some base python,
        # losing the parent's active env -- so a no-op prelude (Env.shell) must run in a plain shell
        # that INHERITS this process's environment (parent's active conda env stays on PATH, so
        # `python` is the parent's). Real conda/spack preludes still override PATH and win.
        self._proc = subprocess.Popen(["bash", "-c", cmd])
        # A failure here raises out of __enter__, so __exit__ never runs: tear down the subprocess and
        # the socket before re-raising, or a hung activation outlives the caller's error.
        try:
            self._conn, _ = self._sock.accept()
            if self._conn.recv(2) != b"HI":
                raise RemoteError(f"worker for {self.env.label} did not hand-shake")
        except socket.timeout:
            self.close(kill=True)
            raise RemoteError(
                f"env {self.env.label!r} never connected within {self.timeout}s -- activation likely "
                f"failed. Prelude was: {self.env.prelude}")
        except BaseException:
            self.close(kill=True)
            raise
        self._conn.settimeout(self.call_timeout)
        return self

    def call(self, func: str, *args, **kwargs):
        """Run `module:qualname` in the target env with these args; return its result (or raise)."""
        if self._conn is None:
            raise RuntimeError("worker not started; use `with Worker(env) as w:` or call .start()")
        try:
            _codec.send(self._conn, {"func": func, "args": list(args), "kwargs": kwargs})
            resp = _codec.recv(self._conn)
        except OSError:
            try:
                self.close(kill=True)
            finally:
                self._conn = None
            raise
        if resp.get("ok"):
            return resp["value"]
        raise RemoteError(resp.get("error", "unknown remote error")
                          + "\n--- remote traceback ---\n" + resp.get("traceback", ""))

    def close(self, kill: bool = False):
        """Close the sockets, remove the socket file, and reap the worker (killed if `kill`)."""
        for c in (self._conn, self._sock):
            try:
                c and c.close()
            except OSError:
                pass
        if self._path:
            try:
                os.unlink(self._path)            # closing an AF_UNIX socket leaves its file behind
            except OSError:
                pass
            self._path = None
        if self._proc:
            if kill:
                self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()


def call(env: Env, func: str, *args, **kwargs):
    """One-shot: start a worker in `env`, run `module:qualname`, return the result, tear down."""
    with Worker(env) as w:
        return w.call(func, *args, **kwargs)


def remote(env: Env):
    """Decorator: the wrapped function's BODY runs in `env`. The function must live in an importable
    module (so the worker can resolve it by `module:qualname`) -- not a closure or __main__-only def."""
    def deco(fn):
        spec = f"{fn.__module__}:{fn.__qualname__}"

        def wrapper(*args, **kwargs):
            return call(env, spec, *args, **kwargs)
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper._envbridge_spec = spec
        return wrapper
    return deco
