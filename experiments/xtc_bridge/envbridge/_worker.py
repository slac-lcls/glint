"""Sidecar entry point. Runs INSIDE the target env, activated by the parent's shell prelude.

Started as `python -m envbridge._worker <socket_path>` with PYTHONPATH pointing at envbridge's own
parent dir, so it imports even in an env where envbridge is not installed. Connects back to the
parent's listening AF_UNIX socket, then serves one request per loop until the parent closes:

    request  = {"func": "package.module:qualname", "args": [...], "kwargs": {...}}
    response = {"ok": True,  "value": <result>}
             | {"ok": False, "error": "<type>: <message>", "traceback": "<tb>"}

The function is imported fresh and called; only its arguments and return value cross the wire, through
the pickle-free codec. Nothing about the parent env is visible here -- this process is entirely the
target env.
"""
from __future__ import annotations

import importlib
import socket
import sys
import traceback

from . import _codec


def _resolve(spec: str):
    """'package.module:qualname' -> the callable."""
    if ":" not in spec:
        raise ValueError(f"envbridge: func spec must be 'module:qualname', got {spec!r}")
    mod_name, qual = spec.split(":", 1)
    obj = importlib.import_module(mod_name)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def main(argv=None):
    argv = argv or sys.argv[1:]
    sock_path = argv[0]
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    sock.sendall(b"HI")                       # handshake so the parent knows activation succeeded
    while True:
        try:
            req = _codec.recv(sock)
        except (ConnectionError, EOFError):
            break                             # parent went away -> exit cleanly
        try:
            fn = _resolve(req["func"])
            value = fn(*req.get("args", []), **req.get("kwargs", {}))
            _codec.send(sock, {"ok": True, "value": value})
        except BaseException as e:            # report the failure across the wire, keep serving
            _codec.send(sock, {"ok": False,
                               "error": f"{type(e).__name__}: {e}",
                               "traceback": traceback.format_exc()})
    sock.close()


if __name__ == "__main__":
    main()
