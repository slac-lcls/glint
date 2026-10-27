"""Pickle-free wire codec for values that cross the env boundary.

The whole point of this library is bridging environments with INCOMPATIBLE numpy ABIs (1.x vs 2.x).
pickle is unsafe across that boundary -- an object-array pickled under numpy 2.0 is not guaranteed to
load under 1.x. So the wire is deliberately restricted to two things:

  * numpy arrays, encoded as (dtype string, shape, raw C-order bytes) -- all three are version-stable,
    so `np.frombuffer(buf, dtype).reshape(shape)` reconstructs an identical array under any numpy.
  * JSON-able scalars/containers (None/bool/int/float/str/list/dict).

That restriction is not a limitation to apologise for; it is the thing that makes the bridge SAFE.
Anything else raises at encode time, loudly, rather than corrupting silently on the far side.

A message is: one JSON blob (the structure, with arrays replaced by {"__nd__": [dtype, shape, i]}
placeholders) followed by N raw array buffers. numpy is imported lazily so the codec still loads in an
env without numpy as long as no arrays are passed.
"""
from __future__ import annotations

import json
import struct

_ND = "__nd__"


def _walk_encode(obj, blobs):
    """Return a JSON-able mirror of obj, appending array bytes to `blobs`."""
    tname = type(obj).__module__ + "." + type(obj).__qualname__
    if tname == "numpy.ndarray":
        import numpy as np
        if obj.dtype.hasobject or obj.dtype.fields is not None:
            raise TypeError(
                f"envbridge: cannot send an array of dtype {obj.dtype} -- object and structured "
                f"dtypes do not survive the (dtype string, shape, bytes) wire. Send plain arrays.")
        a = np.ascontiguousarray(obj)            # may turn a 0-d array into shape (1,) ...
        blobs.append(a.tobytes())
        return {_ND: [a.dtype.str, list(obj.shape), len(blobs) - 1]}   # ... so record obj's shape
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_walk_encode(x, blobs) for x in obj]
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            raise TypeError("envbridge: dict keys must be strings to cross the boundary")
        if _ND in obj:
            # the decoder reads any dict holding this key as an array placeholder
            raise TypeError(f"envbridge: a dict key {_ND!r} is reserved for arrays on the wire")
        return {k: _walk_encode(v, blobs) for k, v in obj.items()}
    raise TypeError(
        f"envbridge: cannot send a {tname!r} across the env boundary. The wire is restricted to "
        f"numpy arrays + JSON types on purpose (pickle is unsafe across incompatible numpy ABIs). "
        f"Reduce it to arrays/JSON on the sending side.")


def _walk_decode(obj, blobs):
    if isinstance(obj, dict) and _ND in obj:
        import numpy as np
        dtype, shape, i = obj[_ND]
        return np.frombuffer(blobs[i], dtype=np.dtype(dtype)).reshape(shape)
    if isinstance(obj, list):
        return [_walk_decode(x, blobs) for x in obj]
    if isinstance(obj, dict):
        return {k: _walk_decode(v, blobs) for k, v in obj.items()}
    return obj


def encode(obj) -> bytes:
    """obj -> a single length-framed bytes payload (header + array blobs)."""
    blobs: list[bytes] = []
    header = json.dumps(_walk_encode(obj, blobs)).encode("utf-8")
    parts = [struct.pack("<I", len(header)), header, struct.pack("<I", len(blobs))]
    for b in blobs:
        parts.append(struct.pack("<Q", len(b)))
        parts.append(b)
    return b"".join(parts)


def _recvn(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("envbridge: peer closed mid-message")
        buf += chunk
    return bytes(buf)


def send(sock, obj) -> None:
    payload = encode(obj)
    sock.sendall(struct.pack("<Q", len(payload)) + payload)


def recv(sock):
    (total,) = struct.unpack("<Q", _recvn(sock, 8))
    payload = _recvn(sock, total)
    off = 0
    (hlen,) = struct.unpack("<I", payload[off:off + 4]); off += 4
    header = json.loads(payload[off:off + hlen]); off += hlen
    (nblob,) = struct.unpack("<I", payload[off:off + 4]); off += 4
    blobs = []
    for _ in range(nblob):
        (blen,) = struct.unpack("<Q", payload[off:off + 8]); off += 8
        blobs.append(payload[off:off + blen]); off += blen
    return _walk_decode(header, blobs)
