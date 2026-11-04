"""Trivial functions the worker can import by `envbridge._selftest:<name>` during self-test."""
from __future__ import annotations


def echo(x):
    return x


def add(a, b):
    return a + b


def array_stats(arr):
    """Take a numpy array, return a dict mixing arrays and JSON scalars (exercises nested codec)."""
    import numpy as np
    a = np.asarray(arr)
    return {"sum": float(a.sum()), "shape": list(a.shape), "doubled": a * 2}


def boom():
    raise ValueError("intentional failure from the target env")


def which_numpy():
    import numpy as np
    return np.__version__

def psana_site() -> dict:
    """Import psana and report where it came from. For the cross-env test on S3DF only.

    This is the function the PYTHONPATH bug broke: conda2's psconda.sh exports a PYTHONPATH
    pointing at a psana built for Python 3.9, and when the worker inherited it that release
    shadowed the python3.11 psana inside xpp_drp_gpu_311, so this import died with "cannot import
    name 'dgram' from partially initialized module 'psana'". Returning the interpreter version and
    psana's __file__ together is what makes a regression visible: the two must agree.
    """
    import sys
    from psana import DataSource          # noqa: F401  -- this is the import that used to fail
    import psana
    return {"python": ".".join(str(v) for v in sys.version_info[:3]),
            "psana": psana.__file__}


def sleep(seconds):
    """Hold the call open for `seconds`, then return them (exercises Worker.call_timeout)."""
    import time
    time.sleep(seconds)
    return seconds


def die():
    """Exit the worker mid-call without replying (a crashed reader)."""
    import os
    os._exit(3)
