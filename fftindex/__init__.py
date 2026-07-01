"""Back-compat alias: this package was renamed ``fftindex`` -> ``glint``.

New code should ``import glint``.  This shim keeps existing scripts that do
``import fftindex`` / ``from fftindex.<mod> import ...`` working: it makes ``fftindex``
*be* the ``glint`` package and aliases every ``glint`` submodule as ``fftindex.<mod>``
(same module object, so ``fftindex.geom is glint.geom``).
"""
import importlib as _importlib
import pkgutil as _pkgutil
import sys as _sys

_glint = _importlib.import_module("glint")

# make `fftindex` resolve to the real package object
_sys.modules[__name__] = _glint

# eagerly alias submodules so `import fftindex.<mod>` returns the *same* object as glint.<mod>
for _m in _pkgutil.iter_modules(_glint.__path__):
    _name = _m.name
    try:
        _sys.modules[f"{__name__}.{_name}"] = _importlib.import_module(f"glint.{_name}")
    except Exception:
        # optional-dependency modules (e.g. torch/numba-guarded) resolve lazily on real use
        pass
