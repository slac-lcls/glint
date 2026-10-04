"""GLINT — GPU-native blind serial-crystallography indexing.

Copyright (c) 2026, The Board of Trustees of the Leland Stanford Junior University, through SLAC National Accelerator Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.
GLINT was developed at SLAC with support from DOE's Basic Energy Sciences and Advanced
Scientific Computing Research programs, including the ILLUMINE project. Work at SLAC National
Accelerator Laboratory is supported by the U.S. Department of Energy under contract
DE-AC02-76SF00515.

Neither the name of the Leland Stanford Junior University, SLAC National Accelerator
Laboratory, U.S. Department of Energy nor the names of its contributors may be used to
endorse or promote products derived from this software without specific prior written
permission.

See the COPYRIGHT file at the repository root.
"""
__all__ = [
    "simulate_shot",
    "Shot",
    "index_shot",
    "IndexResult",
    "score",
    "fft_volume",
    "central_rays",
    "make_dataset",
    "peak_features",
    "LearnedPeakFinder",
]

_EXPORTS = {
    "make_dataset": (".dataset", "make_dataset"),
    "LearnedPeakFinder": (".detector", "LearnedPeakFinder"),
    "peak_features": (".features", "peak_features"),
    "index_shot": (".index", "index_shot"),
    "IndexResult": (".index", "IndexResult"),
    "score": (".metrics", "score"),
    "simulate_shot": (".simulate", "simulate_shot"),
    "Shot": (".simulate", "Shot"),
    "central_rays": (".transform", "central_rays"),
    "fft_volume": (".transform", "fft_volume"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
