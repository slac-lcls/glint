"""A captured known-cell CUDA graph keeps the cell parameters it was captured against alive.

replica_gpu_batch._StageGraph captures _stage_compute against P = _cell_params(Mc), and the graph then reads
P's tensors BY ADDRESS on every replay. In fp32 (KC_FP=32, the default) _cell_params makes fresh copies of the
azimuth grid (ca, sa = CA.to(FP)) that only that P references. index_all_graph caches the graph in _GRAPHS but
builds a NEW P on every call, so once the capturing call had returned, the cached graph read freed memory that
the caching allocator could hand to anything: on an A100 a second fp32 call over the same 480 lysozyme frames
differed on 437-448 of them (review-r2 GPU job 39724840, on the nonfinite-q-rows tree f9c2e3d, whose graph code
is main's; fp64, where .to(FP) is a no-op, was equal).

A CPU cannot capture a graph, so this checks the ownership the fix adds, with torch.cuda's stream and graph API
stood in by no-ops: build a _StageGraph from a fresh fp32 P, drop every other reference to P, and every tensor P
held must still be alive (weakref). A control shows the same P's copies ARE collected without the graph, so the
check can fail. The GPU replay check is pending: no GPU has run this commit yet.

SKIPS, exit 0, without torch or below pyproject's torch floor (>= 1.12), like test_kc_precision_default.py.

  PYTHONPATH=. python experiments/test_stage_graph_keeps_params.py
"""
import contextlib
import gc
import os
import re
import sys
import weakref

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

try:
    import torch
except ImportError:
    print(f"SKIP {os.path.basename(__file__)} -- no torch: glint.replica_gpu_batch cannot import here")
    sys.exit(0)
_m = re.match(r"(\d+)\.(\d+)", torch.__version__)
if _m is None:
    print(f"SKIP {os.path.basename(__file__)} -- unparseable torch version: {torch.__version__}")
    sys.exit(0)
if tuple(int(x) for x in _m.groups()) < (1, 12):
    print(f"SKIP {os.path.basename(__file__)} -- torch {torch.__version__} is below pyproject's floor (>= 1.12)")
    sys.exit(0)

os.environ["KC_FP"] = "32"                     # the default, set so an inherited KC_FP=64 cannot hide the case
sys.path.insert(0, ROOT)
import numpy as np                             # noqa: E402
import glint.replica_gpu_batch as rgb          # noqa: E402
from glint.glint_fast import LYSO              # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("ok    " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


class _Stream:
    def wait_stream(self, other):
        pass


class _Graph:
    def replay(self):
        pass


@contextlib.contextmanager
def _noop(*a, **k):
    yield


STAND_INS = dict(Stream=lambda *a, **k: _Stream(), current_stream=lambda *a, **k: _Stream(),
                 stream=_noop, CUDAGraph=_Graph, graph=_noop)


def tensor_refs(P):
    return [weakref.ref(t) for t in P if isinstance(t, torch.Tensor)]


def main():
    check("KC_FP=32: the module's working precision is float32", rgb.FP == torch.float32, str(rgb.FP))
    Mc = np.asarray(LYSO, float)

    # control: without a graph, P's fp32 azimuth copies die with P (else the check below could not fail)
    P = rgb._cell_params(Mc)
    refs = tensor_refs(P)
    del P
    gc.collect()
    n_dead = sum(r() is None for r in refs)
    check("control: dropping P frees the tensors only P held (its fp32 azimuth copies)", n_dead >= 2,
          f"{n_dead} of {len(refs)} tensors freed")

    saved = {k: getattr(torch.cuda, k) for k in STAND_INS}
    for k, v in STAND_INS.items():
        setattr(torch.cuda, k, v)
    try:
        P = rgb._cell_params(Mc)
        refs = tensor_refs(P)
        G = rgb._StageGraph(P, 2, 64)
        del P
        gc.collect()
        alive = [r() is not None for r in refs]
        check("a _StageGraph keeps every tensor of the P it was captured against alive",
              all(alive), f"{sum(alive)} of {len(alive)} alive")
        del G
    finally:
        for k, v in saved.items():
            setattr(torch.cuda, k, v)

    print()
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
