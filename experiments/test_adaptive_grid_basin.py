"""The batched known-cell engine uses its coarse 4096-direction anchor grid only where the grid is safe.

WHY THIS EXISTS (review r2, finding s4-02). replica_gpu_batch finds the anchor (the cell's shortest axis, L0) on a
Fibonacci half-sphere and refines it. The grid only has to land inside the anchor's angular basin, and that basin
narrows as 1/(L0*qmax). Up to 149370e every orthogonal cell got the coarse 4096 grid instead of the full 16384.
On synthetic stills of a 100x120x150 A cell at 1.8 A (L0*qmax 56) the default StreamDriver path then found the
right orientation on 26/60 frames, against 55/60 on the full grid; most misses were dropped as gate-rejected.
Lysozyme on cxidb-17 (L0*qmax <= 23), where the coarse grid was validated, was not affected. Now a frame gets the
coarse grid only if the cell is orthogonal AND L0 * (that frame's largest |q|) <= KC_ADAPT_L0Q (default 25).

WHAT IT CHECKS, all on CPU torch:
  1. The rule, observed on the grid _stage_compute actually runs with: lysozyme at 2.0 A, the six committed
     cxidb-17 frames with the largest |q|, and 100x120x150 at 4.0 A (L0*qmax = 25) stay on 4096;
     100x120x150 at 1.8 A and a 117x160x200 cell at 2.5 A get 16384; an oblique cell and KC_ADAPTIVE_DIRS off
     get 16384. Plus _adaptive_dirs(Mc, qmax) directly, and every committed cxidb-17 frame on 4096.
  2. A batch mixing frames that need different grids: each frame runs on its own grid, and the answer equals
     the answer from a batch of only its kind (split + reorder).
  3. The CUDA-graph path (index_all_graph) on that mixed batch. There is no GPU in CI, so it runs here with a
     stand-in for the captured graph that calls _stage_compute with the P it was captured with: a graph built
     for one grid must never run a frame that needs the other (the graph cache is keyed on the grid).
  4. Rate: 16 simulated stills of 100x120x150 at 1.8 A through the default call (index_fused, as StreamDriver
     makes it): at least 14 orientation-correct, and no more than 1 fewer than with full_grid=True.
     origin/main (149370e) gets 7/16 here (the full grid 16/16).
Checks 1, 2 and 3 audit every _stage_compute call: no frame with L0*|q|max > 25 may run on the 4096 grid.

SKIPS, exit 0, without torch (replica_gpu imports it unconditionally) or below pyproject's floor (>= 1.12).

  PYTHONPATH=. python experiments/test_adaptive_grid_basin.py      # exit 0 = all pass (CPU torch, ~30 s)
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

try:
    import torch
except ImportError:
    print(f"SKIP {os.path.basename(__file__)} -- no torch: glint.replica_gpu cannot import here")
    sys.exit(0)
if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (1, 12):
    print(f"SKIP {os.path.basename(__file__)} -- torch {torch.__version__} is below pyproject's floor (>= 1.12)")
    sys.exit(0)

import numpy as np                                                         # noqa: E402
import glint.replica_gpu_batch as rgb                                      # noqa: E402
from glint.glint_fast import LYSO, load                                    # noqa: E402
from glint.lattice import cell_to_Ar                                       # noqa: E402
from glint.simulate import simulate_shot                                   # noqa: E402

L0Q = 25.0                       # the documented KC_ADAPT_L0Q default
LO, HI = 4096, 16384
FAILS = []
AUDIT = []                       # one row per _stage_compute call: (frames, grid, frames needing 16384 run on 4096)


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + str(detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


_real_stage = rgb._stage_compute


def _audited_stage(Q, m, P):
    """_stage_compute, recording the grid it ran with and how many real frames in it need the full grid."""
    qn = (Q.double().norm(dim=2) * m).amax(1)
    real = m.any(1)
    grid = len(P[7])
    bad = int(((float(P[0]) * qn > L0Q * 1.001) & real).sum()) if grid == LO else 0
    AUDIT.append((int(real.sum()), grid, bad))
    return _real_stage(Q, m, P)


rgb._stage_compute = _audited_stage


def grids(frames, Mc, **kw):
    """The anchor grids the default call (index_fused, as StreamDriver makes it) runs these frames on."""
    n0 = len(AUDIT)
    rgb.index_fused(frames, Mc, B=max(len(frames), 1), **kw)
    return sorted({g for _, g, _ in AUDIT[n0:]})


def sphere_q(qmax, n, rng):
    """n q-vectors with the largest exactly at |q| = qmax (only the grid choice reads these, not the indexing)."""
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = qmax * rng.random(n) ** (1 / 3); r[0] = qmax
    return d * r[:, None]


def ori_ok(M, Mt):
    """M is a basis of the true lattice (integer unimodular change of basis), i.e. the right orientation."""
    if M is None:
        return False
    M = np.asarray(M, float)
    if abs(np.linalg.det(M)) < 1.0:
        return False
    T = np.linalg.solve(M, Mt)
    return bool(np.abs(T - np.rint(T)).max() < 0.1 and abs(round(np.linalg.det(np.rint(T)))) == 1)


def safe(name, fn):
    try:
        fn()
    except Exception as e:                                         # noqa: BLE001 -- a check that cannot run fails
        check(name, False, f"{type(e).__name__}: {e}")


rng = np.random.default_rng(20261001)
LONG = cell_to_Ar(100., 120., 150., 90., 90., 90.)
PSII = cell_to_Ar(117., 160., 200., 90., 90., 90.)
MONO = cell_to_Ar(60., 70., 80., 90., 105., 90.)
real = [np.asarray(q, float) for q in load(os.path.join(HERE, "frames_cxidb_clean.txt"))]
real = [q for q in real if len(q) >= 6]
qmax_real = np.array([np.linalg.norm(q, axis=1).max() for q in real])
top6 = [real[i] for i in np.argsort(qmax_real)[-6:]]

print("1. the grid rule, as run")
safe("lysozyme at 2.0 A -> 4096", lambda: check(
    "lysozyme at 2.0 A -> 4096", grids([sphere_q(1 / 2.0, 40, rng)], LYSO) == [LO]))
safe("the 6 cxidb-17 frames with the largest |q| -> 4096", lambda: check(
    "the 6 cxidb-17 frames with the largest |q| -> 4096", grids(top6, LYSO) == [LO],
    f"L0*|q|max up to {37.98 * qmax_real.max():.1f}"))
safe("100x120x150 at 4.0 A (L0*qmax = 25) -> 4096", lambda: check(
    "100x120x150 at 4.0 A (L0*qmax = 25) -> 4096", grids([sphere_q(1 / 4.0, 40, rng)], LONG) == [LO]))
safe("100x120x150 at 1.8 A (L0*qmax = 56) -> 16384", lambda: check(
    "100x120x150 at 1.8 A (L0*qmax = 56) -> 16384", grids([sphere_q(1 / 1.8, 40, rng)], LONG) == [HI]))
safe("117x160x200 at 2.5 A (L0*qmax = 47) -> 16384", lambda: check(
    "117x160x200 at 2.5 A (L0*qmax = 47) -> 16384", grids([sphere_q(1 / 2.5, 40, rng)], PSII) == [HI]))
safe("oblique cell (beta 105) at 4.0 A -> 16384", lambda: check(
    "oblique cell (beta 105) at 4.0 A -> 16384", grids([sphere_q(1 / 4.0, 40, rng)], MONO) == [HI]))


def _adapt_off():
    rgb._ADAPT = False
    try:
        check("KC_ADAPTIVE_DIRS=0: lysozyme at 2.0 A -> 16384", grids([sphere_q(1 / 2.0, 40, rng)], LYSO) == [HI])
    finally:
        rgb._ADAPT = True


safe("KC_ADAPTIVE_DIRS=0: lysozyme at 2.0 A -> 16384", _adapt_off)


def _direct():
    check("_adaptive_dirs(100x120x150, 1/1.8) has 16384 directions", len(rgb._adaptive_dirs(LONG, 1 / 1.8)) == HI)
    check("_adaptive_dirs(lysozyme, 1/2.0) has 4096 directions", len(rgb._adaptive_dirs(LYSO, 1 / 2.0)) == LO)
    check("_adaptive_dirs(qmax=None) is the full grid", len(rgb._adaptive_dirs(LYSO, None)) == HI)
    ok = rgb._coarse_ok(LYSO, rgb._frame_qmax(real))
    check("every committed cxidb-17 frame is on the 4096 grid", bool(ok.all()), f"{int(ok.sum())}/{len(real)}")
    check("_frame_qmax", np.allclose(rgb._frame_qmax([np.zeros((0, 3)), [[1., 0, 0], [0, 2., 0]], real[0]]),
                                     [0.0, 2.0, qmax_real[0]]))


safe("_adaptive_dirs / _coarse_ok / _frame_qmax", _direct)

print("2. a batch whose frames need different grids")
srng = np.random.default_rng(7)
hi_shots = [simulate_shot(cell=(100., 120., 150., 90., 90., 90.), n_target=100, dmin=1.8, pos_sigma=2e-4,
                          frac_spurious=0.2, rng=srng) for _ in range(16)]
lo_shots = [simulate_shot(cell=(100., 120., 150., 90., 90., 90.), n_target=60, dmin=4.5, pos_sigma=2e-4,
                          frac_spurious=0.2, rng=srng) for _ in range(4)]
mixed = [q for pair in zip([s.g for s in lo_shots], [s.g for s in hi_shots[:4]]) for q in pair]   # lo, hi, lo, hi ...


def _mixed():
    n0 = len(AUDIT)
    got = rgb.index_known_gpu_cell_batch(mixed, LONG)
    runs = AUDIT[n0:]
    lo_ref = rgb.index_known_gpu_cell_batch(mixed[0::2], LONG)
    hi_ref = rgb.index_known_gpu_cell_batch(mixed[1::2], LONG)
    ref = [M for pair in zip(lo_ref, hi_ref) for M in pair]
    check("the mixed batch runs on both grids", sorted(g for _, g, _ in runs) == [LO, HI], runs)
    same = all((a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b))
               for a, b in zip(got, ref))
    check("each frame's answer equals its single-grid batch's answer, in frame order", same)


safe("mixed batch", _mixed)

print("3. the CUDA-graph path's grouping and graph cache (stand-in graph, CPU)")


class _CudaLookalike(str):
    """DEV for index_all_graph's `DEV != "cuda"` test only; torch still reads it as the string 'cpu'."""
    def __eq__(self, o):
        return o == "cuda"

    def __ne__(self, o):
        return o != "cuda"

    __hash__ = str.__hash__


class _FakeGraph:
    """Stands in for _StageGraph: keeps the P it was 'captured' with and replays _stage_compute with it."""
    built = []

    def __init__(self, P, F, Pmax):
        self.P = P
        _FakeGraph.built.append(len(P[7]))

    def run(self, Q, m):
        return rgb._stage_compute(Q, m, self.P)


def _graph():
    dev, sg, graphs = rgb.DEV, rgb._StageGraph, rgb._GRAPHS
    rgb.DEV, rgb._StageGraph, rgb._GRAPHS = _CudaLookalike("cpu"), _FakeGraph, {}
    try:
        n0 = len(AUDIT)
        out1 = rgb.index_all_graph(mixed, LONG, B=3)       # chunks: 3 coarse | 1 coarse + 2 full (eager) | 2 full
        built1 = list(_FakeGraph.built)
        out2 = rgb.index_all_graph(mixed[::-1], LONG, B=3)
        runs = AUDIT[n0:]
        check("graphs were built for both grids", sorted(set(built1)) == [LO, HI], built1)
        check("a second call reuses the cached graphs", _FakeGraph.built == built1, _FakeGraph.built)
        bad = sum(b for _, _, b in runs)
        check("no frame needing 16384 ran on a 4096 graph", bad == 0, f"{bad} frame-runs")
        check("frame order is kept (reversed input: the same lattice basis per frame)",
              all((a is None and b is None)
                  or (a is not None and b is not None and (np.array_equal(a, b) or ori_ok(a, b)))
                  for a, b in zip(out1, out2[::-1])))
    finally:
        rgb.DEV, rgb._StageGraph, rgb._GRAPHS = dev, sg, graphs


safe("graph path", _graph)

print("4. rate: 16 stills of 100x120x150 A at 1.8 A through the default call")


def _rate():
    qs = [s.g for s in hi_shots]
    d = sum(ori_ok(M, s.M) for M, s in zip(rgb.index_fused(qs, LONG, B=len(qs)), hi_shots))
    f = sum(ori_ok(M, s.M) for M, s in zip(rgb.index_fused(qs, LONG, B=len(qs), full_grid=True), hi_shots))
    check("default call >= 14/16 orientation-correct", d >= 14, f"default {d}/16, full grid {f}/16")
    check("default call within 1 frame of full_grid=True", d >= f - 1, f"{d} vs {f}")


safe("rate", _rate)

bad = sum(b for _, _, b in AUDIT)
check("audit: no frame with L0*|q|max > 25 ran on the 4096 grid, in any call above", bad == 0,
      f"{bad} frame-runs over {len(AUDIT)} calls")

if FAILS:
    print(f"\nFAILED {len(FAILS)}: " + "; ".join(FAILS))
    sys.exit(1)
print("\nall adaptive-grid checks passed")
