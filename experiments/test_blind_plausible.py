"""glint_fast.plausible_t: the blind searches score out a basis the anneal flattened, by has_cell's rule.

WHY THIS EXISTS. The triplets entering the blind anneal pass |det| >= 0.1 |a||b||c|, but the anneal (three
least-squares steps) can pull one onto a plane on a frame with few peaks. A flat basis fits the peaks as a
two-dimensional lattice, so it scored at or above the real cell, won the frame and went into the consensus vote;
reduced, one such basis came out with a 1.01 A axis. ``plausible_t`` applies ``glint.stream.has_cell``'s rule to
the annealed batch on the device, and both blind searches drop what fails it before ranking.

WHAT IT CHECKS:
  * plausible_t and has_cell agree, basis by basis, on real cells (tetragonal, hexagonal, monoclinic, a reduced
    cell with all angles 60, triclinic, rotated) and on the bases has_cell turns down (a zero-length axis, a NaN
    entry, 0.125 A^3, and c = a + b lifted 0.5 A off the ab plane);
  * on the four frames of experiments/frames_cxidb_clean.txt where the search returned a cell has_cell turns
    down (12, 20, 58, 118; on 12 and 118 it was the top-1, two identical 37.8 A axes and zero volume), every cell
    index_blind_nbest returns is one has_cell accepts, so the stream writers keep every frame the search indexes.
    All 120 frames give the same top-1 lattice count with and without the filter (85 on the lysozyme lattice).

SKIPS, exit 0, without torch or below pyproject's torch floor (>= 1.12), as test_nbest_prefix.py does.

  PYTHONPATH=. python experiments/test_blind_plausible.py
"""
import os
import sys
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

try:
    import torch
except ImportError:
    print(f"SKIP {os.path.basename(__file__)} -- no torch: glint.glint_fast cannot import here")
    sys.exit(0)
_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if _ver < (1, 12):
    print(f"SKIP {os.path.basename(__file__)} -- torch {torch.__version__} is below pyproject's "
          f"floor (>= 1.12); the indexer is not supported there")
    sys.exit(0)

import numpy as np                                                         # noqa: E402
import glint.glint_fast as gf                                              # noqa: E402
from glint.glint_fast import DEV, index_blind_nbest, load, plausible_t     # noqa: E402
from glint.lattice import cell_to_Ar                                       # noqa: E402
from glint.stream import has_cell                                          # noqa: E402

FRAMES = os.path.join(HERE, "frames_cxidb_clean.txt")
FRAMES_HIT = (12, 20, 58, 118)          # the frames whose N-best held a has_cell reject before the filter
fails = []


def check(name, cond, msg=""):
    print(f"  {name:72s}: {'PASS' if cond else 'FAIL'}   {msg}")
    if not cond:
        fails.append(name)


def rot(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def check_primitivize_fallback():
    seeds = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                         dtype=torch.float32, device=DEV)
    q = np.arange(1, 25, dtype=float).reshape(8, 3)
    calls = []

    def fake_primitivize(M, Q):
        calls.append(None)
        return np.zeros((3, 3)) if len(calls) == 1 else np.eye(3)

    def fake_refine(starts, Q, w, qmax):
        return torch.zeros((4, 3, 3), dtype=Q.dtype, device=Q.device)

    def fake_objective(T, Q, **kwargs):
        return torch.zeros(T.shape[0], dtype=T.dtype, device=T.device), None

    def fake_maxima(T, f, keep):
        return seeds

    def fake_anneal(M0, Q, **kwargs):
        return torch.eye(3, dtype=M0.dtype, device=M0.device).expand(M0.shape[0], -1, -1).clone()

    def fake_score(M, Q):
        return torch.arange(M.shape[0], 0, -1, dtype=M.dtype, device=M.device), torch.full(
            (M.shape[0],), 8, dtype=M.dtype, device=M.device)

    common = {
        "_refine": fake_refine,
        "objective": fake_objective,
        "distinct_maxima_gpu": fake_maxima,
        "anneal_batch_t": fake_anneal,
        "score_batch_t": fake_score,
        "buerger_reduce": lambda M: M,
        "primitivize": fake_primitivize,
    }
    with patch.multiple(gf, **common):
        fast_cell = gf.index_blind_fast(q)
        check("fast search skips a cell flattened by primitivization",
              fast_cell is not None and has_cell(fast_cell) and len(calls) == 2,
              f"calls={len(calls)}")

    calls.clear()
    with patch.multiple(gf, **common):
        with patch.object(gf, "distinct_cells_gpu",
                          lambda M, key: torch.arange(M.shape[0], device=M.device)):
            nbest = gf.index_blind_nbest(q, N=1)
        check("N-best search skips a cell flattened by primitivization",
              len(nbest) == 1 and has_cell(nbest[0][0]) and len(calls) == 2,
              f"calls={len(calls)}")


def main():
    LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
    flat = LYSO.copy(); flat[:, 2] = 0.0
    thin = LYSO.copy(); thin[:, 2] = LYSO[:, 0] + LYSO[:, 1] + 0.5 * LYSO[:, 2] / np.linalg.norm(LYSO[:, 2])
    nan = LYSO.copy(); nan[1, 1] = np.nan
    good = [LYSO, rot(1) @ LYSO, cell_to_Ar(80.0, 80.0, 200.0, 90, 90, 120),
            cell_to_Ar(111.94, 172.23, 41.23, 90, 106.2, 90), cell_to_Ar(50.0, 50.0, 50.0, 60, 60, 60),
            rot(2) @ cell_to_Ar(31.0, 43.0, 57.0, 77, 84, 69)]
    bad = [flat, nan, 0.5 * np.eye(3), thin, rot(3) @ thin]
    B = np.array(good + bad)
    got = plausible_t(torch.as_tensor(B, dtype=torch.float64, device=DEV)).cpu().numpy().tolist()
    want = [has_cell(M) for M in B]
    check("plausible_t agrees with has_cell on every basis", got == want, f"{got} vs {want}")
    check("real cells pass, the four kinds has_cell turns down do not",
          want == [True] * len(good) + [False] * len(bad), want)

    check_primitivize_fallback()

    allf = [np.asarray(q, float) for q in load(FRAMES)]
    frames = [allf[i] for i in FRAMES_HIT]
    print(f"device {DEV}, torch {torch.__version__}; frames {FRAMES_HIT} "
          f"({', '.join(str(len(q)) for q in frames)} peaks)")
    nb = [index_blind_nbest(q, 3) for q in frames]
    cells = [c for x in nb for c, _ in x]
    check("each of the four frames gives hypotheses", all(nb), [len(x) for x in nb])
    check("every returned cell is one has_cell accepts (before the filter: 7 of 12 here were not)",
          all(has_cell(c) for c in cells), sum(not has_cell(c) for c in cells))

    print(f"\n{'FAILURES: ' + ', '.join(fails) if fails else 'ALL PASS'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
