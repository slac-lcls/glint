"""CPU tests for glint.alias_fold: a min_lead refusal that is only a lattice tied with its own index-2
relative on the same frames. Pure numpy; the blind search is replaced by scripted N-best lists.

Dual mode: `pytest experiments/test_alias_fold.py`, or `python experiments/test_alias_fold.py` for a
PASS/FAIL loop. The helper tests are torch-free; the three hybrid_index wiring tests import
glint.hybrid_stream (which needs torch) and SKIP without it, so the CPU job runs the helper and the
torch-CPU job runs everything.

The pools are SYNTHETIC. They reproduce the shape of the refusal seen on a real run (leader vs an
index-2 supercell of it, lead ~1.2, on the same frames); they are not evidence that the fold helps on
real data. That needs a real-data run with a shuffled-frame null.
"""
import contextlib

import numpy as np

from glint.alias_fold import fold_alias_relatives
from glint.alias_gate import AliasGate, hnf_matrices
from glint.lattice import random_rotation
from glint.multishot import consensus_cell, same_lattice

SEED = 20261115
M_TRUE = np.diag([40.0, 55.0, 70.0])            # orthorhombic P, distinct axes
H2 = hnf_matrices(2)[-1]                        # a det-2 form: M_TRUE @ H2 is an index-2 supercell
M_OTHER = np.diag([44.0, 51.0, 77.0])           # an unrelated lattice


def _still_q(M, rng, n=120, qmax=0.30):
    """A sparse still of lattice M (same construction as experiments/test_alias_gate.py)."""
    Minv = np.linalg.inv(M)
    hb = np.ceil(qmax * np.linalg.norm(M, axis=0)).astype(int)
    out = []
    while sum(len(o) for o in out) < n:
        H = np.column_stack([rng.integers(-hb[i], hb[i] + 1, 4 * n) for i in range(3)]).astype(float)
        q = H[np.abs(H).sum(1) > 0] @ Minv
        out.append(q[np.sqrt((q * q).sum(1)) <= qmax])
    q = np.vstack(out)[:n]
    return q + rng.normal(0.0, 0.0008, q.shape)


def _scenario(rng, k=24, n_lead=24, n_rel=19, rel="super", rel_frames="same", junk=True):
    """Frames of M_TRUE and a scripted N-best list per frame.

    The true cell (in each frame's orientation) is a hypothesis on the first `n_lead` frames; a relative
    of it on `n_rel` frames -- the SAME frames as the leader's (rel_frames="same") or frames where the
    true cell was not proposed ("disjoint", the shuffled-label null). rel = "super" (M_TRUE @ H2),
    "truth-is-runner" (the supercell gets the larger count), or "other" (an unrelated lattice)."""
    n_frames = k if rel_frames == "same" else n_lead + n_rel
    R = [random_rotation(rng) for _ in range(n_frames)]
    frames = [_still_q(R[i] @ M_TRUE, rng) for i in range(n_frames)]
    nb = [[] for _ in range(n_frames)]
    if rel == "truth-is-runner":
        n_lead, n_rel = n_rel, n_lead
    true_on = list(range(n_lead))
    rel_on = list(range(n_rel)) if rel_frames == "same" else list(range(n_lead, n_lead + n_rel))
    S = M_OTHER if rel == "other" else M_TRUE @ H2
    for i in true_on:
        nb[i].append((R[i] @ M_TRUE, 1.0))
    for i in rel_on:
        nb[i].append((R[i] @ S, 0.9))
    if junk:
        for i in range(n_frames):
            J = np.diag(rng.uniform(25.0, 110.0, 3)) @ random_rotation(rng)
            nb[i].append((R[i] @ J, 0.1))
    return frames, nb


def _pool(nb):
    pool = [c for x in nb for c, _ in x]
    fid = [i for i, x in enumerate(nb) for _ in x]
    return pool, fid


LEAD = dict(min_frac=0.02, min_lead=1.5)


def _refused(pool):
    det = {}
    Mc, _ = consensus_cell(pool, detail=det, **LEAD)
    return Mc, det


# ---------------------------------------------------------------------------------- the helper itself
def test_precondition_vote_refuses_on_min_lead_only():
    """The setup reproduces the observed refusal: the vote itself refuses, and only on the ratio test."""
    frames, nb = _scenario(np.random.default_rng(SEED))
    Mc, det = _refused(_pool(nb)[0])
    assert Mc is None and det["failed"] == ["min_lead"], det


def test_fold_drops_a_supercell_runner_and_accepts_the_lattice():
    frames, nb = _scenario(np.random.default_rng(SEED + 1))
    pool, fid = _pool(nb)
    Mc, sup, info = fold_alias_relatives(pool, fid, frames, AliasGate(), **LEAD)
    assert Mc is not None and same_lattice(Mc, M_TRUE), info
    assert sup == 24, info
    assert [p["outcome"] for p in info["pairs"]] == ["relative dropped"], info["pairs"]


def test_fold_drops_a_supercell_leader_when_the_lattice_is_the_runner():
    """The gate, not the count, decides: if the supercell outnumbers the lattice it is the LEADER that goes."""
    frames, nb = _scenario(np.random.default_rng(SEED + 2), rel="truth-is-runner")
    pool, fid = _pool(nb)
    Mc0, det = _refused(pool)
    assert Mc0 is None and det["failed"] == ["min_lead"], det
    assert not same_lattice(np.diag(det["cell"]), M_TRUE)     # the vote's leader IS the supercell
    Mc, sup, info = fold_alias_relatives(pool, fid, frames, AliasGate(), **LEAD)
    assert Mc is not None and same_lattice(Mc, M_TRUE), info
    assert info["pairs"][0]["outcome"] == "leader dropped", info["pairs"]


def test_null_relative_on_other_frames_is_not_folded():
    """Shuffled-label null: the same two lattices, but never proposed on the same frame. No evidence
    that they are one population, so the refusal stands."""
    frames, nb = _scenario(np.random.default_rng(SEED + 3), rel_frames="disjoint")
    pool, fid = _pool(nb)
    assert _refused(pool)[0] is None
    Mc, _, info = fold_alias_relatives(pool, fid, frames, AliasGate(), **LEAD)
    assert Mc is None, info
    assert [p["outcome"] for p in info["pairs"]] == ["not on the same frames"], info["pairs"]


def test_unrelated_runner_is_not_folded():
    """A genuinely different lattice tied with the leader is a real ambiguity: refuse, examine nothing."""
    frames, nb = _scenario(np.random.default_rng(SEED + 4), rel="other")
    pool, fid = _pool(nb)
    assert _refused(pool)[0] is None
    Mc, _, info = fold_alias_relatives(pool, fid, frames, AliasGate(), **LEAD)
    assert Mc is None and info["pairs"] == [], info


def test_dropped_votes_still_count_in_the_pool():
    """min_frac keeps the full pool as its denominator: folding never shrinks the lottery it guards."""
    frames, nb = _scenario(np.random.default_rng(SEED + 5))
    pool, fid = _pool(nb)
    _, _, info = fold_alias_relatives(pool, fid, frames, AliasGate(), **LEAD)
    assert info["n_valid"] == len(pool), info


# ---------------------------------------------------------------------------- wiring in hybrid_index
class Skip(Exception):
    pass


def _hs():
    try:
        import glint.hybrid_stream as hs
    except ImportError as e:                                   # torch absent: the CPU job
        raise Skip(f"glint.hybrid_stream cannot import here ({e})")
    return hs


@contextlib.contextmanager
def _patched(mod, **attrs):
    old = {k: getattr(mod, k) for k in attrs}
    try:
        for k, v in attrs.items():
            setattr(mod, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(mod, k, v)


def _run_hybrid(frames, nb, **kw):
    """hybrid_index with the blind search scripted and the known-cell rescue switched off."""
    hs = _hs()
    by_id = {id(q): x for q, x in zip(frames, nb)}
    with _patched(hs, index_blind_nbest=lambda q, n=3: by_id.get(id(q), []),
                  index_known_gpu_cell=lambda *a, **k: None,
                  CONSENSUS_MIN_FRAC=0.02, CONSENSUS_MIN_LEAD=1.5):
        return hs.hybrid_index(frames, warmup=False, **kw)


def test_hybrid_default_still_refuses():
    frames, nb = _scenario(np.random.default_rng(SEED + 6))
    _, stats = _run_hybrid(frames, nb)
    assert stats["consensus_refused"], stats["consensus_detail"]
    assert "alias_fold" not in stats["consensus_detail"]


def test_hybrid_alias_fold_accepts():
    """Fails on main (no alias_fold argument); with the fold the same pool locks the true lattice and
    every frame is registered on it."""
    frames, nb = _scenario(np.random.default_rng(SEED + 6))
    results, stats = _run_hybrid(frames, nb, alias_gate=AliasGate(), alias_fold=True)
    assert not stats["consensus_refused"], stats["consensus_detail"]
    assert stats["consensus_detail"]["alias_fold"]["pairs"][0]["outcome"] == "relative dropped"
    assert all(r["M"] is not None and same_lattice(r["M"], M_TRUE) for r in results)


def test_hybrid_alias_fold_needs_a_gate():
    frames, nb = _scenario(np.random.default_rng(SEED + 7))
    try:
        _run_hybrid(frames, nb, alias_fold=True)
    except ValueError:
        return
    raise AssertionError("alias_fold=True without alias_gate should raise")


if __name__ == "__main__":
    fails = skips = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Skip as e:
                skips += 1
                print(f"SKIP {name}: {e}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}" + (f" ({skips} skipped)" if skips else ""))
    raise SystemExit(1 if fails else 0)
