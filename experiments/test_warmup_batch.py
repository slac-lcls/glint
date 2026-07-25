"""CPU tests for glint.warmup_batch (parallel MAD-triaged blind warm-up). No GPU, no driver.
Run: `python experiments/test_warmup_batch.py` or `pytest`."""
import numpy as np
from glint.warmup_batch import mad_triage, triage_order, warmup_consensus


def test_mad_triage_ranks():
    """Batched MAD scores signal-rich events far above blanks; triage_order picks exactly them."""
    B, H, W = 12, 48, 48
    rng = np.random.default_rng(0)
    stack = (100 + rng.normal(0, 3, (B, H, W))).astype(np.float32)
    sigframes = [2, 5, 7, 9]
    for f in sigframes:                                          # 40 bright peaks at RANDOM pixels/event
        ys, xs = rng.integers(0, H, 40), rng.integers(0, W, 40)
        stack[f, ys, xs] += 100.0
    _, _, score = mad_triage(stack, z0=4.0)
    sig_scores, blank_scores = score[sigframes], np.delete(score, sigframes)
    assert sig_scores.min() > blank_scores.max(), (score.tolist())   # clean separation
    assert set(triage_order(score, topk=4, floor=1)) == set(sigframes), score.tolist()


def _cell(rng):                                                 # a valid but random reciprocal basis
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R @ np.diag(1.0 / rng.uniform(50, 110, 3))

_R0 = np.array([[0.8776, -0.4794, 0], [0.4794, 0.8776, 0], [0, 0, 1.0]])   # fixed rotation (theta=0.5)
M_TRUE = _R0 @ np.diag([1 / 79.0, 1 / 79.0, 1 / 38.0])          # lyso, one true orientation

def _fake_blind(tag, nbest, rng):
    """index_blind_nbest stand-in: 'g' frames surface the true cell; 'b' frames scatter."""
    if tag == "g":
        return [(M_TRUE.copy(), 1.0)] + [(_cell(rng), 0.3) for _ in range(nbest - 1)]
    return [(_cell(rng), 0.5) for _ in range(nbest)]

def _rc():
    from glint.running_consensus import RunningConsensus
    return RunningConsensus(min_support=3, gap=2)


def test_warmup_locks_one_round():
    """5 promising + 3 junk frames -> the pooled N-best locks the true cell in ONE consensus round."""
    rng = np.random.default_rng(1)
    qs = ["g"] * 5 + ["b"] * 3
    Mc, sup = warmup_consensus(qs, lambda t, k: _fake_blind(t, k, rng), _rc(), nbest=3)
    assert Mc is not None and sup >= 3, (None if Mc is None else "ok", sup)
    axes = np.sort(1.0 / np.linalg.norm(np.asarray(Mc), axis=0))
    assert np.allclose(axes, [38, 79, 79], rtol=0.05), axes.tolist()   # recovered lyso axes


def test_fanout_matches_serial():
    """A parallel fan-out (workers finish out of order) yields the SAME verdict as the serial loop.
    Good frames surface the deterministic true cell; junk scatters (one shared rng per run)."""
    qs = ["g"] * 5 + ["b"] * 3
    rng_s = np.random.default_rng(2)
    serial = warmup_consensus(qs, lambda t, k: _fake_blind(t, k, rng_s), _rc(), nbest=3, fanout=None)
    rng_f = np.random.default_rng(3)
    blindf = lambda t, k: _fake_blind(t, k, rng_f)
    def par(Q, k):                                             # out-of-order workers, order-preserving return
        out = [None] * len(Q)
        for idx in reversed(range(len(Q))):
            out[idx] = blindf(Q[idx], k)
        return out
    fan = warmup_consensus(qs, blindf, _rc(), nbest=3, fanout=par)
    def axes(Mc):
        return np.sort(1.0 / np.linalg.norm(np.asarray(Mc), axis=0))
    assert serial[0] is not None and fan[0] is not None, (serial, fan)
    assert serial[1] == fan[1], (serial[1], fan[1])            # same vote tally regardless of order
    assert np.allclose(axes(serial[0]), [38, 79, 79], rtol=0.05) and np.allclose(axes(fan[0]), [38, 79, 79], rtol=0.05)


if __name__ == "__main__":
    ok = 0
    for t in (test_mad_triage_ranks, test_warmup_locks_one_round, test_fanout_matches_serial):
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/3 passed")
