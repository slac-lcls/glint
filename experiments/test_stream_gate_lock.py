"""CPU tests for the alias gate on the streaming driver's PRIMARY blind lock (StreamDriver._gate_lock).

The gate was previously reachable only from the watchdog's relock, which sits behind
`adaptive_relock` (False by default) -- so passing `alias_gate=AliasGate()` to a default
StreamDriver was a silent no-op and the blind lock that actually sets self.Mc was unguarded. These
tests pin the wiring: what the gate is fed (per frame, in each frame's own orientation), that a
refusal does not lock, and that the gate off is bit-identical.

`_gate_lock` is exercised directly on an uninitialised instance -- constructing a real StreamDriver
needs panels/geometry/peak-finder and a GPU blind indexer, none of which this logic touches.

Dual mode: `pytest experiments/test_stream_gate_lock.py`, or `python experiments/...` for PASS/FAIL.
"""
import numpy as np

from glint.alias_gate import AliasGate, hnf_matrices
from glint.lattice import random_rotation
from glint.multishot import same_lattice
from glint.stream_driver import StreamDriver
from glint.running_consensus import RunningConsensus
from glint.warmup_batch import warmup_consensus

SEED = 20260814
M_TRUE = np.diag([40.0, 55.0, 70.0])


def _still_q(M, rng, n=120, qmax=0.30):
    """A sparse still: n peaks scattered over the many nodes inside |q| <= qmax (see test_alias_gate)."""
    Minv = np.linalg.inv(M)
    rlen = np.sqrt((Minv ** 2).sum(1))
    hb = np.ceil(qmax / np.maximum(rlen, 1e-12)).astype(int)
    out = []
    while sum(len(o) for o in out) < n:
        H = np.column_stack([rng.integers(-hb[i], hb[i] + 1, 4 * n) for i in range(3)]).astype(float)
        q = H[np.abs(H).sum(1) > 0] @ Minv
        out.append(q[np.sqrt((q * q).sum(1)) <= qmax])
    q = np.vstack(out)[:n]
    return q + rng.normal(0.0, 0.0008, q.shape)


def _buf(M, rng, k=12, extra=None):
    """What the driver retains while blind: (q, [candidate cells]) per indexed frame. The frame's own
    N-best carries the lock's lattice in that frame's orientation, which is what the gate needs."""
    out = []
    for _ in range(k):
        Mi = random_rotation(rng) @ M
        cells = [Mi] if extra is None else [Mi, extra(Mi)]
        out.append((_still_q(Mi, rng), cells))
    return out


def _driver(gate, buf):
    """A StreamDriver with only the attributes _gate_lock reads."""
    d = object.__new__(StreamDriver)
    d._alias_gate = gate
    d._gate_buf = buf
    d._gate_last_support = -1
    d.n_gate_refused = 0
    return d


def test_gate_off_is_bit_identical():
    """Default alias_gate=None must not change the locked cell, and must retain no buffer."""
    rng = np.random.default_rng(SEED)
    d = _driver(None, None)
    out = d._gate_lock(M_TRUE, 12)
    assert out is M_TRUE, "gate off altered the lock"
    assert d.n_gate_refused == 0


def test_true_lock_is_confirmed():
    """The real case: consensus locks the true cell, the gate sees it per frame and confirms."""
    rng = np.random.default_rng(SEED + 1)
    d = _driver(AliasGate(), _buf(M_TRUE, rng, k=12))
    out = d._gate_lock(M_TRUE, 12)
    assert out is not None and same_lattice(out, M_TRUE), d._alias_gate.info
    assert d._alias_gate.info["verdict"] == "confirm", d._alias_gate.info
    assert d.n_gate_refused == 0


def test_supercell_lock_is_refused():
    """A doubled axis collects the same peaks and so wins the same votes -- the vote cannot tell it from
    the true cell. The gate is the complement that can, and refusing means NOT locking."""
    rng = np.random.default_rng(SEED + 2)
    H = hnf_matrices(2)[-1]
    buf = [(q, [c @ H for c in cs]) for q, cs in _buf(M_TRUE, rng, k=12)]
    d = _driver(AliasGate(), buf)
    assert d._gate_lock(M_TRUE @ H, 12) is None, "locked a super-cell alias"
    assert d.n_gate_refused == 1


def test_stats_reports_refusals_in_both_branches():
    """A refusal must be visible in stats() WHILE STILL BLIND -- that is the case it explains.

    `_gate_lock` runs only on the blind path, and a refusal is what leaves the driver blind, so the
    run states are not symmetric: every warm-up refusal is observed through the unlocked branch,
    and only a later watchdog relock is observed through the locked one. Reporting the counter on
    the locked branch alone (as glint#164 first did) therefore hid it at the exact moment an
    operator asks "there is consensus support but no cell -- why?" (Copilot review of glint#164).

    BOTH branches are exercised, literally per the name: the first version of this test called
    stats() only while blind, so deleting gate_refused from the LOCKED return -- the line the PR
    originally added -- left it green (Copilot review of glint#164, round 2). The locked driver
    below is initialised with exactly the attributes the locked branch reads, `acc` faked to the
    one method stats() calls on it.
    """
    d = _driver(AliasGate(), None)
    d._blind, d.n_pushed, d.n_warmup = True, 40, 31
    d.n_gate_refused = 3
    d._rc = RunningConsensus(min_support=3, gap=2, adaptive=False)
    s = d.stats()
    assert s["locked"] is False
    assert "gate_refused" in s, "a blind driver hides the refusals that are keeping it blind"
    assert s["gate_refused"] == 3, s

    d._blind = False                                    # ...and the locked branch, same counter
    d.acc = type("Acc", (), {"stats": staticmethod(lambda thr=0.0, n_theoretical=None: {})})()
    d.locked_after, d.consensus_support, d.n_theoretical = 6, 9, 4200
    d.laue = "4/mmm"                                    # the class the merge ran under (glint#180)
    d.n_indexed = d.n_integrated = 25
    d.n_gate_rejected = d.n_fanout_errors = d.n_fanout_missed = 0
    d.n_gate_deferred = 2
    d.warmup_rescue = d.retry_cascade = d.adaptive_relock = d.double_hit = False
    d.qc_frac_threshold = None
    d._writer = d._grefiner = None
    s = d.stats()
    assert s["locked"] is True
    assert s.get("gate_refused") == 3, (
        "the locked branch dropped the counter -- a post-lock watchdog refusal would be invisible")
    # The no-voter subset is its own key: the total's two sources cannot be told apart from the
    # neighbours (a healthy fan-out defers too), so the split is reported, not inferred. The
    # difference -- here 3 - 2 = 1 -- is the count of verdicts the gate itself scored and refused.
    assert s.get("gate_deferred_no_voters") == 2, (
        "the deferral subset is not reported -- refused-vs-deferred is uninferable without it")


def test_adopt_mode_returns_the_tighter_cell():
    """adopt=True: the gate knows which family member the frames prefer, so the driver locks THAT."""
    rng = np.random.default_rng(SEED + 3)
    H = hnf_matrices(2)[-1]
    buf = [(q, [c @ H for c in cs]) for q, cs in _buf(M_TRUE, rng, k=12)]
    d = _driver(AliasGate(adopt=True), buf)
    out = d._gate_lock(M_TRUE @ H, 12)
    assert out is not None and same_lattice(out, M_TRUE), d._alias_gate.info


def test_standing_refusal_costs_one_gate_call():
    """A refusal leaves the vote histogram accumulating, so the same verdict fires again every frame.
    Re-gating is skipped until support actually grows -- otherwise a refused stream pays the gate
    forever."""
    rng = np.random.default_rng(SEED + 4)
    H = hnf_matrices(2)[-1]
    buf = [(q, [c @ H for c in cs]) for q, cs in _buf(M_TRUE, rng, k=12)]
    d = _driver(AliasGate(), buf)
    assert d._gate_lock(M_TRUE @ H, 12) is None
    assert d._gate_lock(M_TRUE @ H, 12) is None, "same support should stay refused"
    assert d.n_gate_refused == 1, "re-ran the gate on unchanged votes"
    assert d._gate_lock(M_TRUE @ H, 13) is None, "more votes -> re-gated"
    assert d.n_gate_refused == 2


def test_no_agreeing_frame_leaves_the_vote_alone():
    """If no buffered frame indexed the locked lattice there is nothing to testify: the gate must not
    manufacture a refusal from an empty voter list."""
    rng = np.random.default_rng(SEED + 5)
    other = np.diag([31.0, 47.0, 83.0])
    d = _driver(AliasGate(), _buf(other, rng, k=6))
    out = d._gate_lock(M_TRUE, 6)
    assert out is M_TRUE and d.n_gate_refused == 0


def test_gate_picks_the_agreeing_hypothesis_not_the_top_one():
    """Frames carry several N-best candidates; the gate must be handed the one that agrees with the
    lock, in that frame's orientation -- not whatever sits first."""
    rng = np.random.default_rng(SEED + 6)
    decoy = np.diag([23.0, 29.0, 37.0])
    buf = [(q, [decoy @ random_rotation(rng)] + cs) for q, cs in _buf(M_TRUE, rng, k=12)]
    d = _driver(AliasGate(), buf)
    out = d._gate_lock(M_TRUE, 12)
    assert out is not None and same_lattice(out, M_TRUE), d._alias_gate.info
    assert d._alias_gate.info["frames_tested"] == 12, d._alias_gate.info


def test_warmup_consensus_sink_collects_per_frame_evidence():
    """The batched warm-up indexes the frames; the gate needs their per-frame N-best. `sink` hands it
    back instead of re-indexing, and must not disturb the vote."""
    rng = np.random.default_rng(SEED + 7)
    qs = [_still_q(random_rotation(rng) @ M_TRUE, rng) for _ in range(6)]
    cells = {id(q): [random_rotation(rng) @ M_TRUE] for q in qs}   # keyed by identity: q are arrays
    blind = lambda q, k: [(c, 1.0) for c in cells[id(q)]]

    rc_a = RunningConsensus(min_support=3, gap=2, adaptive=False)
    Mc_a, sup_a = warmup_consensus(qs, blind, rc_a, nbest=1)
    sink = []
    rc_b = RunningConsensus(min_support=3, gap=2, adaptive=False)
    Mc_b, sup_b = warmup_consensus(qs, blind, rc_b, nbest=1, sink=sink)

    assert sup_a == sup_b and (Mc_a is None) == (Mc_b is None), "sink changed the vote"
    assert len(sink) == len(qs)
    assert all(q is qq for (qq, _), q in zip(sink, qs)), "sink lost frame identity"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
    raise SystemExit(1 if fails else 0)
