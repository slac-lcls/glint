#!/usr/bin/env python3
"""StreamDriver effort= (adaptive effort): the known-cell search depth follows the hit rate, the spare budget buys
the chance-controlled deep search on the misses, decisions happen at flush boundaries only and are logged; off by
default, and off the driver makes exactly the calls it made before.

What is checked, on the numpy path with a fit oracle standing in for BOTH known-cell indexers (the fast path's
rgb.index_fused and the deep search's rgb.index_known_deep_batch -- test_cell_registry's seam, sd.rgb):

  * off by default: no policy object, no "effort" stats key, and the fast path is called WITHOUT depth arguments;
    every setting is validated and named in its error; the constructor option is appended (positional pin);
  * the depth follows the hit rate: a scripted stream at 100 % / 50 % / 25 % hits under a 4 kHz budget walks the
    tiers up 0 -> 1 -> 2 -> 3, every fast-path call carries the tier in force at its flush, the deep search is
    switched on only below the top tier and only once the miss fraction is known, and each change is one
    "effort" event with the numbers behind it;
  * the deep search on the misses: a frame the fast path refuses is recovered by escalate_batch at the top tier
    (one real search plus k_null scrambled-copy searches, all logged), integrated with outcome "escalated" under
    the cell that accepted it; when the deep search finds nothing the frame fails closed down the old miss path;
  * ordering: the deep search runs BEFORE the miss buffer and the blind watchdog, so a recovered frame is
    neither buffered nor blind-indexed (the control without effort= shows the watchdog taking that same frame);
  * `every` spaces the decisions in flushes.

    PYTHONPATH=. python experiments/test_stream_effort.py
"""
import inspect
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import glint.stream_driver as sd                                             # noqa: E402
from glint.multishot import same_lattice                                      # noqa: E402
from glint.stream_driver import StreamDriver, EFFORT_TIERS                    # noqa: E402
from test_cell_registry import A, B, SEED, _driver, _key, _terminal, frame_on  # noqa: E402


class _EffortOracle:
    """Known-cell indexers that decide from the peaks: a frame is on lattice L when its q sit near-integer under L.
    `refuse` = (fast-path call number, position) pairs the fast path returns None for -- a miss the deep search
    may recover; `deep_refuse` = content keys the deep search also misses (fail-closed check). Every fast-path
    call logs its batch size and the depth kwargs it was given; every deep call logs size and depth."""

    def __init__(self, lattices=(A, B), refuse=(), deep_refuse=()):
        self.lattices = [np.asarray(L, float) for L in lattices]
        self.refuse = set(refuse)
        self.deep_refuse = set(deep_refuse)
        self.calls = []                  # (n frames, {depth kwargs}) per fast-path call
        self.deep_calls = []             # (n frames, topa, nc, full_grid) per deep-search call
        self.blind_calls = 0

    def _on(self, q):
        q = np.asarray(q, float)
        if not len(q):
            return None
        for L in self.lattices:
            hf = q @ L
            if (np.abs(hf - np.round(hf)).max(1) < sd.HKL_TOL).mean() >= 0.5:
                return L
        return None

    def index_fused(self, qs, Mc, B=1, **depth):
        c = len(self.calls)
        self.calls.append((len(qs), dict(depth)))
        out = []
        for j, q in enumerate(qs):
            L = self._on(q)
            ok = L is not None and same_lattice(L, Mc) and (c, j) not in self.refuse
            out.append(L.copy() if ok else None)
        return out

    def index_known_deep_batch(self, frames, Mc, topa, nc, full_grid=True, budget=12000, return_errors=False):
        self.deep_calls.append((len(frames), topa, nc, full_grid))
        out = []
        for q in frames:
            L = self._on(q)
            ok = L is not None and same_lattice(L, Mc) and _key(q) not in self.deep_refuse
            out.append(L.copy() if ok else None)
        return (out, [False] * len(out)) if return_errors else out

    def blind(self, q, k):
        self.blind_calls += 1
        L = self._on(q)
        return [(L.copy(), 1.0)] if L is not None else []


BLANK = np.zeros((0, 3))


def _drv(oracle=None, **kw):
    oracle = oracle or _EffortOracle()
    kw.setdefault("B", 4)
    kw.setdefault("hits_only", True)         # B counts hits, so a flush is every 4 hits whatever the blank rate
    kw.setdefault("events", True)
    return _driver(Mc=A, oracle=oracle, **kw)


def _run(drv, plan, rng):
    """plan: 'H' = a frame on lattice A, 'b' = a blank (no usable peaks)."""
    for c in plan:
        drv.push_q(frame_on(A, rng) if c == "H" else BLANK)
    drv.close()


def _effort_events(drv):
    return [e for e in drv.events if e["outcome"] == "effort"]


# --------------------------------------------------------------- off by default, validation --------
def test_off_by_default_and_options_validated():
    d, o = _drv()
    assert d._eff is None and "effort" not in d.stats()
    _run(d, "HHHHHHHH", np.random.default_rng(SEED))
    assert len(o.calls) == 2 and all(kw == {} for _, kw in o.calls), o.calls   # the shipped call, verbatim
    assert "effort" not in d.stats() and not _effort_events(d)
    params = inspect.signature(StreamDriver.__init__).parameters
    assert list(params)[-2:] == ["effort", "null_floor"] and params["effort"].default is None
    assert params["null_floor"].default is None
    bad = [(5, "must be a dict"), ({}, "rate_hz"), (dict(rate_hz=0), "rate_hz"), (dict(rate_hz=-2.0), "rate_hz"),
           (dict(rate_hz=1e3, n_gpu=0), "n_gpu"), (dict(rate_hz=1e3, k_null=0), "k_null"),
           (dict(rate_hz=1e3, k_null=True), "k_null"), (dict(rate_hz=1e3, k_null=1.5), "k_null"),
           (dict(rate_hz=1e3, every=0), "every"), (dict(rate_hz=1e3, hit_window="8"), "hit_window"),
           (dict(rate_hz=1e3, hit_prior=0), "hit_prior"), (dict(rate_hz=1e3, hit_prior=1.5), "hit_prior"),
           (dict(rate_hz=1e3, overhead_ms=-0.1), "overhead_ms"), (dict(rate_hz=1e3, overhead_ms="0.3"), "overhead_ms"),
           (dict(rate_hz=1e3, bogus=1), "unknown"),
           (dict(rate_hz=1e3, tiers=()), "tiers"),
           (dict(rate_hz=1e3, tiers=((8, 16, False, 0.3), (32, 32, False, 0.2))), "increasing"),
           (dict(rate_hz=1e3, tiers=((0, 16, False, 0.3),)), "topa"),
           (dict(rate_hz=1e3, tiers=((8, 16, False),)), "tiers")]
    for cfg, word in bad:
        try:
            _drv(effort=cfg)
        except ValueError as exc:
            assert word in str(exc), (cfg, str(exc))
        else:
            raise AssertionError(f"effort={cfg!r} accepted")
    d, _ = _drv(effort=dict(rate_hz=2000))
    e = d._eff
    assert (e.rate_hz, e.n_gpu, e.k_null, e.round_copies, e.every, e.hit_window, e.miss_window, e.hit_prior,
            e.overhead_ms) == (2000.0, 1.0, 32, 8, 1, 256, 4, 1.0, 0.0)
    assert e.tiers == EFFORT_TIERS and e.tier is None and e.deep is False   # nothing decided before the first flush
    assert d.stats()["effort"]["tier"] is None


# --------------------------------------------------------------- the depth follows the hit rate ----
def test_depth_follows_the_hit_rate_and_changes_are_logged():
    # 4 kHz on one GPU: 100 % hits leave 0.25 ms per hit (tier 0), 75 % 0.33 ms (tier 1), 50 % 0.5 ms (tier 2),
    # 25 % 1.0 ms (tier 3, the top). hit_window=16 so the estimate turns over within a phase.
    d, o = _drv(effort=dict(rate_hz=4000, n_gpu=1, hit_window=16))
    rng = np.random.default_rng(SEED)
    _run(d, "H" * 16 + "Hb" * 16 + "Hbbb" * 16, rng)
    eff = d.stats()["effort"]
    log = eff["log"]
    # first decision, before any history: the conservative prior -> the shipped depth, no deep search
    assert log[0] == dict(at=0, tier=0, deep=False, budget_ms=0.25, hit_est=1.0, miss_frac=1.0, n_cells=1), log[0]
    # one flush later the miss fraction is known (0): the budget also covers deep searches at tier 0
    assert log[1]["at"] == 4 and (log[1]["tier"], log[1]["deep"]) == (0, True), log[1]
    tiers = [e["tier"] for e in log]
    assert tiers == sorted(tiers), tiers                       # the hit rate only falls: the depth only climbs
    assert {1, 2, 3} <= set(tiers), tiers                      # every tier is visited on the way up
    assert (eff["tier"], eff["deep"]) == (3, False)            # at the top tier the deep search is off by rule
    assert all(not e["deep"] for e in log if e["tier"] == 3)
    assert all(e["deep"] for e in log if e["tier"] == 2)       # 0.40 + 0 * ... <= 0.5: on, no misses to spend it on
    # every fast-path call carried the tier in force at its flush -- 12 flushes of 4 hits, 48 hits in all
    assert len(o.calls) == 12 and all(n == 4 for n, _ in o.calls)
    assert all(set(kw) == {"topa", "nc", "full_grid"} for _, kw in o.calls)
    seen = [tuple(kw[k] for k in ("topa", "nc", "full_grid")) for _, kw in o.calls]
    assert set(seen) == {t[:3] for t in EFFORT_TIERS}, seen
    assert sum(eff["fast_by_tier"].values()) == 48 and sorted(eff["fast_by_tier"]) == [0, 1, 2, 3]
    assert o.deep_calls == [] and eff["n_deep_tried"] == eff["n_escalated"] == eff["deep_searches"] == 0
    # each change is one "effort" event with the same numbers, at the flush that made it
    ev = _effort_events(d)
    assert len(ev) == len(log) == eff["n_changes"]
    for e, l in zip(ev, log):                                # ev = the first frame the decision applied to;
        assert e["ev"] == l["at"] <= e["at"]                 # at = n_pushed when it was emitted, as for every event
        assert (e["tier"], e["deep"], e["budget_ms"], e["hit_est"], e["miss_frac"], e["n_cells"]) == \
            (l["tier"], l["deep"], l["budget_ms"], l["hit_est"], l["miss_frac"], l["n_cells"])
    assert d.n_indexed == 48 and d.n_gate_rejected == 0
    # the blanks were counted (48 hits of 112 pushes), the terminal outcomes are one per frame
    term = _terminal(d.events)
    assert len(term) == 112 and sum(t["outcome"] == "blank" for t in term.values()) == 64


# --------------------------------------------------------------- the deep search on the misses -----
DEEP = dict(rate_hz=1000, n_gpu=0.9, k_null=2, round_copies=2, hit_window=16,
            tiers=((8, 16, False, 0.10), (32, 32, False, 0.20), (128, 32, True, 1.0)))
# 0.9 ms per hit at 100 % hits: tier 1 (0.20 <= 0.9 < 1.0); the deep search fits once
# 0.20 + miss_frac * (1 + 2) * 1.0 <= 0.9, i.e. once the miss fraction is known to be <= 0.23.


def test_deep_search_recovers_a_refused_frame_and_logs_its_searches():
    o = _EffortOracle(refuse=((1, 2),))                        # fast-path call 1 (the second flush), position 2 = ev 6
    d, _ = _drv(oracle=o, effort=DEEP)
    _run(d, "H" * 16, np.random.default_rng(SEED))
    eff = d.stats()["effort"]
    assert [(e["at"], e["tier"], e["deep"]) for e in eff["log"]] == [(0, 1, False), (4, 1, True)], eff["log"]
    term = _terminal(d.events)
    assert term[6]["outcome"] == "escalated" and term[6]["cell"] == 0 and term[6]["n_inl"] == 40, term[6]
    assert sum(t["outcome"] == "indexed" for t in term.values()) == 15 and d.n_indexed == 16 and d.n_gate_rejected == 0
    # one real search at the top tier, then its k_null = 2 scrambled copies in one round of round_copies = 2
    assert o.deep_calls == [(1, 128, 32, True), (2, 128, 32, True)], o.deep_calls
    assert (eff["n_deep_tried"], eff["n_escalated"], eff["deep_searches"]) == (1, 1, 3)
    assert eff["miss_frac"] == 1 / 12                           # (0 + 1 + 0) misses over the last three flushes
    assert len(o.calls) == 4 and all(kw["topa"] == 32 for _, kw in o.calls)   # the fast path stayed at tier 1


def test_deep_search_fails_closed_when_it_finds_nothing():
    rng = np.random.default_rng(SEED)
    frames = [frame_on(A, rng) for _ in range(16)]
    o = _EffortOracle(refuse=((1, 2),), deep_refuse={_key(frames[6])})
    d, _ = _drv(oracle=o, effort=DEEP)
    for q in frames:
        d.push_q(q)
    d.close()
    term = _terminal(d.events)
    assert term[6]["outcome"] == "gate_rejected" and d.n_gate_rejected == 1 and d.n_indexed == 15
    eff = d.stats()["effort"]
    assert o.deep_calls == [(1, 128, 32, True)]                 # no fit -> no copies searched
    assert (eff["n_deep_tried"], eff["n_escalated"], eff["deep_searches"]) == (1, 0, 1)


def test_deep_search_runs_before_the_miss_buffer_and_the_watchdog():
    rng = np.random.default_rng(SEED)
    frames = [frame_on(A, rng) for _ in range(16)]
    on = _EffortOracle(refuse=((1, 2),))
    d, _ = _drv(oracle=on, effort=DEEP, adaptive_relock=True, rescue_buffer=8)
    for q in frames:
        d.push_q(q)
    d.close()
    term = _terminal(d.events)
    assert term[6]["outcome"] == "escalated" and on.blind_calls == 0 and len(d._missbuf) == 0
    assert d.stats()["n_watchdog_rescued"] == 0 and d.n_indexed == 16
    # the control: without effort= that same miss goes to the miss buffer and the blind watchdog rescues it
    off = _EffortOracle(refuse=((1, 2),))
    c, _ = _drv(oracle=off, adaptive_relock=True, rescue_buffer=8)
    for q in frames:
        c.push_q(q)
    c.close()
    term = _terminal(c.events)
    assert term[6]["outcome"] == "rescued_watchdog" and off.blind_calls == 1 and len(c._missbuf) == 1
    assert c.stats()["n_watchdog_rescued"] == 1 and off.deep_calls == [] and "effort" not in c.stats()


# --------------------------------------------------------------- the overhead comes off the budget --
def test_overhead_ms_is_taken_off_the_budget_before_the_tier_is_chosen():
    # 2 kHz, all hits: 0.5 ms per hit buys tier 2 (0.40) when the hit is search only; with 0.2 ms of integration per
    # hit the search has 0.3 ms, which buys tier 1 (0.27); with 0.5 ms of overhead nothing is left and the policy
    # falls back to tier 0 with the deep search off.
    for oh, tier, budget in ((0.0, 2, 0.5), (0.2, 1, 0.3), (0.5, 0, 0.0)):
        d, o = _drv(effort=dict(rate_hz=2000, hit_window=16, overhead_ms=oh))
        _run(d, "H" * 8, np.random.default_rng(SEED))
        eff = d.stats()["effort"]
        assert eff["tier"] == tier and abs(eff["budget_ms"] - budget) < 1e-9, (oh, eff["tier"], eff["budget_ms"])
        assert eff["log"][0]["budget_ms"] == round(budget, 4)
        assert all(kw["topa"] == EFFORT_TIERS[tier][0] and kw["nc"] == EFFORT_TIERS[tier][1] for _, kw in o.calls)
    assert eff["deep"] is False and d.n_indexed == 8


# --------------------------------------------------------------- cold start and active cells ------
def test_first_decision_uses_the_prior_then_the_window():
    # a stream that opens at 25 % hits under a 4 kHz budget: the window alone would buy tier 3 (1.0 ms per hit) for
    # the very first known-cell call; the first decision uses hit_prior=1 instead (0.25 ms -> tier 0) and the second
    # decision reads the window (hit 0.25 -> tier 3)
    d, o = _drv(effort=dict(rate_hz=4000, hit_window=16))
    _run(d, "Hbbb" * 8, np.random.default_rng(SEED))         # 8 hits: two flushes of 4
    log = d.stats()["effort"]["log"]
    assert [(e["at"], e["tier"], e["hit_est"]) for e in log] == [(0, 0, 1.0), (16, 3, 0.25)], log
    assert [kw["topa"] for _, kw in o.calls] == [8, 128]     # first flush at the shipped depth, second at the top tier
    # with a prior that says 10 % hits the first flush already goes deep
    d, o = _drv(effort=dict(rate_hz=4000, hit_window=16, hit_prior=0.1))
    _run(d, "Hbbb" * 8, np.random.default_rng(SEED))
    assert d.stats()["effort"]["log"][0]["tier"] == 3 and o.calls[0][1]["topa"] == 128


def test_active_cells_multiply_the_cost_before_the_budget_test():
    pol = sd._EffortPolicy(dict(rate_hz=2000))               # 0.5 ms per hit at 100 % hits
    pol.n_flush = 2; pol.hits.extend([1] * 8); pol.flushes.append((8, 0))
    assert pol.decide(0, n_cells=1) and (pol.tier, pol.deep) == (2, True)      # 0.40 <= 0.5, deep with no misses
    assert pol.decide(8, n_cells=2) and (pol.tier, pol.deep) == (0, True)      # 2 x 0.27 > 0.5, 2 x 0.15 fits
    assert pol.decide(16, n_cells=4) and (pol.tier, pol.deep) == (0, False)    # 4 x 0.15 = 0.6 > 0.5: tier 0 anyway, no deep
    assert [e["n_cells"] for e in pol.log] == [1, 2, 4]
    # the driver passes its active-cell count: one cell in single-cell mode
    d, _ = _drv(effort=dict(rate_hz=2000, hit_window=16))
    _run(d, "H" * 8, np.random.default_rng(SEED))
    assert all(e["n_cells"] == 1 for e in d.stats()["effort"]["log"])


# --------------------------------------------------------------- decision spacing ------------------
def test_every_spaces_the_decisions_in_flushes():
    d, o = _drv(effort=dict(rate_hz=4000, hit_window=16, every=2))
    _run(d, "H" * 24, np.random.default_rng(SEED))            # 6 flushes: decisions at flushes 1, 3, 5
    log = d.stats()["effort"]["log"]
    assert [(e["at"], e["tier"], e["deep"]) for e in log] == [(0, 0, False), (8, 0, True)], log
    assert len(_effort_events(d)) == 2 and len(o.calls) == 6


TESTS = [test_off_by_default_and_options_validated,
         test_depth_follows_the_hit_rate_and_changes_are_logged,
         test_deep_search_recovers_a_refused_frame_and_logs_its_searches,
         test_deep_search_fails_closed_when_it_finds_nothing,
         test_deep_search_runs_before_the_miss_buffer_and_the_watchdog,
         test_overhead_ms_is_taken_off_the_budget_before_the_tier_is_chosen,
         test_first_decision_uses_the_prior_then_the_window,
         test_active_cells_multiply_the_cost_before_the_budget_test,
         test_every_spaces_the_decisions_in_flushes]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:                                        # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(exc).__name__}({exc})")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
