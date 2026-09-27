"""CPU tests for the opt-in escalation: glint.retry_cascade.arm_known_deep and hybrid_index(escalate=...).

What is under test is the ACCEPTANCE LOGIC, not the GPU search: a deep known-cell fit is kept only if
no azimuth-scrambled copy of the same frame matches as many peaks, the null stops at the first copy
that does, a fit that fails the gate costs one search, copy k is seeded [*seed, k] (the seeding of the
experiment that measured it, so its accepts reproduce), and with escalate=None hybrid_index is
untouched; a bad escalate setting raises before any search, and a control search that raises rejects the
fit; escalated frames are marked in both stream writers. The indexers are scripted fakes; glint.glint_fast and glint.replica_gpu (torch at import)
are stubbed for the hybrid_index half, the same way experiments/test_lock_gate_wiring.py does it.

Convention: a cell matrix M has COLUMNS = real-space axes in Angstrom and q @ M = hkl.

  PYTHONPATH=. python experiments/test_escalation.py      # exit 0 = all pass
"""
import importlib
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint.lattice import cell_to_Ar                                    # noqa: E402
from glint.retry_cascade import (DEEP_K_NULL, DEEP_NC, DEEP_ROUND_COPIES, DEEP_SEED, DEEP_TOPA,  # noqa: E402
                                 arm_known_deep, escalate_batch)

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


def rot(a, b, c):
    def R(t, ax):
        s, k = np.sin(t), np.cos(t)
        m = np.eye(3); i, j = [x for x in range(3) if x != ax]
        m[i, i] = m[j, j] = k; m[i, j] = -s; m[j, i] = s
        return m
    return R(a, 0) @ R(b, 1) @ R(c, 2)


def count_strict(M, q):
    """The strict matcher (glint_fast.matched_strict at GATE_TOL = 0.15), numpy only."""
    if M is None:
        return 0
    r = q @ M - np.rint(q @ M)
    return int((np.abs(r).max(1) < 0.15).sum())


# ------------------------------------------------------------------------------------------------
# Part 1: the arm, with a scripted matcher (the i-th call to count returns script[i])
# ------------------------------------------------------------------------------------------------
class Scripted:
    def __init__(self, counts, fit=True):
        self.counts = list(counts); self.calls = []; self.fit = fit

    def index(self, q, Mc, topa=None, nc=None):
        self.calls.append(dict(topa=topa, nc=nc, n=len(q)))
        return np.eye(3) if self.fit else None

    def count(self, M, q):
        return self.counts.pop(0)


q0 = np.random.default_rng(1).normal(size=(40, 3)) * 0.05
yes = lambda M, q: True                                                  # noqa: E731

print("arm_known_deep: accepted only when no scrambled copy matches as many peaks")
s = Scripted([30] + [5] * 32)
M, rec = arm_known_deep(q0, np.eye(3), s.index, s.count, yes, seed=[DEEP_SEED, 0])
check("fit beating all 32 copies is accepted", M is not None and rec["accepted"], rec)
check("p = 1/33 and 33 searches", rec["p"] == 1 / 33 and rec["searches"] == 33 and len(rec["null_m"]) == 32, rec)
check("the search depth reaches the indexer (topa 128, nc 32 by default)",
      all(c["topa"] == DEEP_TOPA and c["nc"] == DEEP_NC for c in s.calls), s.calls[:2])

s = Scripted([30, 5, 5, 31] + [5] * 29)
M, rec = arm_known_deep(q0, np.eye(3), s.index, s.count, yes, seed=[DEEP_SEED, 0])
check("the null stops at the first copy that beats the fit (4 searches)",
      M is None and rec["searches"] == 4 and rec["null_m"] == [5, 5, 31], rec)

s = Scripted([30, 30] + [5] * 31)
M, rec = arm_known_deep(q0, np.eye(3), s.index, s.count, yes, seed=[DEEP_SEED, 0])
check("a tie rejects (a copy matching as many peaks is not evidence)", M is None and rec["searches"] == 2, rec)

s = Scripted([30])
M, rec = arm_known_deep(q0, np.eye(3), s.index, s.count, lambda M, q: False, seed=[DEEP_SEED, 0])
check("a fit that fails the gate costs one search and no null", M is None and rec["searches"] == 1
      and rec["null_m"] == [] and rec["m"] == 30, rec)

s = Scripted([], fit=False)
M, rec = arm_known_deep(q0, np.eye(3), s.index, s.count, yes, seed=[DEEP_SEED, 0])
check("no fit at all: one search", M is None and rec["searches"] == 1, rec)

s = Scripted([30] + [5] * 8)
M, rec = arm_known_deep(q0, np.eye(3), s.index, s.count, yes, seed=[DEEP_SEED, 0], k_null=8, topa=32, nc=16)
check("k_null / topa / nc overrides", M is not None and rec["p"] == 1 / 9 and rec["searches"] == 9
      and s.calls[0]["topa"] == 32 and s.calls[0]["nc"] == 16, rec)

print("\narm_known_deep: a control that cannot be evaluated rejects the fit (fail closed)")


class Flaky(Scripted):
    """The real search succeeds; the scrambled copy number `bad` raises (a transient GPU/indexer failure)."""
    def __init__(self, counts, bad):
        super().__init__(counts); self.bad = bad

    def index(self, q, Mc, topa=None, nc=None):
        if len(self.calls) == 1 + self.bad:
            self.calls.append(dict(topa=topa, nc=nc, n=len(q)))
            raise RuntimeError("CUDA out of memory")
        return super().index(q, Mc, topa, nc)


s = Flaky([30] + [5] * 32, bad=3)
M, rec = arm_known_deep(q0, np.eye(3), s.index, s.count, yes, seed=[DEEP_SEED, 0])
check("a copy whose search raises rejects the fit (not scored as 0 matched)",
      M is None and not rec["accepted"] and rec.get("null_error") is True, rec)
check("  and stops there: 5 searches, 3 copies scored", rec["searches"] == 5 and rec["null_m"] == [5, 5, 5], rec)
for bad in (0, -1, True, 2.0, None):
    try:
        arm_known_deep(q0, np.eye(3), Scripted([30] + [5] * 32).index, lambda M, q: 30, yes,
                       seed=[DEEP_SEED, 0], k_null=bad)
        check(f"k_null={bad!r} raises (no null = no acceptance rule)", False)
    except ValueError:
        check(f"k_null={bad!r} raises (no null = no acceptance rule)", True)

print("\narm_known_deep: copy k is scrambled with default_rng([*seed, k]) -- the experiment's seeding")
seen = []


def spy_scramble(q, rng):
    seen.append(rng.uniform())
    return q


s = Scripted([30] + [5] * 32)
arm_known_deep(q0, np.eye(3), s.index, s.count, yes, seed=[DEEP_SEED, 7], scramble=spy_scramble)
want = [np.random.default_rng([DEEP_SEED, 7, k]).uniform() for k in range(DEEP_K_NULL)]
check("32 copies, seeded [20260926, frame, k]", seen == want, (seen[:2], want[:2]))

print("\narm_known_deep: the sequential decision equals the full-null decision (500 random scripts)")
r = np.random.default_rng(3); agree = 0
for _ in range(500):
    m = int(r.integers(5, 40)); null = [int(x) for x in r.integers(0, 45, 32)]
    s = Scripted([m] + null)
    M, rec = arm_known_deep(q0, np.eye(3), s.index, s.count, yes, seed=[DEEP_SEED, 0])
    agree += (M is not None) == (m > max(null))
check("500/500 agree with 'm > max of all 32' (escalate.py's rule)", agree == 500, agree)

# ------------------------------------------------------------------------------------------------
# Part 1b: escalate_batch -- the same rule, batched in rounds of copies (scripted engine)
# ------------------------------------------------------------------------------------------------
class BatchWorld:
    """Real frame i is q with q[0] = (i, 0, 0); its fit matches m[i] peaks (None if i in nofit) and passes the
    gate iff gate[i]; copy k of frame i matches null[i][k]. The scramble tags copies (i, k, 1) in row 0 and
    checks it was handed default_rng([*seeds[i], k]). err: (i, k) copies whose search errors; raise_call: the
    search call (1-based) that raises outright."""

    def __init__(self, m, null, gate, seeds, nofit=(), err=(), raise_call=None):
        self.m, self.null, self.g, self.seeds = m, null, gate, seeds
        self.nofit, self.err, self.raise_call = set(nofit), set(err), raise_call
        self.calls = 0; self.k = {}; self.rng_ok = True; self.batch_sizes = []

    def frames(self):
        return [np.vstack([[i, 0.0, 0.0], q0[:5]]) for i in range(len(self.m))]

    def scramble(self, q, rng):
        i = int(q[0, 0]); k = self.k.get(i, 0); self.k[i] = k + 1
        want = np.random.default_rng([*self.seeds[i], k]).integers(1 << 62)
        self.rng_ok &= bool(rng.integers(1 << 62) == want)
        return np.vstack([[i, k, 1.0], q[1:]])

    def search(self, qs, Mc):
        self.calls += 1; self.batch_sizes.append(len(qs))
        if self.raise_call == self.calls:
            raise RuntimeError("scripted batch failure")
        res, err = [], []
        for q in qs:
            i, k, cp = int(q[0, 0]), int(q[0, 1]), q[0, 2] == 1.0
            res.append(None if (not cp and i in self.nofit) else np.eye(3))
            err.append(cp and (i, k) in self.err)
        return res, err

    def count(self, M, q):
        i, k, cp = int(q[0, 0]), int(q[0, 1]), q[0, 2] == 1.0
        return self.null[i][k] if cp else self.m[i]

    def gate(self, M, q):
        return bool(self.g[int(q[0, 0])])


def run_batch(w, k_null=DEEP_K_NULL, rc=DEEP_ROUND_COPIES):
    return escalate_batch(w.frames(), np.eye(3), w.search, w.count, w.gate, w.seeds, k_null=k_null,
                          round_copies=rc, scramble=w.scramble)


print("\nescalate_batch: the accept set equals the rule and the sequential arm (400 random multi-frame scripts)")
r = np.random.default_rng(5); agree = seq_agree = prefix_ok = cost_ok = rng_ok = 0; N = 400
for t in range(N):
    n = int(r.integers(1, 7)); rc = int(r.choice([1, 2, 3, 8, 32, 40]))
    m = [int(x) for x in r.integers(5, 40, n)]
    null = [[int(x) for x in r.integers(0, 45, 32)] for _ in range(n)]
    gate = [bool(x) for x in r.random(n) < 0.8]
    nofit = {i for i in range(n) if r.random() < 0.1}
    seeds = [[DEEP_SEED, int(i)] for i in r.permutation(1000)[:n]]
    w = BatchWorld(m, null, gate, seeds, nofit=nofit)
    Ms, recs = run_batch(w, rc=rc)
    want = [i not in nofit and gate[i] and m[i] > max(null[i]) for i in range(n)]
    agree += [M is not None for M in Ms] == want
    ok_seq = ok_pre = ok_cost = True
    for i in range(n):                                        # the per-frame arm on the same script
        sq = Scripted(([m[i]] + null[i]) if i not in nofit else [], fit=i not in nofit)
        Mi, ri = arm_known_deep(q0, np.eye(3), sq.index, sq.count, lambda M, q, g=gate[i]: g,
                                seed=seeds[i], scramble=lambda q, rng: q)
        ok_seq &= (Mi is not None) == (Ms[i] is not None)
        ok_pre &= recs[i]["null_m"][:len(ri["null_m"])] == ri["null_m"]
        extra = recs[i]["searches"] - ri["searches"]
        ok_cost &= extra == 0 if (Ms[i] is not None or not ri["null_m"]) else 0 <= extra <= rc - 1
    seq_agree += ok_seq; prefix_ok += ok_pre; cost_ok += ok_cost; rng_ok += w.rng_ok
check(f"{N}/{N} accept sets equal 'gate and m > max of all 32' (rounds of 1..40 copies)", agree == N, agree)
check(f"{N}/{N} equal arm_known_deep's per-frame decisions", seq_agree == N, seq_agree)
check(f"{N}/{N}: each frame's null_m starts with the sequential null's counts", prefix_ok == N, prefix_ok)
check(f"{N}/{N}: extra searches only on null-rejected frames, at most round_copies - 1", cost_ok == N, cost_ok)
check(f"{N}/{N}: copy k of frame i scrambled with default_rng([*seeds[i], k])", rng_ok == N, rng_ok)

print("\nescalate_batch: rounds, cost and fail-closed controls")
w = BatchWorld([30, 30, 30], [[5] * 32, [5] * 32, [5] * 3 + [30] + [5] * 28], [True] * 3, [[1], [2], [3]])
Ms, recs = run_batch(w)
check("32 copies in rounds of 8: 1 real batch + 4 copy batches", w.calls == 5 and w.batch_sizes[:2] == [3, 24],
      (w.calls, w.batch_sizes))
check("frame 2 (copy 3 ties) rejected after its first round, others accepted",
      [M is not None for M in Ms] == [True, True, False] and recs[2]["searches"] == 9, [r["searches"] for r in recs])
check("accepted records carry p = 1/33 and all 32 null counts",
      all(abs(recs[i]["p"] - 1 / 33) < 1e-12 and len(recs[i]["null_m"]) == 32 for i in (0, 1)))
w = BatchWorld([30, 30], [[5] * 32, [5] * 32], [True, True], [[1], [2]], err={(0, 5)})
Ms, recs = run_batch(w)
check("a copy whose search errors rejects that frame only (null_error), fail closed",
      Ms[0] is None and recs[0].get("null_error") and Ms[1] is not None and "null_error" not in recs[1],
      [(M is not None, r.get("null_error")) for M, r in zip(Ms, recs)])
w = BatchWorld([30, 30], [[5] * 32, [5] * 32], [True, True], [[1], [2]], raise_call=2)
Ms, recs = run_batch(w)
check("a copy batch that raises rejects every frame in it (null_error)",
      Ms == [None, None] and all(r.get("null_error") for r in recs), [r.get("null_error") for r in recs])
w = BatchWorld([30, 30], [[5] * 32, [5] * 32], [True, True], [[1], [2]], raise_call=1)
Ms, recs = run_batch(w)
check("a real-search batch that raises: every frame a miss, no null run",
      Ms == [None, None] and all(r["searches"] == 1 and not r["null_m"] for r in recs))
w = BatchWorld([30], [[5] * 32], [False], [[1]])
Ms, recs = run_batch(w)
check("a fit failing the gate costs one search and no null", Ms == [None] and recs[0]["searches"] == 1 and w.calls == 1)
check("no frames: no search", escalate_batch([], np.eye(3), w.search, w.count, w.gate, []) == ([], []))
for kw in (dict(k_null=0), dict(k_null=True), dict(round_copies=0), dict(round_copies=2.5)):
    try:
        escalate_batch(w.frames(), np.eye(3), w.search, w.count, w.gate, w.seeds, **kw)
        check(f"escalate_batch({kw}) raises ValueError", False)
    except ValueError:
        check(f"escalate_batch({kw}) raises ValueError", True)
try:
    escalate_batch(w.frames(), np.eye(3), w.search, w.count, w.gate, [])
    check("seeds must match the frames", False)
except ValueError:
    check("seeds must match the frames", True)

# ------------------------------------------------------------------------------------------------
# Part 2: hybrid_index(escalate=...), with glint_fast / replica_gpu stubbed
# ------------------------------------------------------------------------------------------------
print("\nhybrid_index: escalation runs only when asked, only on gate-failing frames, and labels what it adds")
CELL = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
JUNK = rot(2.71, 1.34, 0.19) @ cell_to_Ar(53.7, 61.3, 71.9, 90, 90, 90)
TIE = np.full((3, 3), 7.0)                           # marker: "this copy matches as many peaks as the fit"


def frame_on(M, rng, n=40):
    hkl = rng.integers(-5, 6, size=(4 * n, 3)).astype(float)
    hkl = hkl[np.abs(hkl).sum(1) > 0][:n]
    return hkl @ np.linalg.inv(M)


rng = np.random.default_rng(11)
TRUE = [rot(*rng.uniform(0, 3, 3)) @ CELL for _ in range(12)]
FRAMES = [frame_on(Mt, rng) for Mt in TRUE]
EASY = set(range(8))            # the blind pass finds these (they vote)
DEEP = {8, 9}                   # the shipped rescue misses these; the deep search finds them
TIED = {10}                     # the deep search fits it, but a scrambled copy does as well
POOR = {11}                     # the shipped rescue returns a poor fit (fails the gate); deep finds it


class World:
    def __init__(self):
        self.calls = []; self.current = None

    def nbest(self, q, k):
        i = next((j for j, f in enumerate(FRAMES) if f is q), None)
        return [(TRUE[i], 1.0)] if i in EASY else []

    def known(self, q, Mc, topa=8, nc=None):
        i = next((j for j, f in enumerate(FRAMES) if f is q), None)
        self.calls.append((i, topa, nc))
        deep = topa > 8
        if i is not None:
            self.current = i
            if i in POOR and not deep:
                return rot(0.09, 0.0, 0.0) @ TRUE[i]          # right cell, ~5 deg off: kept, fails the gate
            if deep and i in DEEP | TIED | POOR:
                return TRUE[i]
            return None
        # a scrambled copy: find its frame by |q| (the scramble keeps every |q|), so copies searched in a
        # batch, after all the real frames, are still told apart
        nq = np.sort(np.linalg.norm(q, axis=1))
        src = next((j for j, f in enumerate(FRAMES) if len(f) == len(q) and
                    np.allclose(np.sort(np.linalg.norm(f, axis=1)), nq)), self.current)
        return TIE if src in TIED else JUNK

    def matched(self, M, q):
        return 10 ** 6 if np.array_equal(np.asarray(M), TIE) else count_strict(M, q)


def load_hybrid(world):
    """Import glint.hybrid_stream against stub glint_fast / replica_gpu; return (module, restore)."""
    saved = {k: sys.modules.get(k) for k in ("glint.glint_fast", "glint.replica_gpu", "glint.replica_gpu_batch",
                                              "glint.hybrid_stream")}
    gf = types.ModuleType("glint.glint_fast")
    gf.GATE_FRAC, gf.GATE_MIN = 0.25, 10
    gf.index_blind_fast = lambda q: None
    gf.index_blind_nbest = world.nbest
    gf.load = lambda p: []
    gf.matched_strict = world.matched
    rg = types.ModuleType("glint.replica_gpu")
    rg.index_known_gpu_cell = world.known
    rgb = types.ModuleType("glint.replica_gpu_batch")          # the batched deep search, on the active world
    rgb.world = world

    def index_known_deep_batch(qs, Mc, topa, nc, full_grid=True, budget=12000, return_errors=False):
        res = [rgb.world.known(q, Mc, topa=topa, nc=nc) for q in qs]
        return (res, [False] * len(res)) if return_errors else res
    rgb.index_known_deep_batch = index_known_deep_batch
    sys.modules["glint.glint_fast"] = gf; sys.modules["glint.replica_gpu"] = rg
    sys.modules["glint.replica_gpu_batch"] = rgb
    sys.modules.pop("glint.hybrid_stream", None)
    hs = importlib.import_module("glint.hybrid_stream")

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return hs, restore


def suite(mode):
    """Part 2 in one mode: mode = {} (the default, batched) or dict(batch=False) (the per-frame arm)."""
    batched = mode.get("batch", True)

    def E(escalate):                   # the mode rides along in every dict form of `escalate`
        if isinstance(escalate, dict):
            return {**mode, **escalate} if escalate or mode else escalate
        if escalate is True or isinstance(escalate, np.bool_) and escalate:
            return dict(mode) if mode else escalate
        return escalate

    def TIE_COST(k_null):              # searches spent on the tied frame: its first copy ties
        return 1 + min(DEEP_ROUND_COPIES, k_null) if batched else 2

    def use(w):
        hs.index_blind_nbest = w.nbest; hs.index_known_gpu_cell = w.known; hs.matched_strict = w.matched
        sys.modules["glint.replica_gpu_batch"].world = w

    w0 = World(); use(w0)
    res0, st0 = hs.hybrid_index(FRAMES, warmup=False)
    check("escalate=None: no escalation stats", not any(k.startswith(("n_escalat", "escalation")) for k in st0), st0.keys())
    check("escalate=None: no frame labelled", not any("escalated" in r for r in res0))
    check("escalate=None: the known-cell search only ever runs at the shipped depth",
          all(t == 8 and nc is None for _, t, nc in w0.calls), set((t, nc) for _, t, nc in w0.calls))
    base_idx = st0["n_idx"]

    w1 = World(); use(w1)
    res1, st1 = hs.hybrid_index(FRAMES, warmup=False, escalate=E(True))
    esc = sorted(i for i, r in enumerate(res1) if r.get("escalated"))
    check("escalated: the deep-found frames and the poor-fit frame, not the tied one", esc == [8, 9, 11], esc)
    check("stats: 4 candidates (8, 9, 10, 11), 3 escalated", st1["n_escalation_candidates"] == 4
          and st1["n_escalated"] == 3, {k: st1[k] for k in st1 if "escalat" in k})
    check(f"searches: 3 x 33 accepted + {TIE_COST(32)} for the tied frame", st1["escalation_searches"] == 3 * 33 + TIE_COST(32),
          st1["escalation_searches"])
    check("n_idx counts newly indexed frames once (the poor-fit frame was already counted)",
          st1["n_idx"] == base_idx + 2, (base_idx, st1["n_idx"]))
    check("the escalated frame carries its record (p = 1/33)",
          all(abs(res1[i]["escalation"]["p"] - 1 / 33) < 1e-12 for i in esc))
    check("the untouched frames are identical to the escalate=None run",
          all(np.array_equal(res0[i]["M"], res1[i]["M"]) if res0[i]["M"] is not None else res1[i]["M"] is None
              for i in range(len(FRAMES)) if i not in esc))

    w2 = World(); use(w2)
    res2, st2 = hs.hybrid_index(FRAMES, warmup=False, escalate=E(dict(k_null=8)))
    check("escalate=dict(k_null=8): 9 searches per accepted frame", st2["escalation_searches"] == 3 * 9 + TIE_COST(8),
          st2["escalation_searches"])
    try:
        hs.hybrid_index(FRAMES, warmup=False, escalate=E(dict(kk=1)))
        check("unknown escalate key raises", False)
    except ValueError:
        check("unknown escalate key raises", True)

    def run(escalate):
        w = World(); use(w)
        res, st = hs.hybrid_index(FRAMES, warmup=False, escalate=E(escalate))
        return w, sorted(i for i, r in enumerate(res) if r.get("escalated")), st

    w3, esc3, st3 = run(dict(topa=16, nc=4))
    check("escalate=dict(topa=16, nc=4): the overrides reach the deep search",
          {(t, nc) for _, t, nc in w3.calls if t != 8} == {(16, 4)} and st3["escalation"]["topa"] == 16,
          sorted({(t, nc) for _, t, nc in w3.calls}))
    _, esc4, st4 = run(dict(k_null=np.int64(8)))
    check("numpy integers are accepted and stored as int", st4["escalation"]["k_null"] == 8
          and type(st4["escalation"]["k_null"]) is int and st4["escalation_searches"] == 3 * 9 + TIE_COST(8), st4["escalation"])
    for form in ({}, np.True_):
        _, e, st = run(form)
        check(f"escalate={form!r} ({type(form).__name__}) runs with the defaults, like True", e == esc and st["escalation_searches"] ==
              st1["escalation_searches"] and st["escalation"] == st1["escalation"], (e, st.get("escalation")))
    _, e, st = run(False)
    check("escalate=False is off, like None", e == [] and "n_escalated" not in st)
    for bad, exc in (("yes", TypeError), (1, TypeError), ([("k_null", 8)], TypeError),
                     (dict(batch="yes"), ValueError), (dict(round_copies=0), ValueError),
                     (dict(topa=1.5), ValueError), (dict(nc=0), ValueError), (dict(k_null=0), ValueError),
                     (dict(k_null=True), ValueError), (dict(seed=-1), ValueError), (dict(topa="128"), ValueError)):
        w = World(); use(w)
        try:
            hs.hybrid_index(FRAMES, warmup=False, escalate=E(bad))
            check(f"escalate={bad!r} raises {exc.__name__} before any search", False)
        except exc:
            check(f"escalate={bad!r} raises {exc.__name__} before any search", w.calls == [], w.calls[:3])
    return res0, res1, esc


w0 = World(); hs, restore = load_hybrid(w0)
try:
    out = {}
    for label, mode in (("batched (default)", {}), ("per-frame (batch=False)", dict(batch=False))):
        print(f"\n  -- mode: {label}")
        out[label] = suite(mode)
    a, b = out["batched (default)"][1], out["per-frame (batch=False)"][1]
    check("batched and per-frame runs give identical results (escalated set, M and null counts)",
          [bool(r.get("escalated")) for r in a] == [bool(r.get("escalated")) for r in b]
          and all((x["M"] is None and y["M"] is None) or (x["M"] is not None and y["M"] is not None
                  and np.array_equal(x["M"], y["M"])) for x, y in zip(a, b))
          and all(x["escalation"]["null_m"] == y["escalation"]["null_m"] for x, y in zip(a, b) if x.get("escalated")))
    res0, res1, esc = out["batched (default)"]
finally:
    restore()

# ------------------------------------------------------------------------------------------------
# Part 3: the label reaches the stream (both writers the CLI uses)
# ------------------------------------------------------------------------------------------------
print("\nstream writers: escalated frames carry glint/escalated and the null's numbers; no other frame does")
import re                                                                # noqa: E402
import tempfile                                                          # noqa: E402

from glint.predict import write_stream_integrated                        # noqa: E402
from glint.stream import escalation_lines, write_stream                  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    for name, writer in (("stream.write_stream", write_stream),
                         ("predict.write_stream_integrated", write_stream_integrated)):
        path = os.path.join(tmp, name + ".stream")
        writer(res1, path)
        chunks = open(path).read().split("----- Begin chunk -----")[1:]
        flagged = sorted(int(re.search(r"Image serial number: (\d+)", c).group(1)) - 1
                         for c in chunks if "glint/escalated = 1" in c)
        check(f"{name}: exactly the escalated frames are flagged", flagged == esc, (flagged, esc))
        c8 = chunks[esc[0]]
        rec8 = res1[esc[0]]["escalation"]
        want = {"glint/escalation_p": f"{1 / 33:.6g}", "glint/escalation_matched": str(rec8["m"]),
                "glint/escalation_null_max": str(max(rec8["null_m"]))}
        got = {k: re.search(re.escape(k) + r" = (\S+)", c8).group(1) for k in want if (k + " = ") in c8}
        check(f"{name}: p, matched and best-copy counts written", got == want, (got, want))
        body = c8.split("--- Begin crystal")[1].split("Reflections measured after indexing")[0]
        check(f"{name}: inside the crystal block", "glint/escalated = 1" in body)
check("escalation_lines is empty for a frame without the label", escalation_lines(res0[0]) == []
      and escalation_lines({"escalated": False, "escalation": {"p": 0.5, "m": 3, "null_m": [1]}}) == [])

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
