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
from glint.retry_cascade import DEEP_K_NULL, DEEP_NC, DEEP_SEED, DEEP_TOPA, arm_known_deep  # noqa: E402

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
        # a scrambled copy of self.current
        return TIE if self.current in TIED else JUNK

    def matched(self, M, q):
        return 10 ** 6 if np.array_equal(np.asarray(M), TIE) else count_strict(M, q)


def load_hybrid(world):
    """Import glint.hybrid_stream against stub glint_fast / replica_gpu; return (module, restore)."""
    saved = {k: sys.modules.get(k) for k in ("glint.glint_fast", "glint.replica_gpu", "glint.hybrid_stream")}
    gf = types.ModuleType("glint.glint_fast")
    gf.GATE_FRAC, gf.GATE_MIN = 0.25, 10
    gf.index_blind_fast = lambda q: None
    gf.index_blind_nbest = world.nbest
    gf.load = lambda p: []
    gf.matched_strict = world.matched
    rg = types.ModuleType("glint.replica_gpu")
    rg.index_known_gpu_cell = world.known
    sys.modules["glint.glint_fast"] = gf; sys.modules["glint.replica_gpu"] = rg
    sys.modules.pop("glint.hybrid_stream", None)
    hs = importlib.import_module("glint.hybrid_stream")

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return hs, restore


w0 = World(); hs, restore = load_hybrid(w0)
try:
    res0, st0 = hs.hybrid_index(FRAMES, warmup=False)
    check("escalate=None: no escalation stats", not any(k.startswith(("n_escalat", "escalation")) for k in st0), st0.keys())
    check("escalate=None: no frame labelled", not any("escalated" in r for r in res0))
    check("escalate=None: the known-cell search only ever runs at the shipped depth",
          all(t == 8 and nc is None for _, t, nc in w0.calls), set((t, nc) for _, t, nc in w0.calls))
    base_idx = st0["n_idx"]

    w1 = World(); hs.index_blind_nbest = w1.nbest; hs.index_known_gpu_cell = w1.known; hs.matched_strict = w1.matched
    res1, st1 = hs.hybrid_index(FRAMES, warmup=False, escalate=True)
    esc = sorted(i for i, r in enumerate(res1) if r.get("escalated"))
    check("escalated: the deep-found frames and the poor-fit frame, not the tied one", esc == [8, 9, 11], esc)
    check("stats: 4 candidates (8, 9, 10, 11), 3 escalated", st1["n_escalation_candidates"] == 4
          and st1["n_escalated"] == 3, {k: st1[k] for k in st1 if "escalat" in k})
    check("searches: 3 x 33 accepted + 2 for the tied frame", st1["escalation_searches"] == 3 * 33 + 2,
          st1["escalation_searches"])
    check("n_idx counts newly indexed frames once (the poor-fit frame was already counted)",
          st1["n_idx"] == base_idx + 2, (base_idx, st1["n_idx"]))
    check("the escalated frame carries its record (p = 1/33)",
          all(abs(res1[i]["escalation"]["p"] - 1 / 33) < 1e-12 for i in esc))
    check("the untouched frames are identical to the escalate=None run",
          all(np.array_equal(res0[i]["M"], res1[i]["M"]) if res0[i]["M"] is not None else res1[i]["M"] is None
              for i in range(len(FRAMES)) if i not in esc))

    w2 = World(); hs.index_blind_nbest = w2.nbest; hs.index_known_gpu_cell = w2.known; hs.matched_strict = w2.matched
    res2, st2 = hs.hybrid_index(FRAMES, warmup=False, escalate=dict(k_null=8))
    check("escalate=dict(k_null=8): 9 searches per accepted frame", st2["escalation_searches"] == 3 * 9 + 2,
          st2["escalation_searches"])
    try:
        hs.hybrid_index(FRAMES, warmup=False, escalate=dict(kk=1))
        check("unknown escalate key raises", False)
    except ValueError:
        check("unknown escalate key raises", True)

    def run(escalate):
        w = World(); hs.index_blind_nbest = w.nbest; hs.index_known_gpu_cell = w.known; hs.matched_strict = w.matched
        res, st = hs.hybrid_index(FRAMES, warmup=False, escalate=escalate)
        return w, sorted(i for i, r in enumerate(res) if r.get("escalated")), st

    w3, esc3, st3 = run(dict(topa=16, nc=4))
    check("escalate=dict(topa=16, nc=4): the overrides reach the deep search",
          {(t, nc) for _, t, nc in w3.calls if t != 8} == {(16, 4)} and st3["escalation"]["topa"] == 16,
          sorted({(t, nc) for _, t, nc in w3.calls}))
    _, esc4, st4 = run(dict(k_null=np.int64(8)))
    check("numpy integers are accepted and stored as int", st4["escalation"]["k_null"] == 8
          and type(st4["escalation"]["k_null"]) is int and st4["escalation_searches"] == 3 * 9 + 2, st4["escalation"])
    for form in ({}, np.True_):
        _, e, st = run(form)
        check(f"escalate={form!r} ({type(form).__name__}) runs with the defaults, like True", e == esc and st["escalation_searches"] ==
              st1["escalation_searches"] and st["escalation"] == st1["escalation"], (e, st.get("escalation")))
    _, e, st = run(False)
    check("escalate=False is off, like None", e == [] and "n_escalated" not in st)
    for bad, exc in (("yes", TypeError), (1, TypeError), ([("k_null", 8)], TypeError),
                     (dict(topa=1.5), ValueError), (dict(nc=0), ValueError), (dict(k_null=0), ValueError),
                     (dict(k_null=True), ValueError), (dict(seed=-1), ValueError), (dict(topa="128"), ValueError)):
        w = World(); hs.index_blind_nbest = w.nbest; hs.index_known_gpu_cell = w.known; hs.matched_strict = w.matched
        try:
            hs.hybrid_index(FRAMES, warmup=False, escalate=bad)
            check(f"escalate={bad!r} raises {exc.__name__} before any search", False)
        except exc:
            check(f"escalate={bad!r} raises {exc.__name__} before any search", w.calls == [], w.calls[:3])
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
