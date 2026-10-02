"""CPU tests for hybrid_index(select=...): which consensus-consistent candidate a frame keeps.

select="first" (the default) is the shipped rule: the first N-best cell same_lattice with the consensus cell,
the known-cell search only on frames without one. select="matched" runs the known-cell search on every frame
and keeps the consistent candidate (N-best cells in order, then the known-cell fit) matching the most peaks,
ties keeping that order. The indexers are scripted fakes; glint.glint_fast and glint.replica_gpu (torch at
import) are stubbed, the same way experiments/test_escalation.py does it.

Convention: a cell matrix M has COLUMNS = real-space axes in Angstrom and q @ M = hkl.

  PYTHONPATH=. python experiments/test_select_matched.py      # exit 0 = all pass
"""
import importlib
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint.lattice import cell_to_Ar                                    # noqa: E402

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


def frame_on(M, rng, n=40):
    hkl = rng.integers(-5, 6, size=(4 * n, 3)).astype(float)
    hkl = hkl[np.abs(hkl).sum(1) > 0][:n]
    return hkl @ np.linalg.inv(M)


CELL = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
JUNK = rot(2.71, 1.34, 0.19) @ cell_to_Ar(53.7, 61.3, 71.9, 90, 90, 90)      # not the consensus lattice
rng = np.random.default_rng(12)
TRUE = [rot(*rng.uniform(0, 3, 3)) @ CELL for _ in range(12)]
FRAMES = [frame_on(Mt, rng) for Mt in TRUE]
OFF = {i: rot(0.09, 0.0, 0.0) @ TRUE[i] for i in range(12)}                   # right cell, ~5 deg off: fails the gate

# What the blind N-best list and the known-cell search return, per frame.
EASY = set(range(7))            # blind top-1 is right, the known-cell fit is right too: a tie, blind kept
BLIND_POOR = 7                  # blind: only a poor consistent cell; known-cell: right  -> matched swaps to KC
DEEPER = 8                      # blind: [JUNK, poor, right]; known-cell: poor           -> matched keeps the 3rd blind
NO_BLIND = 9                    # blind: nothing; known-cell: right                        -> both keep KC
KC_WRONG = 10                   # blind: nothing; known-cell: a different lattice          -> both leave it unindexed
KC_POOR = 11                    # blind: right; known-cell: poor                           -> both keep blind


class World:
    def __init__(self):
        self.known_calls = []; self.nbest_calls = 0

    def idx(self, q):
        return next((j for j, f in enumerate(FRAMES) if f is q), None)

    def nbest(self, q, k):
        self.nbest_calls += 1
        i = self.idx(q)
        if i in EASY or i == KC_POOR:
            return [(TRUE[i], 1.0)]
        if i == BLIND_POOR:
            return [(OFF[i], 1.0)]
        if i == DEEPER:
            return [(JUNK, 3.0), (OFF[i], 2.0), (TRUE[i], 1.0)]
        return []

    def known(self, q, Mc, topa=8, nc=None):
        i = self.idx(q)
        self.known_calls.append(i)
        if i in (DEEPER, KC_POOR):
            return OFF[i]
        if i == KC_WRONG:
            return JUNK
        return TRUE[i]

    def matched(self, M, q):
        return count_strict(M, q)


def load_hybrid(world):
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


def run(hs, w, **kw):
    hs.index_blind_nbest = w.nbest; hs.index_known_gpu_cell = w.known; hs.matched_strict = w.matched
    return hs.hybrid_index(FRAMES, warmup=False, **kw)


def same_M(a, b):
    return (a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b))


w = World()
hs, restore = load_hybrid(w)
try:
    print("hybrid_index(select=...): the default is the shipped rule")
    w0 = World(); r0, s0 = run(hs, w0)
    w1 = World(); r1, s1 = run(hs, w1, select="first")
    check("select='first' is the default: same matrices, same stats",
          all(same_M(a["M"], b["M"]) for a, b in zip(r0, r1))
          and {k: v for k, v in s0.items() if k not in ("Mc", "edges")} == {k: v for k, v in s1.items() if k not in ("Mc", "edges")}
          and np.array_equal(s0["Mc"], s1["Mc"])
          and w0.known_calls == w1.known_calls)
    check("a consensus cell formed", s1["Mc"] is not None, s1.get("support"))
    check("first: the known-cell search runs only on frames with no consistent blind cell",
          sorted(w1.known_calls) == [NO_BLIND, KC_WRONG] and s1["n_kc_searches"] == 2, w1.known_calls)
    check("first: a poor consistent blind cell is kept (the case select='matched' is for)",
          same_M(r1[BLIND_POOR]["M"], OFF[BLIND_POOR]))
    check("first: the first consistent cell in N-best order is kept, not the best one",
          same_M(r1[DEEPER]["M"], OFF[DEEPER]))
    check("first: stats report select and no swaps", s1["select"] == "first" and s1["n_select_swaps"] == 0, s1)

    print("\nselect='matched': every frame gets the known-cell search; the best-matching candidate wins")
    w2 = World(); r2, s2 = run(hs, w2, select="matched")
    check("the known-cell search runs on every frame, once", sorted(w2.known_calls) == list(range(12))
          and s2["n_kc_searches"] == 12, w2.known_calls)
    check("a poor blind cell loses to a better known-cell fit", same_M(r2[BLIND_POOR]["M"], TRUE[BLIND_POOR]))
    check("a deeper blind cell that matches more wins over the first one (and over a poor KC fit)",
          same_M(r2[DEEPER]["M"], TRUE[DEEPER]))
    check("a tie keeps the blind cell (N-best order, then the known-cell fit)",
          all(r2[i]["M"] is TRUE[i] and w2.known_calls.count(i) == 1 for i in EASY))
    check("a poorer known-cell fit does not replace a good blind cell", r2[KC_POOR]["M"] is TRUE[KC_POOR])
    check("no blind cell: the known-cell fit is kept, as before", same_M(r2[NO_BLIND]["M"], TRUE[NO_BLIND]))
    check("a known-cell fit on another lattice is still refused", r2[KC_WRONG]["M"] is None)
    check("the swaps are counted against the shipped rule (2 frames change pick)", s2["n_select_swaps"] == 2, s2)
    check("counters: one KC pick that replaced a blind cell + one with none; one non-top-1 N-best pick",
          s2["n_resc"] == 2 and s2["n_nbest"] == 1 and s2["n_idx"] == s1["n_idx"] == 11, s2)
    check("the frames the two rules agree on are identical",
          all(same_M(r1[i]["M"], r2[i]["M"]) for i in range(12) if i not in (BLIND_POOR, DEEPER)))
    gained = [i for i in range(12) if count_strict(r2[i]["M"], FRAMES[i]) > count_strict(r1[i]["M"], FRAMES[i])]
    check("select='matched' never matches fewer peaks than 'first' on any frame, and more on the two swaps",
          gained == [BLIND_POOR, DEEPER] and all(count_strict(r2[i]["M"], FRAMES[i]) >= count_strict(r1[i]["M"], FRAMES[i])
                                                 for i in range(12)), gained)

    print("\nselect='matched' with a supplied cell, and without a consensus")
    w3 = World(); r3, s3 = run(hs, w3, select="matched", Mc_known=CELL)
    check("Mc_known: same picks as with the consensus cell here",
          all(same_M(a["M"], b["M"]) for a, b in zip(r2, r3)) and len(w3.known_calls) == 12)
    hs.CONSENSUS_MIN_FRAC = 2.0                                  # no pooled cluster can clear this: consensus refuses
    try:
        w4 = World(); r4, s4 = run(hs, w4, select="matched")
        w5 = World(); r5, s5 = run(hs, w5, select="first")
    finally:
        hs.CONSENSUS_MIN_FRAC = importlib.import_module("glint.multishot").CONSENSUS_MIN_FRAC
    check("consensus refused: no known-cell search in either mode, identical top-1 fallback",
          s4["consensus_refused"] and not w4.known_calls and not w5.known_calls
          and all(same_M(a["M"], b["M"]) for a, b in zip(r4, r5)), (s4.get("consensus_refused"), w4.known_calls))

    print("\nbad arguments")
    for bad in ("best", None, "MATCHED", 1):
        w6 = World()
        try:
            run(hs, w6, select=bad)
            ok = False
        except ValueError:
            ok = not w6.known_calls and w6.nbest_calls == 0
        check(f"select={bad!r} raises ValueError before any search", ok)
    check("SELECTS lists the choices", hs.SELECTS == ("first", "matched"), hs.SELECTS)
finally:
    restore()

print(f"\n{'FAILURES: ' + ', '.join(FAILS) if FAILS else 'ALL PASS'}")
sys.exit(1 if FAILS else 0)
