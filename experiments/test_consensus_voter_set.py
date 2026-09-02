"""Does a consensus group's SUPPORT equal the hypotheses its representative actually covers? (glint#182)

THE BUG. `RunningConsensus.add` folds groups together when an arriving cell matches several of them
(merge-on-multi-match) and used to make that arriving WITNESS the merged group's representative, on
the argument that the witness matches every group it merged. Tolerance matching is not transitive: a
member that matched a group's FOUNDER need not match the witness, so the merged group's support could
exceed the number of pooled hypotheses within tolerance of the matrix that gets locked. StreamDriver
locks that matrix and its alias gate collects voters against it (`same_lattice(c, Mc)`), so a lock
could carry a `consensus_support` its representative did not command. The issue's minimal probe:
cells diag(a, 120, 140) with a = 100, 97, 97, 97, 106, then the witness 103 -> support 6 while only
3 pooled cells match the locked a=103 (the 97s are 6% from it, past rtol = 5%).

THE FIX. Every group keeps its members. After a merge the representative is chosen among {the
heaviest merged group's representative, the witness, the other merged groups' representatives} by
how many members it covers (ties -> the heaviest group's representative, so the lock churns least),
and support is RECOUNTED as that coverage. Members the chosen representative does not cover stay in
the group -- so a later recount is exact -- but do not count. `leader_counts()` reports both numbers.

PROBES
  1. the issue's probe: support == voters within tolerance of the representative (5 == 5: the founder
     a=100 covers 100, 97x3 and 103; 106 is absorbed but uncovered), leader_counts() == (5, 6);
  2. a witness that GENUINELY covers everyone (a = 100, 101x3, 106, witness 103; all within 3% of it):
     support stays 6 == voters, so the recount does not tax the case the merge exists for;
  3. randomised, 3%-spaced length ladders in random order (most seeds merge repeatedly; the guard
     against a vacuous probe is on the total): after EVERY add each group's support equals its members
     within tolerance of its representative (member for member, via `_match`), every hypothesis sits
     in exactly one member list, the leader's support never exceeds the pool's matches for its
     representative and never shrinks across an add; merge=False never merges, so there support ==
     members throughout;
  4. the split-vote repair the merge exists for (test_running_consensus.py's A/B halves + witness) still
     recombines both halves under the witness: support 19, unchanged;
  5. StreamDriver.stats() reports `consensus_members` beside `consensus_support` while blind.

INSTRUMENT CHECK. Against the pre-fix running_consensus.py this file prints probe 1 as support 6,
voters 3 and exits 1 (recorded in the PR that added it).

  PYTHONPATH=. python experiments/test_consensus_voter_set.py       # exit 0 = all pass, < 5 s
"""
import contextlib
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from glint.running_consensus import RunningConsensus          # noqa: E402
from glint.multishot import same_lattice                       # noqa: E402

fails = []


def check(name, cond, msg=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + str(msg)) if msg else ''}")
    if not cond:
        fails.append(name)


def cell(a, b=120.0, c=140.0):
    return np.diag([1.0 / a, 1.0 / b, 1.0 / c])


def voters(rep, pool):
    """The driver's own voter test: hypotheses within tolerance of the LOCKED matrix."""
    return sum(1 for M in pool if same_lattice(M, rep))


def a_of(M):
    return 1.0 / M[0, 0]


def feed(avals, **kw):
    rc = RunningConsensus(min_support=3, gap=2, adaptive=False, **kw)
    pool = []
    for a in avals:
        M = cell(a)
        rc.add(M)
        pool.append(M)
    return rc, pool


def members_in_tolerance(rc, g):
    """Members of group g within tolerance of ITS representative, counted with the scalar `_match`."""
    return sum(1 for (_M, lm, cm, dm) in g[5] if rc._match(lm, cm, dm, g))


def probe(title, avals, exp_rep_a, exp_support, exp_members):
    print(f"\n{title}")
    rc, pool = feed(avals)
    rep, sup, _lead = rc.verdict()                    # what the driver locks: the stop rule must fire
    check("verdict() fires on this pool", rep is not None, f"support {sup}")
    if rep is None:
        return
    v = voters(rep, pool)
    print(f"  groups: {[(round(a_of(g[0]), 1), g[4]) for g in rc.groups]}  ->  representative "
          f"a={a_of(rep):.0f}, support {sup}, voters {v}")
    check("support == voters within tolerance of the representative", sup == v,
          f"support {sup}, voters {v}")
    check(f"representative is a={exp_rep_a:.0f}", abs(a_of(rep) - exp_rep_a) < 1e-9, f"a={a_of(rep):.1f}")
    check(f"support == {exp_support}", sup == exp_support, sup)
    try:
        s, m = rc.leader_counts()
    except AttributeError as e:
        check("leader_counts() exposes (support, members)", False, e)
        return
    check("leader_counts()[0] == verdict() support", s == sup, (s, sup))
    check(f"leader_counts()[1] == {exp_members} hypotheses folded into the group", m == exp_members, m)
    check("every hypothesis sits in exactly one group's member list",
          sum(len(g[5]) for g in rc.groups) == rc.npool == len(pool),
          (sum(len(g[5]) for g in rc.groups), rc.npool, len(pool)))
    check("each group's support == its members within tolerance of its representative",
          all(g[4] == members_in_tolerance(rc, g) for g in rc.groups),
          [(g[4], members_in_tolerance(rc, g)) for g in rc.groups])


def randomised(seed, n=300, rungs=8, step=1.03):
    """3%-spaced rungs in random order: neighbours match (3%), next-nearest (6.1%) do not, so a rung
    between two group representatives two rungs apart is a multi-match witness."""
    rng = np.random.default_rng(seed)
    ladder = 100.0 * step ** np.arange(rungs)
    avals = rng.choice(ladder, n)
    rc, rc_off = (RunningConsensus(min_support=3, gap=2, adaptive=False),
                  RunningConsensus(min_support=3, gap=2, adaptive=False, merge=False))
    pool, merges, prev_max, bad = [], 0, 0, []
    for i, a in enumerate(avals):
        M = cell(a)
        before = len(rc.groups)
        rc.add(M); rc_off.add(M); pool.append(M)
        if len(rc.groups) < before:
            merges += 1
        top = max(g[4] for g in rc.groups)
        if top < prev_max:
            bad.append((i, "leader support shrank", prev_max, top))
        prev_max = top
        for g in rc.groups:
            cov = members_in_tolerance(rc, g)
            if g[4] != cov:
                bad.append((i, "support != covered members", g[4], cov))
        if sum(len(g[5]) for g in rc.groups) != rc.npool:
            bad.append((i, "member lists do not partition the pool"))
        rep, sup, _ = rc.leaders()                    # NOT verdict(): its cell is None until the rule fires
        if sup > voters(rep, pool):
            bad.append((i, "support exceeds pool matches of the representative", sup, voters(rep, pool)))
        for g in rc_off.groups:
            if g[4] != len(g[5]):
                bad.append((i, "merge=False: support != members", g[4], len(g[5])))
    return merges, bad, rc.npool


def main():
    probe("PROBE 1 -- the issue's probe: a = 100, 97, 97, 97, 106, then witness 103",
          (100, 97, 97, 97, 106, 103), exp_rep_a=100, exp_support=5, exp_members=6)
    probe("PROBE 2 -- the witness covers everyone: a = 100, 101, 101, 101, 106, then witness 103",
          (100, 101, 101, 101, 106, 103), exp_rep_a=103, exp_support=6, exp_members=6)

    print("\nPROBE 3 -- randomised ladders (invariants after every add)")
    # A seed CAN found its groups three rungs apart ({0, 3, 6}: every rung within one of exactly one
    # representative), and then no arrival is ever a multi-match -- seed 1 does. So the guard against
    # a vacuous probe is on the TOTAL merges across seeds, with each seed's count printed.
    total_merges = 0
    for seed in (0, 1, 2, 3):
        try:
            merges, bad, npool = randomised(seed)
        except Exception as e:                        # pre-fix: no member lists -> IndexError/TypeError
            check(f"seed {seed}: invariants hold", False, f"{type(e).__name__}: {e}")
            continue
        total_merges += merges
        check(f"seed {seed}: invariants hold after every add", not bad,
              f"{merges} merges over {npool} hypotheses" if not bad else bad[:3])
    check("the ladders produced merges (else the probe is vacuous)", total_merges > 0,
          f"{total_merges} merges over the seeds")

    print("\nPROBE 4 -- the split-vote repair is intact (A/B halves + witness, test_running_consensus.py)")
    TRUE = np.diag(1.0 / np.array([79.0, 79.0, 38.0]))
    A, B = TRUE / 0.972, TRUE / 1.028                    # -2.8% / +2.8% from TRUE, 5.8% from each other
    SPUR = np.diag(1.0 / np.array([61.0, 67.0, 73.0]))
    rc = RunningConsensus(min_support=3, gap=2, adaptive=False)
    for M in [A] * 9 + [B] * 9 + [SPUR] * 12 + [TRUE]:
        rc.add(M)
    top = max(rc.groups, key=lambda g: g[4])
    check("pooled winner is the true lattice", same_lattice(top[0], TRUE))
    check("witness chosen as representative (covers both halves): support 19", top[4] == 19, top[4])
    check("...and 19 == the pool's matches for it", top[4] == voters(top[0], [A] * 9 + [B] * 9 + [TRUE]))

    @contextlib.contextmanager
    def _no_torch_needed():
        """Stand in for glint.glint_fast so the blind branch's lazy import resolves without torch
        (the seam test_lock_gate_wiring.py uses; nothing here indexes a frame)."""
        stub = types.ModuleType("glint.glint_fast")
        stub.index_blind_nbest = lambda q, k: []
        had, prev = "glint.glint_fast" in sys.modules, sys.modules.get("glint.glint_fast")
        sys.modules["glint.glint_fast"] = stub
        try:
            yield
        finally:
            if had:
                sys.modules["glint.glint_fast"] = prev
            else:
                sys.modules.pop("glint.glint_fast", None)

    print("\nPROBE 5 -- StreamDriver.stats() reports consensus_members beside consensus_support (blind)")
    try:
        from glint.stream_driver import StreamDriver

        N = 64
        panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=1e4,
                       cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5), coffset=0.0,
                       min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
        with _no_torch_needed():
            d = StreamDriver(None, panels, 0.1, 1.3, (N, N), dtype=np.uint16, B=8, dmin=3.0,
                             use_gpu=False, lock_support=3, lock_gap=2, adaptive_gap=False)
        for a in (100, 97, 97, 97, 106, 103):
            d._rc.add(cell(a))
        s = d.stats()
        check("blind stats(): consensus_support is the covered count (5)", s.get("consensus_support") == 5,
              s.get("consensus_support"))
        check("blind stats(): consensus_members is the folded count (6)", s.get("consensus_members") == 6,
              s.get("consensus_members"))
        check("driver starts with an empty lock record for members", d.consensus_members is None,
              getattr(d, "consensus_members", "<missing>"))
    except Exception as e:
        check("StreamDriver probe ran", False, f"{type(e).__name__}: {e}")


    print("\nPROBE 6 -- a SUCCESSFUL lock persists the folded count, on both lock paths")
    # Probe 5 only reads stats() while the driver is still blind, so removing either
    # `self.consensus_members = ...` assignment would leave it green (Copilot review of #187). These
    # two drive a lock to completion -- the sequential path through _push_blind's verdict and the
    # batched path through warmup_batch -- and assert the count survives onto the locked driver.
    try:
        from glint.stream_driver import StreamDriver

        N = 64
        panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=1e4,
                       cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5), coffset=0.0,
                       min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
        LADDER = (100, 97, 97, 97, 106, 103)          # the issue's pool: folds to 6, covers 5

        def _blind_driver():
            with _no_torch_needed():
                return StreamDriver(None, panels, 0.1, 1.3, (N, N), dtype=np.uint16, B=8, dmin=3.0,
                                    use_gpu=False, lock_support=3, lock_gap=2, adaptive_gap=False)

        # --- sequential path: push frames through _push_blind so the DRIVER's own verdict fires the
        # lock. Setting consensus_members by hand here would test nothing -- that is the flaw the
        # review named -- so the blind indexer is stubbed to emit one ladder rung per frame and the
        # driver is left to pool, vote, gate and lock on its own.
        d = _blind_driver()
        it_seq = iter(LADDER)

        def _one_rung(q, k=3, fanout=None):
            try:
                return [(cell(next(it_seq)), 1.0)]
            except StopIteration:
                return []

        d._blind_index = _one_rung
        # _push_blind only reaches the vote when the frame yields >= min_peaks peaks, so give it a
        # sparse grid of bright pixels: what is being tested is the lock bookkeeping, and the stubbed
        # indexer supplies the cells regardless of where the peaks fall.
        _rng = np.random.default_rng(0)
        spotty = _rng.poisson(20, (N, N)).astype(np.uint16)      # a background the finder can measure
        for r in range(8, N - 8, 10):
            for c in range(8, N - 8, 10):
                spotty[r - 1:r + 2, c - 1:c + 2] = 3000          # 3x3 blobs -> ~25 peaks, >= min_peaks
        for _ in LADDER:
            if not d._blind:
                break
            d._push_blind(spotty)
        want_sup, want_mem = d.consensus_support, d.consensus_members
        check("sequential: _push_blind drove the driver to a lock", d._blind is False and d.Mc is not None,
              f"blind={d._blind}")
        if d.Mc is not None:
            check("sequential: the locked driver kept the folded count",
                  d.consensus_members is not None and d.consensus_members >= 1,
                  f"{d.consensus_members}")
            check("sequential: support is the covered count, and members >= support",
                  d.consensus_support is not None and d.consensus_members >= d.consensus_support,
                  f"support {d.consensus_support}, members {d.consensus_members}")
            st = d.stats()
            check("sequential: stats() reports both after the lock",
                  st.get("consensus_members") == want_mem and st.get("consensus_support") == want_sup,
                  f"{st.get('consensus_support')}, {st.get('consensus_members')}")

        # --- batched path: warmup_batch over a real (B,H,W) stack. The frames carry no Bragg peaks,
        # so the driver's own peak-find yields nothing and the vote is driven by stubbing the blind
        # indexer to emit the ladder -- the point is the LOCK BOOKKEEPING, not the indexing.
        d2 = _blind_driver()
        frames = np.zeros((len(LADDER), N, N), np.uint16)
        it = iter(LADDER)

        def _stub_batch(qs, blind, rc, nbest, fanout, sink=None):
            for a in it:
                rc.add(cell(a))
            return rc.verdict()[0], rc.verdict()[1]

        locked = False
        import glint.warmup_batch as wb_mod          # warmup_batch imports it lazily, by module
        real_wc = wb_mod.warmup_consensus
        wb_mod.warmup_consensus = _stub_batch
        try:
            locked = bool(d2.warmup_batch(frames))
        except Exception as e:                       # a stub mismatch must not read as a pass
            check("batched: warmup_batch ran", False, f"{type(e).__name__}: {e}")
        finally:
            wb_mod.warmup_consensus = real_wc
        if locked:
            check("batched: the locked driver kept the folded count",
                  d2.consensus_members is not None and d2.consensus_members >= d2.consensus_support,
                  f"members {d2.consensus_members}, support {d2.consensus_support}")
            check("batched: stats() reports the folded count after the lock",
                  d2.stats().get("consensus_members") == d2.consensus_members,
                  d2.stats().get("consensus_members"))
        else:
            check("batched: warmup_batch reached a lock", False,
                  "no lock -- the stub did not drive warmup_consensus to a verdict")
    except Exception as e:
        check("locked-driver probe ran", False, f"{type(e).__name__}: {e}")

    return 0 if not fails else 1


if __name__ == "__main__":
    rc = main()
    print()
    print("ALL PASS" if not fails else f"FAIL: {len(fails)} check(s): {fails}")
    sys.exit(rc)
