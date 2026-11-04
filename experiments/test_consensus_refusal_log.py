"""The consensus refusal log says which test refused and who the runner-up was.

WHY THIS EXISTS. The line used to read "best cluster N = S% of P pooled -- below the min_frac/min_lead
gate". It printed the share, but a share of 20-30% clears min_frac = 2%; the test that refuses such a
cluster is min_lead, and the runner-up's size and cell were not printed at all. Reading that line, one
blames the share.

Checked here:
  * the rule is unchanged: consensus_accept agrees with a literal copy of the old body on a grid;
  * passing detail= does not change the returned cell or support, on either consensus path;
  * the detail and the message name exactly the tests that failed, and the runner-up's count and cell;
  * _report prints it (torch half, skips where glint.hybrid_stream cannot import).

  PYTHONPATH=. python experiments/test_consensus_refusal_log.py     # exit 0 = all pass
"""
import contextlib
import io
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glint.multishot import consensus_cell                                     # noqa: E402
from glint.running_consensus import consensus_accept, consensus_failed_tests, refusal_reason  # noqa: E402

FAILS = []


def check(name, ok, info=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  [{info}]" if info and not ok else ""))
    if not ok:
        FAILS.append(name)


def old_accept(win, runner, n_pool, min_support, min_frac, min_lead, min_gap, pool_switch):
    """The body of consensus_accept before the log change, verbatim."""
    if win < min_support:
        return False
    scale_tests = n_pool >= pool_switch if pool_switch > 0 else True
    if scale_tests and min_frac > 0.0 and win < min_frac * n_pool:
        return False
    if scale_tests and min_lead > 1.0 and runner > 0 and win < min_lead * runner:
        return False
    if min_gap > 0 and (win - runner) < min_gap:
        return False
    return True


def cell(a, b, c, rng):
    """A cell with these edges in a random orientation, as the blind indexer would return it."""
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q @ np.diag([a, b, c])


def pool(rng, *groups):
    """groups = (count, (a, b, c)), with 0.2% jitter so members are near each other but not equal."""
    out = []
    for count, edges in groups:
        for _ in range(count):
            out.append(cell(*(np.array(edges) * (1 + 0.002 * rng.normal(size=3))), rng))
    return out


def rule_unchanged():
    print("rule")
    n = bad = 0
    for args in itertools.product((0, 2, 3, 4, 12, 45, 100), (0, 1, 3, 8, 30, 90), (0, 10, 71, 72, 500, 5535),
                                  (1, 3), (0.0, 0.02, 0.3), (1.0, 1.5, 2.0), (0, 3), (0, 72)):
        n += 1
        new = consensus_accept(*args)
        bad += new != old_accept(*args)
        bad += new != (not consensus_failed_tests(*args))
    check(f"consensus_accept == the old body on {n} cases", bad == 0, f"{bad} differ")


def both_paths(pooled, **kw):
    """Run with and without detail= on both consensus paths; return the detail of each."""
    details = []
    for fast in ("0", "1"):
        os.environ["CONSENSUS_FAST"] = fast
        try:
            plain, s0 = consensus_cell(pooled, **kw)
            d = {}
            with_d, s1 = consensus_cell(pooled, detail=d, **kw)
        finally:
            os.environ.pop("CONSENSUS_FAST", None)
        same = s0 == s1 and ((plain is None and with_d is None) or np.array_equal(plain, with_d))
        check(f"CONSENSUS_FAST={fast}: detail= leaves the cell and support unchanged", same)
        details.append(d)
    return details


def detail_names_the_failure():
    rng = np.random.default_rng(0)
    print("lead fails, share passes (the case the old line got wrong)")
    pooled = pool(rng, (60, (40.0, 60.0, 90.0)), (50, (33.0, 47.0, 75.0)))
    for d in both_paths(pooled, min_frac=0.02, min_lead=1.5):
        check("failed == [min_lead]", d["failed"] == ["min_lead"], repr(d["failed"]))
        check("runner-up count", d["runner"] == 50, repr(d["runner"]))
        check("leader and runner-up cells", np.allclose(d["cell"], (40, 60, 90), atol=0.6)
              and np.allclose(d["runner_cell"], (33.0, 47.0, 75.0), atol=0.6), repr((d["cell"], d["runner_cell"])))
        msg = refusal_reason(d)
        check("message names min_lead with the ratio", "min_lead: lead 1.20 < 1.50" in msg, msg)
        check("message does not blame the share", "min_frac" not in msg, msg)
        check("message gives the runner-up", "runner-up 50 at 33." in msg, msg)

    print("share fails, lead passes")
    edges = [(30.0 * 1.3 ** i, 30.0 * 1.3 ** j, 30.0 * 1.3 ** k)
             for i, j, k in itertools.combinations_with_replacement(range(6), 3)]
    pooled = pool(rng, (4, (40.0, 60.0, 90.0)), *[(1, e) for e in edges])
    for d in both_paths(pooled, min_frac=0.2, min_lead=1.5):
        check("failed == [min_frac]", d["failed"] == ["min_frac"], repr(d["failed"]))
        msg = refusal_reason(d)
        check("message names min_frac with the share", "min_frac: share" in msg and "< 20.0%" in msg, msg)
        check("message does not blame the lead", "min_lead" not in msg, msg)

    print("both fail")
    pooled = pool(rng, (4, (40.0, 60.0, 90.0)), (3, (33.0, 47.0, 75.0)), *[(1, e) for e in edges])
    for d in both_paths(pooled, min_frac=0.2, min_lead=1.5):
        check("failed == [min_frac, min_lead]", d["failed"] == ["min_frac", "min_lead"], repr(d["failed"]))

    print("no runner-up, leader below min_support")
    for d in both_paths(pool(rng, (2, (40.0, 60.0, 90.0)))):
        check("failed == [min_support]", d["failed"] == ["min_support"], repr(d["failed"]))
        msg = refusal_reason(d)
        check("message says there is no runner-up", msg.endswith("no runner-up") and "min_support: 2 < 3" in msg, msg)

    print("accepted")
    for d in both_paths(pool(rng, (60, (40.0, 60.0, 90.0)), (10, (33.0, 47.0, 75.0))), min_frac=0.02, min_lead=1.5):
        check("failed is empty, runner counted, its medoid not computed",
              d["failed"] == [] and d["runner"] == 10 and d["runner_cell"] is None, repr(d))


def report_half():
    print("_report")
    try:
        from glint.hybrid_stream import _report
    except ImportError as e:
        print(f"  SKIP -- {e}")
        return
    rng = np.random.default_rng(1)
    d = {}
    Mc, support = consensus_cell(pool(rng, (60, (40.0, 60.0, 90.0)), (50, (33.0, 47.0, 75.0))),
                                 min_frac=0.02, min_lead=1.5, detail=d)
    check("the planted pool is refused", Mc is None)
    stats = {"n": 110, "n_blind": 110, "support": support, "edges": None, "n_nbest": 0, "n_resc": 0, "n_casc": 0,
             "n_idx": 110, "Mc": None, "select": "first", "n_pool": 110, "support_frac": support / 110,
             "consensus_refused": True, "consensus_detail": d}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _report(stats, "x.stream")
    line = next((ln for ln in buf.getvalue().splitlines() if "consensus REFUSED" in ln), "")
    check("best cluster and share kept, for anyone reading old logs alongside",
          "best cluster 60 = 54.5% of 110 pooled" in line, line)
    check("names min_lead and the runner-up", "min_lead: lead 1.20 < 1.50" in line and "runner-up 50" in line, line)
    check("still says what the run did next", "per-frame top-1 fallback, NO rescue" in line, line)
    check("the old wording is gone", "min_frac/min_lead gate" not in line, line)


if __name__ == "__main__":
    rule_unchanged()
    detail_names_the_failure()
    report_half()
    print(f"{len(FAILS)} failed" if FAILS else "all passed")
    sys.exit(1 if FAILS else 0)
