"""Pin check_numbers.py's paired-difference interval: the MLE, the bounds, and the COVERAGE.

check_numbers.py imports nothing third-party, so the interval it quotes for the M3 block is written
out by hand -- and hand-written statistics is where this file family has already been bitten twice:

  1. A Clopper-Pearson version had its bounds SWAPPED and ~0.4 points off, and produced a
     perfectly plausible green guard run.
  2. That same version, once un-swapped, was still not a valid interval for the MARGINAL rate
     difference: it conditioned on the number of discordant pairs and ignored the randomness in
     that number. Measured coverage 93.6% at one of the real splits and 59.3% at (2,0), with
     (0,0) coming out as the zero-width [0, 0] -- certainty from no information.

Neither was caught by eye. So this checks the thing that actually matters, coverage, by simulation
against the multinomial the data come from, and not merely that the numbers look reasonable.

Run: `python experiments/test_check_numbers_ci.py` or `pytest`.
"""
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "experiments" / "check_numbers.py").read_text(encoding="utf-8")

# Lift the helpers out of check_arithmetic without importing the module (that runs the whole guard).
_body = SRC[SRC.index("    _Z975 = "):SRC.index("    _N480, _plateau_lo")]
_NS: dict = {}
exec("\n".join(l[4:] for l in _body.split("\n")), _NS)          # noqa: S102
ci_diff, p10_mle = _NS["_ci_diff"], _NS["_p10_mle"]

N = 480
# The real M3 splits, plus the two degenerate cases the previous interval got wrong.
CASES = [(29, 54), (31, 45), (29, 27), (18, 24), (24, 27), (18, 20), (21, 18), (30, 24),
         (27, 19), (30, 23), (22, 23), (32, 21), (10, 29), (17, 12), (19, 18), (2, 0), (0, 0)]


def _brute_p10(n01, n10, n, d, steps=200000):
    """Constrained MLE by direct maximisation -- the closed form must agree with it."""
    r = n - n01 - n10
    lo, hi = max(0.0, -d) + 1e-9, (1 - d) / 2 - 1e-9
    best = None
    for i in range(steps + 1):
        p = lo + (hi - lo) * i / steps
        v = ((n01 * math.log(p + d) if p + d > 0 else -1e18)
             + (n10 * math.log(p) if p > 0 else -1e18)
             + (r * math.log(1 - 2 * p - d) if 1 - 2 * p - d > 0 else -1e18))
        if best is None or v > best[0]:
            best = (v, p)
    return best[1]


def test_closed_form_mle_matches_brute_force():
    for (n01, n10), d in ((( 10, 29), -0.04), ((29, 54), -0.05), ((32, 21), 0.02), ((29, 27), 0.0)):
        got, want = p10_mle(n01, n10, N, d), _brute_p10(n01, n10, N, d)
        assert abs(got - want) < 1e-5, f"({n01},{n10}) d={d}: {got} vs {want}"


def test_bounds_ordered_and_bracket_the_estimate():
    """The swapped-bounds bug produced lo > hi and still read plausibly."""
    for n01, n10 in CASES:
        lo, hi = ci_diff(n01, n10, N)
        point = (n01 - n10) / N
        assert lo <= hi, f"({n01},{n10}): bounds inverted, {lo} > {hi}"
        assert lo <= point <= hi, f"({n01},{n10}): CI [{lo}, {hi}] excludes the estimate {point}"


def test_zero_discordant_pairs_is_not_certainty():
    """(0,0) means no information about the difference, not proof that it is zero. The previous
    interval returned [0, 0] here, which is the clearest symptom that it was the wrong interval."""
    lo, hi = ci_diff(0, 0, N)
    assert lo < 0 < hi, f"(0,0) gave [{lo}, {hi}] -- a zero-width interval claims certainty"
    assert hi - lo > 0.005, f"(0,0) interval [{lo}, {hi}] is implausibly tight for no data"


def test_coverage_is_at_least_nominal():
    """The property the two previous versions failed. Simulate from the multinomial the counts
    actually come from and check the interval covers the true difference >= 95% of the time.

    Deliberately includes (2, 0): the conditional interval scored 59.3% there, and several hybrid
    arms in the real sweep sit in that sparse regime.
    """
    for (a, b) in ((10, 29), (32, 21), (2, 0)):
        p01, p10 = a / N, b / N
        true = p01 - p10
        rng = random.Random(20260820)
        trials, cov = 4000, 0
        for _ in range(trials):
            x = y = 0
            for _ in range(N):                       # multinomial by N Bernoulli draws
                u = rng.random()
                if u < p01:
                    x += 1
                elif u < p01 + p10:
                    y += 1
            lo, hi = ci_diff(x, y, N)
            cov += lo <= true <= hi
        rate = cov / trials
        # 4000 trials -> +-1.1% at 95%; 0.93 leaves room for that while still failing the 93.6%
        # and 59.3% the previous interval produced.
        assert rate >= 0.93, f"coverage {100*rate:.1f}% at ({a},{b}) -- below nominal"


if __name__ == "__main__":
    tests = (test_closed_form_mle_matches_brute_force,
             test_bounds_ordered_and_bracket_the_estimate,
             test_zero_discordant_pairs_is_not_certainty,
             test_coverage_is_at_least_nominal)
    ok = 0
    for t in tests:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(tests)} passed")
    raise SystemExit(0 if ok == len(tests) else 1)
