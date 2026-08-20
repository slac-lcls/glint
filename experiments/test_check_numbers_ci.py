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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "experiments" / "check_numbers.py").read_text(encoding="utf-8")

# Lift the helpers out of check_arithmetic without importing the module (that runs the whole guard).
_BEG = "# --- BEGIN paired-difference interval"
_END = "# --- END paired-difference interval"
if _BEG not in SRC or _END not in SRC:                 # the markers are the contract; without them
    raise SystemExit("check_numbers.py no longer marks the interval block -- this test would "
                     "silently extract the wrong code, so it refuses to run")
_body = SRC[SRC.index(chr(10), SRC.index(_BEG)) + 1:SRC.index(_END)]
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


def _exact_coverage(p01: float, p10: float, n: int, w: int = 9) -> "tuple[float, float]":
    """EXACT coverage by enumerating the multinomial, not by sampling.

    The previous version of this test simulated 4000 trials and accepted >= 93%. At 4000 trials a
    true 93.6% usually clears 93%, so it could not reliably separate the broken interval (93.67%
    here) from a correct one (95.06%) -- it only ever caught the old code through the sparse (2,0)
    case. Enumeration removes the sampling noise entirely, so the threshold can sit between the
    two without a power argument.

    Returns (covered mass, mass outside the enumerated window) so the truncation is reported rather
    than assumed; the window is +-w sd, which leaves ~1e-13 unaccounted.
    """
    true = p01 - p10
    lg = math.lgamma
    m0, s0 = n * p01, math.sqrt(n * p01 * (1 - p01)) + 1
    m1, s1 = n * p10, math.sqrt(n * p10 * (1 - p10)) + 1
    xs = range(max(0, int(m0 - w * s0)), min(n, int(m0 + w * s0)) + 1)
    ys = range(max(0, int(m1 - w * s1)), min(n, int(m1 + w * s1)) + 1)
    NEG = -1e18                                       # log 0: a cell that cannot occur
    l0 = math.log(p01) if p01 > 0 else NEG
    l1 = math.log(p10) if p10 > 0 else NEG
    lr = math.log(1 - p01 - p10)
    cov = tot = 0.0
    for x in xs:
        for y in ys:
            if x + y > n:
                continue
            lp = (lg(n + 1) - lg(x + 1) - lg(y + 1) - lg(n - x - y + 1)
                  + (x * l0 if x else 0.0) + (y * l1 if y else 0.0) + (n - x - y) * lr)
            if lp < -700:                             # underflows to 0 anyway
                continue
            pm = math.exp(lp)
            tot += pm
            lo, hi = ci_diff(x, y, n)
            if lo <= true <= hi:
                cov += pm
    return cov, 1.0 - tot


def test_coverage_is_at_least_nominal():
    """The property both previous versions failed, checked exactly.

    Measured with this enumeration, the retired conditional interval scores 93.67% at (10,29) and
    59.35% at (2,0) -- the second being the sparse regime several hybrid arms of the real sweep sit
    in. Tango scores 95.06% and 98.37%. The bar is 94.5%: above anything the broken interval
    achieves at (10,29), below what a valid one does, and with no sampling noise to argue about.
    """
    for (a, b) in ((10, 29), (32, 21), (29, 54), (2, 0)):
        cov, neglected = _exact_coverage(a / N, b / N, N)
        assert abs(neglected) < 1e-9, f"({a},{b}): {neglected:.1e} of the mass fell outside the window"
        assert cov >= 0.945, f"coverage {100*cov:.2f}% at ({a},{b}) -- below nominal"


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
