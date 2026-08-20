"""Pin check_numbers.py's hand-rolled Clopper-Pearson against a reference.

check_numbers.py imports nothing third-party, so its exact CI on the McNemar rate difference is
written out by hand -- and the first version had the bounds SWAPPED and ~0.4 percentage points off
while looking perfectly reasonable in the guard's output. A statistic nobody checks is a decoration,
which is the same failure this whole file family exists to prevent.

Checks against scipy.stats.beta when it is importable, and against hard-coded reference values
(computed from that same scipy, recorded here) when it is not -- so the CPU CI box, which installs
no scipy... actually installs scipy for glint itself, but the fallback keeps this meaningful
anywhere. Run: `python experiments/test_check_numbers_ci.py` or `pytest`.
"""
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "experiments" / "check_numbers.py").read_text(encoding="utf-8")

# Lift the three helpers out of check_arithmetic without importing the module (importing it runs
# the whole guard). They are self-contained and depend only on math.comb.
_body = SRC[SRC.index("    def _cp_lower("):SRC.index("    _N480, _plateau_lo")]
_NS: dict = {}
exec("from math import comb\n" + "\n".join(l[4:] for l in _body.split("\n")), _NS)   # noqa: S102
ci_diff = _NS["_ci_diff"]

# (n01, n10) -> (lo, hi) on the RATE difference at n=480, from scipy.stats.beta.
REFERENCE = {
    (32, 21): (-0.008818165608042, 0.052001404880143),
    (29, 54): (-0.087160915543348, -0.013168534873971),
    (29, 27): (-0.027934470932826, 0.035800408057592),
    (10, 29): (-0.060064225017146, -0.012789504524963),
    (17, 12): (-0.013372399702448, 0.031986069392505),
    (22, 23): (-0.030557734029390, 0.026669301463731),
    (18, 20): (-0.030113456086110, 0.022449845181242),
    (0, 5): (-0.010416666666667, 0.000450863265196),
    (7, 0): (0.002637918100629, 0.014583333333333),
}


def test_ci_matches_scipy_or_reference():
    try:
        from scipy.stats import beta                                    # noqa: PLC0415

        def ref(n01, n10, n=480, alpha=0.05):
            m = n01 + n10
            lo = 0.0 if n01 == 0 else beta.ppf(alpha / 2, n01, m - n01 + 1)
            hi = 1.0 if n01 == m else beta.ppf(1 - alpha / 2, n01 + 1, m - n01)
            return m * (2 * lo - 1) / n, m * (2 * hi - 1) / n
        src = "scipy"
    except ImportError:
        def ref(n01, n10, n=480, alpha=0.05):
            return REFERENCE[(n01, n10)]
        src = "recorded reference"

    worst = 0.0
    for (n01, n10), _ in REFERENCE.items():
        got, want = ci_diff(n01, n10, 480), ref(n01, n10)
        worst = max(worst, abs(got[0] - want[0]), abs(got[1] - want[1]))
    assert worst < 1e-9, f"CI disagrees with {src} by {worst:.3g}"


def test_bounds_are_ordered_and_bracket_the_point_estimate():
    """The swapped-bounds bug produced lo > hi and still read plausibly. Pin the ordering, and
    that the interval actually contains the observed difference."""
    for (n01, n10) in REFERENCE:
        lo, hi = ci_diff(n01, n10, 480)
        point = (n01 - n10) / 480.0
        assert lo <= hi, f"({n01},{n10}): bounds inverted, {lo} > {hi}"
        assert lo <= point <= hi, f"({n01},{n10}): CI [{lo}, {hi}] excludes the estimate {point}"


def test_no_discordant_pairs_is_a_zero_width_interval():
    assert ci_diff(0, 0, 480) == (0.0, 0.0)


if __name__ == "__main__":
    tests = (test_ci_matches_scipy_or_reference,
             test_bounds_are_ordered_and_bracket_the_point_estimate,
             test_no_discordant_pairs_is_a_zero_width_interval)
    ok = 0
    for t in tests:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(tests)} passed")
    raise SystemExit(0 if ok == len(tests) else 1)
