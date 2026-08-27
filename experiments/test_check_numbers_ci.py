"""Pin check_numbers.py's paired-difference interval: the MLE, the bounds, and the COVERAGE.

check_numbers.py imports nothing third-party, so the interval it quotes for the M3 block is written
out by hand -- and hand-written statistics is where this file family has already been bitten twice:

  1. A Clopper-Pearson version had its bounds SWAPPED and ~0.4 points off, and produced a
     perfectly plausible green guard run.
  2. That same version, once un-swapped, was still not a valid interval for the MARGINAL rate
     difference: it conditioned on the number of discordant pairs and ignored the randomness in
     that number. Measured coverage 93.6% at one of the real splits and 59.3% at (2,0), with
     (0,0) coming out as the zero-width [0, 0] -- certainty from no information.

Neither was caught by eye. So this checks the thing that actually matters -- coverage -- by
ENUMERATING the multinomial the counts come from, not by sampling it: an earlier version simulated
4000 trials and could not separate 93.6% from 95%.

Run: `python experiments/test_check_numbers_ci.py` or `pytest`.
"""
import math
import re
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
    """Constrained MLE by direct maximisation -- the closed form must agree with it.

    A grid search, deliberately, not a library call: scipy has no Tango implementation, so there is
    no independent reference for the interval itself. The MLE is the piece that CAN be checked
    against something that shares none of its algebra, and the coverage test below checks the rest.
    """
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


# --------------------------------------------------------------------- rule injection regressions
# A rule that does not fire on its own trigger is worse than no rule: it reports a file as checked.
# Every rule added with the manuscript targets on 2026-08-26 was injection-tested by hand before it
# was written down, and these pin that testing so it survives the next edit to a pattern.
#
# The NEGATIVE cases matter at least as much. Three of them are real near-misses found while tuning
# these patterns against the actual files, not hypotheticals:
#   * a 200-character Jungfrau window reaches past \bottomrule into tab:realindex's footnote and
#     fires on the row that was just CORRECTED to 96%/1506/1563;
#   * `5\.8\s*%` without a (?<![\d.]) guard fires inside the cxidb-62 cell "105.8/105.8/75.5 A";
#   * `29` beside "throughput" fires on GLINT_REPORT.md's banner date "Snapshot: 2026-06-29."
# Each is trap (a) or a window-width variant of it, and each would have shipped as a false positive.
import importlib.util as _ilu
import sys as _sys

_spec = _ilu.spec_from_file_location("_cn", ROOT / "experiments" / "check_numbers.py")
_cn = _ilu.module_from_spec(_spec)
# Registered BEFORE exec: @dataclass resolves its own annotations through sys.modules[__module__],
# so a module executed outside it dies with a bare AttributeError on NoneType. Importing the file is
# safe -- only main() is guarded by __main__, and nothing at module scope runs the guard.
_sys.modules["_cn"] = _cn
_spec.loader.exec_module(_cn)


def _fires(text: str, rule_name: str) -> bool:
    """Does `rule_name` (and only rules at all) fire on this text? Uses the real scan()."""
    return any(f"/{rule_name}]" in f for f in _cn.scan(Path("probe.tex"), text))


# (rule name, text that MUST fire, text that must NOT)
INJECTIONS = [
    ("jungfrau-93pct",
     r"Jungfrau-4M & lysozyme & tetragonal & $93\%$ blind (support $54/60$) & --- \\",
     # the corrected row, plus enough trailing float to reach cxidb-45's legitimate 93% footnote
     "Jungfrau-4M & lysozyme & tetragonal & $96\\%$ blind ($1506/1563$); $95\\%$ final "
     "($1482/1563$) & --- \\\\\n\\bottomrule\n\\end{tabular}\n\n{\\footnotesize the $93\\%$ "
     "blind rate is measured on GLINT's own uncapped peak lists.}"),
    ("jungfrau-support-54-60",
     r"(consensus support $54/60$) and indexes $93\%$.",
     r"(consensus support $1506/1563$) and indexes $96\%$."),
    ("compare3-346-at-480",
     "480 frames of the same run leaves them indistinguishable (346 vs 350, p=0.70)",
     # the DRP page's microsecond range: a bare 346 with no 480 anywhere near it
     "cupy full path (346--1,136 us), 11-node captured graph (~22 us, flat)"),
    ("warmup-5.8pct",
     r"swept from 120 to 3000 frames the warm-up falls from $5.8\%$ of the run",
     r"cxidb-62: a $105.8/105.8/75.5$~\AA\ cell, warm-up a fixed five frames"),
    ("integ-fused-6-32x",
     r"fused GPU box-integration (frame resident) & 7.6 & 0.33 & 6--32$\times$ \\",
     r"fused GPU box-integration (frame resident) & 7.6 & 0.33 & 23$\times$ \\"),
    ("stream-band-120-at-480",
     r"the live path indexes 61--65\% of the 480-frame set",
     # both bands, each against its OWN denominator -- the shape glint.tex already uses
     "read 61--65\\% (73--78 of 120) against 76\\% (91/120).\n"
     "At n=480 the same arms read 67--69\\% (323--331 of 480)."),
    # ...and the SAME defect with an ordinary LaTeX hard wrap between the band and the count. The
    # window was [^.\n] until #154's review, so this case -- a caption reflowed by anyone but its
    # author -- walked straight past the rule. test_stream_band_rule_crosses_a_latex_hard_wrap
    # below pins that the OLD pattern misses it, which is what makes this entry a regression test
    # rather than one more example.
    ("stream-band-120-at-480",
     "\\caption{Streaming yield. The live path indexes 61--65\\%\n"
     "of the 480-frame set, against 76\\% offline}",
     # the correct two-band paragraph, wrapped the same way -- the period still ends the window
     "read 61--65\\% (73--78 of 120)\nagainst 76\\% (91/120).\n"
     "At n=480 the same arms\nread 67--69\\% (323--331 of 480)."),
    # The negative is the CORRECTLY QUALIFIED form -- the same 117/120, with the bar and the arm
    # named, which is precisely what the rule's own `instead` asks for. Until #154's review this
    # rule rejected it, so it had no satisfiable output: the only way to clear it was to delete a
    # true historical citation. The negative used to dodge that by renumbering to 115/120, which
    # tested the pattern's digits and not the thing the rule is about.
    ("consensus-117-of-120",
     "Consensus rescues the weak frames: ~71% -> 97% (117/120) blind on the same peaks.",
     "Consensus at the >=10-reflection gate, offline: 117 of 120 blind on the same peaks."),
    ("fps-29",
     "the blind pipeline sustains 29 frames/s on one A100",
     # the banner date that the first version of this pattern fired on
     "**Snapshot: 2026-06-29. The throughput figures here are superseded.**"),
    ("legacy-shots",
     "GPU + numba + coarse peak-finding: 4 -> 892 shots / s on one A100",
     # \b892\b would have accepted neither of these; (?<![\d.])892(?![\d.]) refuses both for the
     # stated reason. The run ID is the case that passed by luck for months.
     "runs mfxl1038923 r0278 and r0058, and a rate of 892.5 shots/s on the LEGACY bench"),
]


def test_every_new_rule_fires_on_its_own_trigger():
    for name, bad_text, _ in INJECTIONS:
        assert _fires(bad_text, name), f"{name} did NOT fire on its own retired value"


def test_no_new_rule_fires_on_corrected_text():
    for name, _, good_text in INJECTIONS:
        assert not _fires(good_text, name), f"{name} fired on text that is CORRECT -- false positive"


def test_rule_names_are_unique_and_carry_replacements():
    """A duplicate name would make the injection tests above check one rule twice and miss another;
    an empty `instead` leaves the reader with a complaint and no action."""
    names = [r.name for g in (_cn.RETIRED, _cn.OVERCLAIM, _cn.AMBIGUOUS) for r in g]
    assert len(names) == len(set(names)), f"duplicate rule name(s): {sorted({n for n in names if names.count(n) > 1})}"
    for name, _, _ in INJECTIONS:
        rule = next(r for r in _cn.RETIRED if r.name == name)
        assert rule.instead, f"{name} has no `instead` text"


def test_stream_band_rule_crosses_a_latex_hard_wrap():
    """The window must stop at a SENTENCE, not at a line.

    _normalize() folds spaces and tabs and deliberately keeps newlines (scan() maps match offsets
    to line numbers off the normalized text), so `[^.\\n]{0,80}` made the rule a function of where
    the editor's wrap landed. The defect it guards -- the Fig 9 caption pairing the 120-frame band
    with 480-frame counts -- lives in a caption, i.e. exactly the text most likely to be rewrapped
    by someone other than its author. This pins BOTH halves: the new pattern catches it, and the
    old one demonstrably did not.
    """
    wrapped = ("\\caption{Streaming yield. The live path indexes 61--65\\%\n"
               "of the 480-frame set, against 76\\% offline}")
    assert _fires(wrapped, "stream-band-120-at-480"), "the wrapped defect is not caught"
    old = re.compile(
        r"61\s*-{1,2}\s*65\s*\\?%[^.\n]{0,80}?(?<![\d.])(?:480|323|331|357)(?![\d.])"
        r"|(?<![\d.])(?:480|323|331|357)(?![\d.])[^.\n]{0,80}?61\s*-{1,2}\s*65\s*\\?%", re.I)
    assert not old.search(_cn._normalize(wrapped)), (
        "the OLD [^.\\n] window now catches this, so this case no longer tests the fix -- pick "
        "one it misses, or retire the assertion honestly")


def test_consensus_117_accepts_the_form_its_own_advice_requests():
    """A guard whose advice its own pattern refuses has no correct output.

    consensus-117-of-120 says "name the bar and the arm, e.g. '>=10-reflection gate, offline'" and
    then fired on that sentence, so the only way to clear it was to delete a true, correctly
    qualified historical citation. Each disambiguator is checked ALONE -- a `needs` tuple passes as
    soon as any one of its entries is nearby, so testing them together would hide a dead entry.
    """
    assert _fires("Consensus indexes 117/120 frames blind.", "consensus-117-of-120")
    for qualified in ("At the >=10-reflection gate the same pipeline reached 117/120.",
                      "At the $\\geq$10-reflection gate the same pipeline reached 117/120.",
                      "The offline hybrid reached 117 of 120 on that subset.",
                      "Quoted at the loose bar, this is 117/120."):
        assert not _fires(qualified, "consensus-117-of-120"), (
            f"the rule rejects a correctly labelled citation: {qualified!r}")


def test_guard_advice_is_not_itself_retired():
    """The `instead` of one rule must not be a value another rule retires.

    Not hypothetical: legacy-shots recommended "~29 shots/s", which is 1000/34 -- the reciprocal of
    the blind figure blind-34ms retires. Had it fired it would have walked an editor straight into
    a fresh violation. It now interpolates FACTS['blind_fps']; this keeps it that way.

    ⚑ This used to SKIP every rule carrying an `exempt`, which silently exempted four of them --
    legacy-shots among them, so the very defect the docstring describes could have come back as
    "892 shots/s" unnoticed (found in review of #154). The fix is not to re-implement the
    exemption logic here but to run the advice through the REAL scan(), which applies `exempt` and
    `needs` with the same _near() semantics a deliverable gets. A rule matching its OWN advice is
    still allowed: retiring a value means quoting it (see stream-band-120-at-480, whose advice
    ends "or keep 61--65% and quote it against 120").
    """
    for rule in _cn.RETIRED:
        if not rule.instead:
            continue
        tripped = [f for f in _cn.scan(Path("advice.txt"), rule.instead)
                   if "[RETIRED/" in f and f"/{rule.name}]" not in f]
        assert not tripped, f"{rule.name}'s advice {rule.instead!r} trips:\n" + "".join(tripped)


def test_advice_check_covers_the_exempt_rules():
    """...and the skip is gone for good: the four rules that carry an `exempt` are now evaluated.

    Pinned by construction rather than by inspection -- inject an advice string that only an
    exempted rule refuses, and require the check above to catch it.
    """
    exempted = [r.name for r in _cn.RETIRED if r.exempt]
    assert exempted, "no rule carries an exempt any more -- this test has nothing to protect"
    assert "legacy-shots" in exempted
    # scan() must still fire on the bare form. The real legacy-shots advice clears it through the
    # word LEGACY sitting in the same sentence, which is the exemption working, not being skipped.
    assert _fires("use ~892 shots/s instead", "legacy-shots")
    assert not _fires("mark LEGACY, or use ~39 shots/s", "legacy-shots")


# ------------------------------------------------------------------- REQUIRED (file invariants)
# The merge-quality FACTS were DECORATIVE until #154's review: jungfrau_ccstar / _rsplit_pct /
# _iovers and the pk45 trio sat in the table, commented at length, read by check_arithmetic's
# ordering guards -- and by nothing that looks at a deliverable. scan() only hunts RETIRED
# regexes, so tab:realmerge's cells could be edited to anything at all and every run stayed green.
#
# The probes below are BUILT FROM FACTS on both sides, so the tests move when the measurement
# does. The rendering is the one thing that cannot come from the table: FACTS stores CC* as a
# float, and 0.90 (cxidb-45, two decimals) round-trips through Python as "0.9" while 0.915
# (Jungfrau, three) does not. The deliverables print the two rows at different widths, so the
# widths live here and in the Required needles -- and the first test below is what keeps the two
# in step rather than letting them drift into a needle nothing can satisfy.
MERGE_RENDERED = {
    "jungfrau_ccstar":        f"{_cn.FACTS['jungfrau_ccstar']:.3f}",
    "jungfrau_rsplit_pct":    f"{_cn.FACTS['jungfrau_rsplit_pct']:g}",
    "jungfrau_iovers":        f"{_cn.FACTS['jungfrau_iovers']:g}",
    "jungfrau_blind_of1563":  f"{_cn.FACTS['jungfrau_blind_of1563']:d}",
    "jungfrau_final_of1563":  f"{_cn.FACTS['jungfrau_final_of1563']:d}",
    "jungfrau_frames_total":  f"{_cn.FACTS['jungfrau_frames_total']:d}",
    "pk45_ccstar":            f"{_cn.FACTS['pk45_ccstar']:.2f}",
    "pk45_rsplit_pct":        f"{_cn.FACTS['pk45_rsplit_pct']:g}",
    "pk45_iovers":            f"{_cn.FACTS['pk45_iovers']:g}",
}


def _required_fires(text: str, name: str) -> bool:
    """Does REQUIRED entry `name` fire on this text? Uses the real check_required()."""
    return any(f"[REQUIRED/{name}]" in f for f in _cn.check_required(Path("probe.tex"), text))


def _jungfrau_row(**edit) -> str:
    v = dict(MERGE_RENDERED, **edit)
    return (f"Jungfrau-4M lysozyme & {v['jungfrau_final_of1563']} & ${v['jungfrau_ccstar']}$ & "
            f"${v['jungfrau_rsplit_pct']}\\%$ & ${v['jungfrau_iovers']}$ & $99\\%$ complete, "
            f"$2.1$\\,\\AA \\\\")


def _pk45_row(**edit) -> str:
    v = dict(MERGE_RENDERED, **edit)
    return (f"cxidb-45 Proteinase~K & 290 & ${v['pk45_ccstar']}$ & ${v['pk45_rsplit_pct']}\\%$ & "
            f"${v['pk45_iovers']}$ & full lattice, $70.7\\times$ redundancy \\\\")


def _jungfrau_prose(**edit) -> str:
    v = dict(MERGE_RENDERED, **edit)
    return (f"\\paragraph{{Jungfrau-4M lysozyme images.}} GLINT indexes ${{96}}\\%$ blind "
            f"(${v['jungfrau_blind_of1563']}/{v['jungfrau_frames_total']}$); $95\\%$ final "
            f"(${v['jungfrau_final_of1563']}/{v['jungfrau_frames_total']}$). The resulting "
            f"{v['jungfrau_final_of1563']}-crystal set reaches $CC^{{*}}={v['jungfrau_ccstar']}$, "
            f"$R_{{\\mathrm{{split}}}}={v['jungfrau_rsplit_pct']}\\%$, "
            f"$\\langle I/\\sigma\\rangle={v['jungfrau_iovers']}$.")


def test_every_merge_fact_is_read_by_a_required_rule():
    """The finding itself: each of these nine FACTS must be something the guard can miss in a file.

    If a key drops out of REQUIRED it becomes a comment again, and a comment cannot fail.
    """
    needles = [rx for r in _cn.REQUIRED for rx in r._needles]
    for key, shown in MERGE_RENDERED.items():
        probe = f"cell {shown}\\% end"          # covers the bare needles and the `...\\?%` ones
        assert any(rx.search(probe) for rx in needles), (
            f"no REQUIRED needle matches FACTS[{key!r}] as the deliverables render it ({shown!r})")


def test_required_merge_rows_pass_on_the_measured_values():
    """A document carrying both widths' worth of numbers must be clean end to end.

    The rows are ALSO checked on their own against their own entry only -- a bare tab:realmerge
    row legitimately trips the whole-file entry, because a row does not state the 1506/1563 blind
    counts. That is the two widths doing different jobs, not a false positive.
    """
    assert not _required_fires(_jungfrau_row(), "jungfrau-merge-row")
    assert not _required_fires(_pk45_row(), "pk45-merge-row")
    for probe in (_pk45_row(), _jungfrau_prose(),
                  _jungfrau_row() + "\n" + _pk45_row() + "\n" + _jungfrau_prose()):
        assert not _cn.check_required(Path("probe.tex"), probe), (
            f"REQUIRED fires on text stating the measured values:\n{probe}")


def test_required_merge_rows_fire_on_an_edited_cell():
    """One cell, edited in one row -- the case the whole-file form cannot see.

    0.915 is written at three sites in the manuscript, so a whole-file "the value must appear
    somewhere" check stayed green when tab:realmerge's CC* cell alone was changed. Verified against
    the real file before the windowed entry was added.
    """
    for name, probe in (
            ("jungfrau-merge-row", _jungfrau_row(jungfrau_ccstar="0.925")),
            ("jungfrau-merge-row", _jungfrau_row(jungfrau_rsplit_pct="32.6")),
            ("jungfrau-merge-row", _jungfrau_row(jungfrau_iovers="8.7")),
            ("jungfrau-merge-row", _jungfrau_row(jungfrau_final_of1563="1483")),
            ("pk45-merge-row", _pk45_row(pk45_ccstar="0.91")),
            ("pk45-merge-row", _pk45_row(pk45_rsplit_pct="33")),
            ("pk45-merge-row", _pk45_row(pk45_iovers="8.8"))):
        assert _required_fires(probe, name), f"{name} did not fire on:\n{probe}"


def test_required_whole_file_form_catches_a_value_leaving_the_file():
    """...and the other width: a renumber that removes the value from the document entirely."""
    assert _required_fires(_jungfrau_prose(jungfrau_ccstar="0.925"), "jungfrau-merge-facts")
    assert _required_fires(_jungfrau_prose(jungfrau_blind_of1563="1500"), "jungfrau-merge-facts")
    assert _required_fires("On 907 readable cxidb-45 Proteinase K frames GLINT merges to "
                           "$CC^{*}=0.91$ with $R_{\\mathrm{split}}=31\\%$ and "
                           "$\\langle I/\\sigma\\rangle=7.8$.", "pk45-merge-facts")


def test_required_stays_silent_without_its_trigger():
    """The scoping half. A deck that names the DETECTOR or the PROTEIN but merges nothing must not
    be told to grow a merge table -- both near-misses are real lines from the real targets."""
    for probe in (
            # build_glint.py: the mfx r199 front-end card, and the DRP page's calib note
            'statcard(s,6.74,2.0,"86%","real crystal end to end\\nmfx r199 - jungfrau-16M",GREEN)',
            "`CalibGPUMultiGain` (bit-exact vs `det.calib`, max|D|=0, Jungfrau 1M/4M)",
            # build_pitch.py: Proteinase K indexing, no merge anywhere in the deck
            "Head-to-head on real Proteinase K: per frame classical DIALS leads (62%), but "
            "GLINT's consensus turns that around.",
            # the cover letter quotes the headline CC* without naming the dataset
            "reaching crystallographic merge quality ($CC^{*}=0.90$) on public data"):
        assert not _cn.check_required(Path("probe.tex"), probe), (
            f"REQUIRED fired on a file that merges nothing:\n{probe}")


# --------------------------------------------------------- arithmetic guards, by PERTURBATION
# These used to be re-implemented predicates: the test asserted 1482 <= 1506 <= 1563 itself rather
# than calling check_arithmetic(), so deleting the guard from check_numbers.py left the test green
# (found in review of #154). Every case below drives the REAL function with FACTS temporarily
# edited and requires the REAL message, so a deleted guard fails here.
def _arith(**overrides):
    """check_arithmetic() with FACTS perturbed, restored on the way out (including on failure)."""
    F = _cn.FACTS
    saved = {k: F[k] for k in overrides}
    try:
        F.update(overrides)
        return _cn.check_arithmetic()
    finally:
        F.update(saved)


# (what a careless edit does, the substring the guard must answer with)
ARITHMETIC_PERTURBATIONS = [
    ({"jungfrau_final_of1563": 1520},                 "must nest"),
    ({"jungfrau_blind_of1563": 1400},                 "jungfrau_blind_rate_pct"),
    ({"jungfrau_final_rate_pct": 90},                 "jungfrau_final_rate_pct"),
    ({"jungfrau_frames_total": 1600},                 "jungfrau_blind_rate_pct"),
    ({"jungfrau_ccstar": 0.94},                       "Jungfrau CC*"),
    ({"jungfrau_xg_ccstar": 0.900},                   "Jungfrau CC*"),
    ({"jungfrau_rsplit_pct": 36.0},                   "Jungfrau R_split"),
    ({"jungfrau_xg_rsplit_pct": 30.0},                "Jungfrau R_split"),
    # THE COLLAPSE BRANCH, from both sides -- it had never been exercised at all. Neither edit
    # trips any neighbouring guard (0.90 and 0.915 both still sit below xgandalf's 0.930), so each
    # one reaches this check and nothing else.
    ({"pk45_ccstar": 0.915},                          "collapsed into one number"),
    ({"jungfrau_ccstar": 0.90},                       "collapsed into one number"),
]


def test_check_arithmetic_is_green_on_the_shipped_table():
    bad, _ = _cn.check_arithmetic()
    assert not bad, "the shipped FACTS table contradicts itself:\n" + "\n".join(bad)


def test_each_arithmetic_guard_fires_when_its_facts_are_perturbed():
    for edit, expect in ARITHMETIC_PERTURBATIONS:
        bad, _ = _arith(**edit)
        assert any(expect in b for b in bad), (
            f"perturbing {edit} did not produce a failure mentioning {expect!r}; got:\n"
            + ("\n".join(bad) or "  (nothing at all -- the guard is gone)"))


BLIND_PAIR_CASES = [
    # (label, text, must_fire)
    ("prose form, one line", "GLINT indexes blind above xgandalf's rate (76% vs 71% at the same gate)", True),
    # THE REGRESSION: ordinary Markdown/LaTeX wrapping put a newline between the two numerals and
    # the first version of this rule used `[^.\n]`, so reflowing the guarded paragraph defeated it
    # -- the same defect the stream-band rule had at :824 (Copilot review of glint#160).
    ("prose form, WRAPPED", "GLINT indexes blind above xgandalf's rate (76% vs\n71% at the same gate)", True),
    ("report table row", "| xgandalf | blind | 71% | 11,542 | 0.087 |", True),
    ("table row, corrected", "| xgandalf | blind | 72% (86/120) | 11,542 | 0.087 |", False),
    # a builder's own comment RECORDING that it once shipped the pair is not a claim of it
    ("historical mention", '# one build behind -- it shipped 76%/71% and "~3 frames" for a week', False),
    # both numerals are still correct alone: 91/120 offline, and the 85/120 lattice-bar ceiling
    ("lone 76%, offline", "the offline hybrid reaches 76% (91/120) at the strict bar", False),
    ("lone 71%, ceiling", "Blind indexing saturates at ~71% gated on sparse cxidb", False),
]


def test_blind_pair_rules_fire_and_stay_silent():
    """The retired 76/71 blind pair, in every form the deliverables actually wrote it.

    This pair sat in README.md and GLINT_REPORT.md -- both GUARDED targets -- while the guard ran
    green, because the two rules that existed keyed on adjacencies neither file used: one needs
    `xgandalf` next to 71% (table pipes break it), the other needs 71% within 40 chars of GLINT
    (the README's phrasing is 45). Both directions are pinned here, because a rule that fires on
    everything is as useless as one that fires on nothing.
    """
    for label, text, must_fire in BLIND_PAIR_CASES:
        fired = any("blind-pair" in f for f in _cn.scan(Path("probe.md"), text))
        assert fired == must_fire, (
            f"{label}: fired={fired}, expected {must_fire} -- {text!r}")


def test_submission_files_are_default_targets():
    """The files that GO TO THE JOURNAL must be scanned by default -- all three of them.

    Nothing else here covers this. Every other test hands text to `scan`/`check_required`
    directly, so a target quietly dropped from DEFAULT_TARGETS leaves all of them green, and CI
    cannot notice either: the papers checkout lives outside the repo, so these entries are SKIPped
    on a runner and only the pre-push hook ever reads them. That combination is exactly how
    glint_SI.tex went unguarded until 2026-08-26 while carrying the retired Jungfrau
    `93% (support 54/60)` the main text had already been corrected out of -- the SI holds PARALLEL
    copies of the dataset tables, so guarding the manuscript alone just moves the drift one file
    over. Asserting the whole submission set, not only the file that drifted: the same argument
    covers each of them (Copilot review of glint#159).
    """
    for name in ("glint_rewrite_JAC_refined.tex", "glint_SI.tex", "cover_letter_JAC.tex"):
        assert any(p.name == name for p in _cn.DEFAULT_TARGETS), (
            f"{name} is not in DEFAULT_TARGETS -- it would go to the journal unguarded")


def test_perturbations_are_restored():
    """A leaked override would make every later test run against a table nobody measured."""
    for edit, _ in ARITHMETIC_PERTURBATIONS:
        _arith(**edit)
    bad, _ = _cn.check_arithmetic()
    assert not bad, "FACTS was left perturbed:\n" + "\n".join(bad)


def _main_quiet(argv, *, default_targets=None, pdf_targets=None) -> int:
    """Run the real main() with module targets patched and its chatter swallowed.

    These exist because the missing/unreadable-target behavior was only ever verified by MANUAL
    injection (documented in #168's PR body) -- the committed suite never called main(), so a
    refactor could silently restore green-on-missing (Copilot review of #168, round 2).
    """
    import contextlib
    import io
    saved_d, saved_p = _cn.DEFAULT_TARGETS[:], _cn.PDF_TARGETS[:]
    try:
        if default_targets is not None:
            _cn.DEFAULT_TARGETS[:] = default_targets
        if pdf_targets is not None:
            _cn.PDF_TARGETS[:] = pdf_targets
        with contextlib.redirect_stdout(io.StringIO()):
            return _cn.main(["check_numbers.py", *argv])
    finally:
        _cn.DEFAULT_TARGETS[:] = saved_d
        _cn.PDF_TARGETS[:] = saved_p


def test_main_fails_on_missing_requested_targets():
    """An explicit path and a --pdf entry that do not exist must FAIL, not skip."""
    assert _main_quiet(["/nonexistent/nowhere.md"]) == 1
    assert _main_quiet(["--pdf"], default_targets=[_cn.REPO / "docs/lineage.md"],
                       pdf_targets=[Path("/nonexistent/deck.pdf")]) == 1


def test_main_fails_on_missing_or_unreadable_in_repo_defaults():
    """A renamed in-repo docs page, or one replaced by a directory, is a failure in the default
    run -- while an absent OUT-of-repo default stays a skip, which is what keeps the bare CI
    runner green (it has no ~/git/papers)."""
    ok = _cn.REPO / "docs/lineage.md"
    assert _main_quiet([], default_targets=[ok, _cn.REPO / "docs/zz-renamed-away.md"]) == 1
    assert _main_quiet([], default_targets=[ok, _cn.REPO / "docs"]) == 1, (
        "a directory at a requested target path passed as if it were guarded")
    assert _main_quiet([], default_targets=[ok, Path("/nonexistent/outside/papers.tex")]) == 0


if __name__ == "__main__":
    tests = (test_closed_form_mle_matches_brute_force,
             test_bounds_ordered_and_bracket_the_estimate,
             test_zero_discordant_pairs_is_not_certainty,
             test_coverage_is_at_least_nominal,
             test_every_new_rule_fires_on_its_own_trigger,
             test_no_new_rule_fires_on_corrected_text,
             test_rule_names_are_unique_and_carry_replacements,
             test_stream_band_rule_crosses_a_latex_hard_wrap,
             test_consensus_117_accepts_the_form_its_own_advice_requests,
             test_guard_advice_is_not_itself_retired,
             test_advice_check_covers_the_exempt_rules,
             test_every_merge_fact_is_read_by_a_required_rule,
             test_required_merge_rows_pass_on_the_measured_values,
             test_required_merge_rows_fire_on_an_edited_cell,
             test_required_whole_file_form_catches_a_value_leaving_the_file,
             test_required_stays_silent_without_its_trigger,
             test_blind_pair_rules_fire_and_stay_silent,
             test_submission_files_are_default_targets,
             test_check_arithmetic_is_green_on_the_shipped_table,
             test_each_arithmetic_guard_fires_when_its_facts_are_perturbed,
             test_perturbations_are_restored,
             test_main_fails_on_missing_requested_targets,
             test_main_fails_on_missing_or_unreadable_in_repo_defaults)
    ok = 0
    for t in tests:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(tests)} passed")
    raise SystemExit(0 if ok == len(tests) else 1)