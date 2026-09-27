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



def test_bare_86_of_120_needs_to_say_whose_it_is():
    """86/120 is xgandalf's blind rate AND GLINT's oracle-reachable ceiling. The rule must fire on an
    unlabelled one, stay silent on either labelling, and -- the trap that caught its own first draft
    -- not skip a value that ends a sentence: (?![\\d.]) rejects ANY following period, so
    "below XGANDALF's 86/120." went unseen on the one line that disambiguates it correctly."""
    fires = lambda t: _fires(t, "bare-86-of-120")
    assert fires("the counts (92/120 and 86/120) are reproduced")
    assert fires("GLINT reaches 86 of 120 frames")
    for ok in ("below XGANDALF's 86/120.",                 # sentence-final, labelled
               "the oracle-reachable ceiling is 86/120",
               "xgandalf & blind & 72\\% (86/120) \\\\",
               next(r.instead for r in _cn.AMBIGUOUS if r.name == "bare-86-of-120")):   # never flag its own advice
        assert not fires(ok), ok
    for no in ("a value of 86.120 in the fit", "frames 186/120 nonsense"):
        assert not fires(no), no


def test_fused_row_91_pairs_the_pipeline_yield_with_the_engine_speed():
    """The submitted Table 2 carries 'GLINT-(1) (batched) 76% (91/120) ... 0.17 ... 5900' (fixed in R1): the known-cell
    PIPELINE's yield (91/120, the 32 ms row) on the fused ENGINE's timing (whose yield is 80/120). The rule must
    fire on that row and on prose pairing 91/120 with 0.17, and stay silent on the corrected row, on the
    pipeline's own 32 ms row, and on 91/120 in prose that carries no fused timing."""
    fires = lambda t: _fires(t, "fused-row-91")
    assert fires("GLINT-\\textcircled{1} (batched)$^{\\ddagger}$ & known-cell & $76\\%$ (91/120) & --- & "
                 "\\textbf{0.17} & \\textbf{5900} \\\\")
    assert fires("the known-cell engine indexes 91/120 frames at 0.17 ms per frame")
    assert fires("91/120 at $0.17$~ms per frame")
    for ok in ("\\rev{GLINT (fused known-cell)}$^{\\ddagger}$ & known-cell & \\rev{$67\\%$ (80/120)} & --- & "
               "\\textbf{0.17} & \\textbf{5900} \\\\",
               "GLINT-\\textcircled{1} & known-cell & $76\\%$ (91/120) & --- & 32 & 31 \\\\",
               "supplying the reference cell directly to the same registration path gives 91/120 frames",
               "a value of 91/120 and a batch of 0.175 ms",
               "The pipeline indexes 91/120 frames; its comparison has p=0.17",   # a non-timing 0.17
               "91/1200 frames at 0.17 ms"):                                      # not 91/120
        assert not fires(ok), ok


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
    # fps-29's lookahead is same-line by design (the ISO-date trap in its note), so the deck
    # statcard -- unit label on the source line BEFORE the bare quoted numeral -- walked past it.
    # The negative is the SAME statcard carrying the corrected 39.
    ("fps-29-statcard",
     '("Throughput  (frames / s, one A100)",{"size":13,"bold":True,"color":ICE})]),\n'
     '    ("29",{"size":52,"bold":True,"color":TEAL}),',
     '("Throughput  (frames / s, one A100)",{"size":13,"bold":True,"color":ICE})]),\n'
     '    ("39",{"size":52,"bold":True,"color":TEAL}),'),
    # The negative is the COMMENTED mark -- the legitimate parking spot for reviewer feedback,
    # which is exactly the shape the submission tex ships at :240.
    ("hl-rendering",
     "\\hl{What is the definition of acceptance rate?}",
     "text before the note % \\hl{What is the definition of acceptance rate?}"),
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
        # all three groups, not just RETIRED: every injected rule HAPPENED to be retired until
        # hl-rendering (OVERCLAIM) joined, and the RETIRED-only lookup died with StopIteration.
        rule = next(r for g in (_cn.RETIRED, _cn.OVERCLAIM, _cn.AMBIGUOUS) for r in g
                    if r.name == name)
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


def test_fps29_statcard_covers_the_deck_shape_fps29_cannot_see():
    """The companion exists BECAUSE fps-29 is same-line by construction.

    fps-29's lookahead is [^\\n]{0,60}, kept that way so GLINT_REPORT.md's banner date cannot
    collide (its own note). The deck statcard inverts the geometry -- the unit label sits on the
    source line BEFORE the bare quoted numeral -- so this pins all four sides at once: the old
    rule demonstrably misses the statcard (else the companion is dead weight and the two should
    be folded), the companion catches it, the ISO-date banner trap that shaped fps-29 stays
    silent, and the corrected 39 statcard stays silent.
    """
    card = ('("Throughput  (frames / s, one A100)",{"size":13,"bold":True,"color":ICE})]),\n'
            '    ("29",{"size":52,"bold":True,"color":TEAL}),')
    assert _fires(card, "fps-29-statcard"), "the statcard shape is not caught"
    assert not _fires(card, "fps-29"), (
        "fps-29 now sees across the newline, so the companion no longer tests anything -- fold "
        "the two rules or retire this assertion honestly")
    banner = "**Snapshot: 2026-06-29. The throughput figures here are superseded.**"
    assert not _fires(banner, "fps-29-statcard"), "the ISO-date banner trap regressed"
    assert not _fires(card.replace('("29"', '("39"'), "fps-29-statcard"), \
        "fired on the corrected 39 statcard"


def test_hl_rendering_fires_on_marks_and_spares_comments_and_the_macro_def():
    """A rendering \\hl{ must fail the guard; a commented one and the \\newcommand must not.

    The refusals are load-bearing, not politeness: the submission tex legitimately carries the
    \\newcommand{\\FIXME}[1]{\\hl{...}} definition (:46) and a %-commented reviewer note (:240),
    so a rule without both escapes has no satisfiable green state on the very file it guards.
    The \\newcommand refusal is a lookahead INSIDE the pattern rather than exempt=("newcommand",)
    because the window/clause exempts reach across newlines -- a real mark on the line after the
    definition would have been exempted by its neighbour.
    """
    assert _fires("\\hl{draft}", "hl-rendering"), "line-start mark not caught"
    assert _fires("text \\hl{x}", "hl-rendering"), "mid-line mark not caught"
    assert _fires("Accuracy: 95\\% \\hl{verify}", "hl-rendering"), \
        "escaped percent was mistaken for a comment"
    assert not _fires("% \\hl{x}", "hl-rendering"), "fired on a commented mark"
    assert not _fires("text % \\hl{x}", "hl-rendering"), "fired past a trailing %"
    assert not _fires("\\newcommand{\\FIXME}[1]{\\hl{[FIXME: #1]}}", "hl-rendering"), \
        "fired on the macro-definition line"
    # ...and the corridor case the lookahead exists for: a REAL mark on the line right after the
    # definition must still fire, which exempt=("newcommand",) would have suppressed.
    both = "\\newcommand{\\FIXME}[1]{\\hl{[FIXME: #1]}}\n\\hl{draft}"
    assert _fires(both, "hl-rendering"), "a mark adjacent to the macro def was exempted"


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


def test_no_rule_text_anywhere_is_itself_retired():
    """`instead` AND `exempt`, across every rule class -- not just RETIRED's `instead`.

    The test above covers `RETIRED[*].instead`. That left the exemption markers unchecked, and they
    were occupied: TWO of `live-merge.exempt`'s eight markers were "20x short" / "20× short", which
    live-gap-20x retires. An author reaching for the guard's own sanctioned escape hatch therefore
    committed a fresh violation. `khz-demonstrated.instead` carried the same stale figure.

    ⚑ `why` is deliberately NOT checked. A why-string's job is often to NAME the retired value and
    its predecessor -- "179 f/s is the reciprocal of the retired 5.58 ms" is correct prose that has
    to keep quoting both. Only text the author is told to WRITE can walk them into a violation:
    `instead` (what to write instead) and `exempt` (what to write to be excused). Checking `why`
    here too was tried first and flagged six legitimate explanations, which is how this scope was
    settled. `why` strings still must not go STALE -- live-merge's quoted a superseded rate and was
    rewritten to interpolate FACTS -- but that is a correctness duty, not something this test can
    enforce without failing on correct prose.

    The general lesson: the guard's own prose is a deliverable. Anything it tells an author to
    write must survive being written. Interpolate from FACTS rather than quoting a numeral and this
    test stays quiet by construction.
    """
    classes = [("RETIRED", _cn.RETIRED), ("OVERCLAIM", _cn.OVERCLAIM), ("AMBIGUOUS", _cn.AMBIGUOUS)]
    bad = []
    for cls, rules in classes:
        for rule in rules:
            # ⚑ ALL THREE exemption channels, not just `exempt`: scan() also honours
            # `exempt_before` and `exempt_after` (both live on nstar-stale-32), and a retired
            # phrase added to either would recreate exactly this failure while the test stayed
            # green. Enumerating the fields by name is deliberate -- a new escape hatch should
            # break this test loudly rather than be silently uncovered.
            # (Found by Copilot in review of #179.)
            fields = [("instead", rule.instead)]
            for chan in ("exempt", "exempt_before", "exempt_after"):
                fields += [(f"{chan}[{i}]", e)
                           for i, e in enumerate(getattr(rule, chan, ()) or ())]
            for field, text in fields:
                if not text:
                    continue
                # A rule matching its OWN advice stays allowed, for the reason the test above gives:
                # retiring a value sometimes means naming it.
                tripped = [f for f in _cn.scan(Path("advice.txt"), text)
                           if "[RETIRED/" in f and f"/{rule.name}]" not in f]
                if tripped:
                    bad.append(f"{cls}/{rule.name}.{field} = {text!r}\n" + "".join(tripped))
    assert not bad, "guard prose trips the guard:\n\n" + "\n".join(bad)


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
    ({"fused_strict_of120": 81},                      "fused_strict_of120"),
    ({"indexing_rate": "79/115"},                     "fused_strict_of120"),
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
    # THE GATE SOURCE-TIE, one perturbation per constant: the tie compares FACTS to the shipped
    # glint_fast source, so perturbing FACTS away from the source must produce the drift message.
    # Without these, deleting the tie loop left both the unit suite and the guard green -- and
    # preventing exactly that drift is the feature (Copilot review of #170, round 2).
    ({"gate_tol": 0.16},                              "gate_tol"),
    ({"gate_frac": 0.26},                             "gate_frac"),
    ({"gate_min": 12},                                "gate_min"),
    # THE S16 GUARDS, one perturbation per branch -- they were mutation-verified by hand when
    # added and never encoded, so removing any of them left the committed suite green (Copilot
    # review of #163, round 2). The 400.9 case is the int()-truncation trap specifically: the
    # identity must fire on a fractional per-N count, not truncate it to a pass.
    ({"subset_draws_total": 3250},                    "subset_draws_total"),
    ({"subset_draws_per_n": 400.9},                   "subset_draws_total"),
    # seeds was the one input of the draw identity never perturbed -- a literal 8 in place of the
    # FACTS lookup left every case above green (Copilot review of #163, round 4)
    ({"subset_seeds": 7},                             "subset_draws_total"),
    ({"recov_r0278_n12_pct": 90.4},                   "would be 12"),
    # EXACTLY the bar, and a FRACTIONAL N: these pin the two operators the implementation
    # promises. 90.4 fires under both `>= 90` and `> 90`, so it cannot catch the boundary being
    # loosened; 90.0 can. And 32/24 fire whether or not the comparison truncates through int(),
    # so only a fractional N catches a reintroduced int() (Copilot review of #163, round 9).
    ({"recov_r0278_n12_pct": 90.0},                   "AT OR ABOVE"),
    ({"nstar_r0278": 16.4},                           "moved without the measurement"),
    ({"nstar_r0278": 32},                             "moved without the measurement"),
    ({"n_cross90_r0058": 24},                             "moved without the measurement"),
    ({"recov_r0278_n16_pct": 88.0},                   "BELOW the 90% bar"),
    ({"recov_r0058_n16_pct": 88.0},                   "BELOW the 90% bar"),
    # THE SI S17 GUARDS, one perturbation per branch. Same history as the S16 block above: I
    # mutation-verified these by hand when adding them and did not encode it, so deleting any of
    # them would have left the committed suite green (Copilot review of #179, round 2). The lesson
    # keeps costing the same amount, so it is now written down twice.
    #
    # Both indexers, because the XGANDALF arms were stored and then never read -- a drift to
    # roibin_nominal_xgd = 400 passed check_arithmetic() silently until this round.
    ({"roibin_p_nominal": 0.55},                      "the p-value and the counts disagree"),
    ({"roibin_p_xgd_nominal": 0.20},                  "the p-value and the counts disagree"),
    ({"roibin_nominal_xgd": 400},                     "XGANDALF nominal"),
    ({"roibin_disc_xgd_tight": 31},                   "opposite parity"),
    ({"roibin_disc_tight": 54},                       "opposite parity"),
    # the delta-exceeds-its-own-discordance branch: 245 gives delta -101 against discordance 61,
    # which overshoots while keeping (disc + delta) EVEN, so the parity branch stays silent and
    # this perturbation actually exercises the bound. (246 trips parity too and would pass this
    # assertion without ever proving the bound works.) The negative-flip-count branch does also
    # speak, which is correct -- an overshoot implies a negative count -- so this pins two
    # consequences of one inconsistency, not two independent guards.
    ({"roibin_aggr_glint": 245},                      "exceeds its own discordant count"),
    # the flip split: sum, then the SIGNED identity the sum alone cannot pin (halves swapped)
    ({"roibin_flip_down": 34},                        "no longer sums"),
    ({"roibin_flip_up": 35, "roibin_flip_down": 22},  "implies a rate change"),
    ({"roibin_nominal_glint": 333},                   "S17's sign is wrong"),
    ({"roibin_lo_cmphi": 50},                         "denoising result has inverted"),
    ({"roibin_sigma_q1": 0.99},                       "quartiles do not bracket"),
    ({"roibin_sigma_q3": 0.70},                       "quartiles do not bracket"),
    # THE SINGLE-FRAME FRONT END and the EXTREME-VALUE FLOOR, added 2026-09-01 with the two FACTS
    # blocks themselves, one perturbation per branch. Both blocks exist because their numbers
    # reached the submission unguarded: tab:negatives' "69% (best)" reconciled with no other rate
    # in the paper (triage 153/268), and "97% lie above the +3sigma null level" reported the
    # above-FLOOR count against the above-3sigma threshold (triage 216).
    ({"sf_strict_rate_pct": 67},                      "sf_strict_rate_pct"),
    ({"sf_lattice_rate_pct": 70},                     "sf_lattice_rate_pct"),
    ({"sf_selmiss_rate_pct": 9},                      "sf_selmiss_rate_pct"),
    ({"sf_genmiss_rate_pct": 25},                     "sf_genmiss_rate_pct"),
    ({"sf_negatives_lattice_pct": 72},                "sf_negatives_lattice_pct"),
    ({"sf_lattice_of120": 70, "sf_lattice_rate_pct": 58},  "looser bar cannot pass fewer"),
    # the partition, with its percentage moved TOGETHER so the pct tie above cannot claim the
    # catch -- otherwise this case proves the wrong guard (which is how the two were conflated
    # when they were mutation-checked by hand).
    ({"sf_genmiss_of120": 30, "sf_genmiss_rate_pct": 25},  "partition the 120 frames"),
    # ...and NOT an identity: 79 + 8 = 87 against a printed ceiling of 86 at HEAD, so a
    # solved+selmiss check here would fail on the shipped measurement. Pinned as a NEGATIVE below
    # in test_oracle_ceiling_is_not_asserted_as_a_sum.
    ({"floor_above_pct": 97},                         "floor_above_pct"),
    ({"floor_above_3sd_pct": 80},                     "floor_above_3sd_pct"),
    ({"floor_at_K": 60.0},                            "floor_at_K = floor_fit_a"),
    # Copilot (#184): close() is relative, so 50.2 against 50.645 (0.9%) passed a tol=0.01 check.
    ({"floor_at_K": 50.2},                            "floor_at_K = floor_fit_a"),
    ({"floor_fit_b": 6.0},                            "floor_at_K = floor_fit_a"),
    ({"floor_K": 500},                                "floor_at_K = floor_fit_a"),
    ({"floor_measured_at_K": 70.0},                   "come apart"),
    ({"floor_mean_sigmas": 5.0},                      "right-skewed"),
    # the threshold ORDERING, percentage moved together for the same reason as the partition case
    ({"floor_above_of80": 50, "floor_above_pct": 62}, "lower bar cannot pass fewer"),
    # INTEGRALITY of the frame counts. Without these the int() coercions in the partition and the
    # two percentage loops truncated a fractional edit into a pass -- sf_strict_of120 = 79.9 left
    # every guard green (#184 review, suppressed comment), the same fail-open the
    # subset_draws_per_n = 400.9 case pins one block up. One per denominator family.
    ({"sf_strict_of120": 79.9},                       "must be integral"),
    ({"floor_above_of80": 78.5},                      "must be integral"),
    ({"sf_lattice_of120": 121, "sf_lattice_rate_pct": 101}, "must satisfy 0 <= count <= 120"),
    ({"floor_above_of80": 81, "floor_above_pct": 101}, "must satisfy 0 <= count <= floor_real_frames"),
    ({"floor_real_frames": 0},                        "must be positive"),
    ({"floor_null_pool": 0},                          "must be positive"),
    # ...and the oracle ceiling's real bounds. sel_miss <= ceiling <= solved + sel_miss: the upper
    # bound is NOT an equality (HEAD prints 86 against 79+8=87) so both ends are exercised from
    # outside, not by nudging toward the measured value.
    ({"sf_ceiling_of120": 95},                        "solved + selection-miss"),
    ({"sf_ceiling_of120": 5},                         "solved + selection-miss"),
]


GLINT_FAST = ROOT / "glint" / "glint_fast.py"

# (source-mutation, expected fragment of the failure) -- the gate tie's FUNCTIONAL half.
# ARITHMETIC_PERTURBATIONS moves FACTS and proves the tie notices; these move the SHIPPED SOURCE
# and prove the tie notices that too, which is the half a FACTS perturbation cannot reach:
# re-inlining a literal into matched()'s default or a gpass() operand leaves FACTS and the
# declarations in perfect agreement while the shipped gate diverges from the published one
# (Copilot review of #170, round 4). The expression case belongs here for the same reason: the
# tie's regex read only a numeric PREFIX, so "GATE_TOL = 0.15 + 0.01" parsed as 0.15 (round 3).
GATE_SOURCE_MUTATIONS = [
    ("GATE_TOL = 0.15  ", "GATE_TOL = 0.15 + 0.01  ", ("GATE_TOL", "could not be located")),
    ("def matched(M, q, tol=GATE_TOL):", "def matched(M, q, tol=0.15):",
     ("matched()", "tol default", "GATE_TOL")),
    ("m = matched_strict(M, q)", "m = matched(M, q)", ("gpass(", "matched_strict")),
    # The RETURN EXPRESSION, which a regex over the source could not distinguish from the same
    # word in the docstring above it (Copilot review of #170, round 5) -- the tie reads the
    # parsed body now, so this severing is visible.
    ("return int((np.abs(r).max(1) < GATE_TOL).sum())",
     "return int((np.abs(r).max(1) < 0.15).sum())", ("matched_strict(", "GATE_TOL")),
    # The refactor that defeats a NAME-PRESENCE check: reference the constant, then compare
    # against a literal. Only inspecting the comparison itself sees it.
    ("    return int((np.abs(r).max(1) < GATE_TOL).sum())",
     "    tol = GATE_TOL\n    return int((np.abs(r).max(1) < 0.16).sum())",
     ("matched_strict(", "GATE_TOL")),
    (">= GATE_FRAC)", ">= 0.25)", ("gpass(", "GATE_FRAC")),
    (">= GATE_MIN)",  ">= 10)",   ("gpass(", "GATE_MIN")),
]


def test_gate_tie_catches_a_severed_functional_use():
    """Move the SHIPPED SOURCE, not FACTS, and the tie must still fail.

    Restored in a finally: a failure here must not leave a mutated glint_fast.py behind for the
    rest of the suite (or the working tree).
    """
    original = GLINT_FAST.read_text(encoding="utf-8")
    try:
        for old, new, expect in GATE_SOURCE_MUTATIONS:
            assert old in original, f"anchor vanished from glint_fast.py: {old!r}"
            GLINT_FAST.write_text(original.replace(old, new, 1), encoding="utf-8")
            bad, _ = _cn.check_arithmetic()
            # TOKENS, not a prose fragment. Matching the guard's sentence verbatim made this
            # test fail three separate times today purely because the MESSAGE was reworded while
            # the check kept working -- a red build that says nothing about the code under test.
            # The tokens are what the failure must identify: which function, and which constant.
            assert any(all(tok in b for tok in expect) for b in bad), (
                f"severing {old!r} produced no failure naming all of {expect!r}; got:\n"
                + ("\n".join(bad) or "  (nothing at all -- the functional check is gone)"))
    finally:
        GLINT_FAST.write_text(original, encoding="utf-8")
    bad, _ = _cn.check_arithmetic()
    assert not any("gate" in b.lower() for b in bad), "glint_fast.py was not restored cleanly"


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


def test_superseded_files_are_exempt_from_required_rules_only():
    """A self-declared SUPERSEDED deliverable owes no ADDED text, but still may not QUOTE a stale one.

    `glint_SI.tex` carries the superseded banner and remains on DEFAULT_TARGETS so stale values in
    that deliverable still get caught by QUOTE rules. A REQUIRED rule inverts that: the only ways
    to satisfy one are to edit a file the banner freezes, or to leave the guard permanently red
    (#184 review, where `negatives-69-is-the-lattice-bar` fired on glint_SI.tex's `69% (best)`
    cell). So REQUIRED is scoped off by the banner -- and this test pins that the exemption is
    that narrow, because an exemption that also silenced RETIRED/OVERCLAIM would defeat glint#159.
    """
    banner = "% SUPERSEDED (2026-08-27): kept as a historical reference only.\n"
    trigger = r"scorer: coverage-gated defect & \textbf{69\% (best)} & --- \\" + "\n"
    stale = (r"When the real crystal frames are evaluated against the reference cell, 97\% lie "
             r"above the $+3\sigma$ null level, with a median separation of 7.5 null standard "
             r"deviations." + "\n")

    # 1. the trigger alone, in a LIVE file -> REQUIRED fires
    live = _cn.check_required(Path("live.tex"), trigger)
    assert any("negatives-69-is-the-lattice-bar" in f for f in live), (
        "the REQUIRED rule no longer fires on an unqualified '69% (best)' in a live file; "
        f"got: {live}")

    # 2. the same trigger behind the banner -> REQUIRED is silent
    frozen = _cn.check_required(Path("frozen.tex"), banner + trigger)
    assert not frozen, f"a SUPERSEDED file is still being asked to ADD text: {frozen}"

    # 3. ...but a stale QUOTE in that same frozen file still fails the scan
    still = _cn.scan(Path("frozen.tex"), banner + stale)
    assert any("floor-97-3sigma" in f for f in still), (
        "the SUPERSEDED exemption leaked past REQUIRED and silenced the OVERCLAIM scan, which is "
        f"exactly what glint#159 put these files on the target list to catch; got: {still}")

    # 4. the banner must be a banner: a line-start SUPERSEDED in running text exempts nothing
    near = _cn.check_required(Path("live.tex"), "title\nbody\nmore\nSUPERSEDED old note\n" + trigger)
    assert any("negatives-69-is-the-lattice-bar" in f for f in near), (
        "line-start 'SUPERSEDED' outside the opening banner is exempting REQUIRED rules")

    # 5. ...including when the word appears deep in running text
    deep = _cn.check_required(Path("live.tex"), ("x" * 900) + "\nSUPERSEDED\n" + trigger)
    assert any("negatives-69-is-the-lattice-bar" in f for f in deep), (
        "'SUPERSEDED' outside the file's opening banner is exempting REQUIRED rules")


def test_oracle_ceiling_is_not_asserted_as_a_sum():
    """sf_ceiling_of120 must NOT be checked as sf_strict + sf_selmiss, or as >= solved.

    oracle_blind.py PRINTS the ceiling as "(solved+selection-miss)" but counts reachability
    independently, off all_annealed's candidate list. At HEAD one frame is selected-and-passing
    while never counted reachable, so the shipped numbers are 79 + 8 = 87 against a printed 86;
    on the 2026-06-29 archive they agree (77 + 9 = 86). A well-meaning later edit that "restores
    the identity" would therefore make the guard fail on the real measurement, or push someone to
    edit 86 to 87 and invent one. Pinned as a negative so the omission reads as deliberate.
    """
    F = _cn.FACTS
    assert int(F["sf_strict_of120"]) + int(F["sf_selmiss_of120"]) != int(F["sf_ceiling_of120"]), (
        "the shipped counts now satisfy solved+selmiss == ceiling; if oracle_blind.py was fixed "
        "to count reachability consistently, re-measure and then this test should be retired")
    bad, _ = _arith()
    assert not any("ceiling" in b and "selection-miss" in b for b in bad), (
        "check_arithmetic asserts the ceiling as solved+selection-miss, which the shipped "
        f'measurement contradicts: {F["sf_strict_of120"]} + {F["sf_selmiss_of120"]} != '
        f'{F["sf_ceiling_of120"]}')
    low_bad, _ = _arith(sf_ceiling_of120=70)
    assert not any("BELOW the solved count" in b or "reachable by definition" in b for b in low_bad), (
        "check_arithmetic asserts an oracle ceiling lower bound that oracle_blind.py's independent "
        "reachability/solved sets do not justify")


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


def _s16_prose(**edit) -> str:
    """SI S16's protocol + result sentences, as the manuscript writes them, values substitutable.

    Carries all ten banked claims in the manuscript's own phrasing, because the rule's needles
    bind each value to its N and run context -- a probe stating fewer claims, or stating them
    context-free, would test a weaker rule than the one that ships.
    """
    v = dict(per_n=f"{_cn.FACTS['subset_draws_per_n']:d}",
             seeds=_cn._numword(_cn.FACTS['subset_seeds']),   # co-moves with FACTS, like the needle
             idx278=f"{_cn.FACTS['indexed_r0278']:d}",
             idx058=f"{_cn.FACTS['indexed_r0058']:d}",
             draws=f"{_cn.FACTS['subset_draws_total']:d}",
             n12=f"{_cn.FACTS['recov_r0278_n12_pct']:g}",
             n16a=f"{_cn.FACTS['recov_r0278_n16_pct']:g}",
             n16b=f"{_cn.FACTS['recov_r0058_n16_pct']:g}",
             nstar=f"{_cn.FACTS['nstar_r0278']:d}",
             at12="at $N=12$", run278="for r0278", run058="for r0058",
             cross="Both runs cross the $90\\%$ level by", extra="")
    v.update(edit)                       # update, not **edit: an override is the whole point here
    return (f"\\SIsec{{S16. Consensus recovery from random subsets of long runs}}\n"
            f"Two runs from LCLS experiment mfxl1038923: r0278, with ${v['idx278']}$ indexed "
            f"frames, and r0058, with ${v['idx058']}$. {v['extra']}For a subset size $N$, draw $R={v['per_n']}$ "
            f"draws per $N$, repeated over {v['seeds']} random seeds.\n"
            f"{v['cross']} $N^{{\\star}}={v['nstar']}$: pooled recovery is "
            f"${v['n12']}\\%$ {v['at12']} and ${v['n16a']}\\%$ at $N=16$ {v['run278']}, and "
            f"${v['n16b']}\\%$ at $N=16$ {v['run058']} (${v['draws']}$ draws per point).")


def test_s16_required_passes_on_the_measured_values():
    """The section as written must be clean -- otherwise the rule is unsatisfiable, not a guard."""
    assert not _required_fires(_s16_prose(), "s16-subset-recovery-facts")


def test_s16_required_fires_on_every_edited_claim():
    """The finding: #163 banked ten S16 numbers and no rule read the section they came from.

    One perturbation per needle, because a Required entry that fires on only some of its values is
    the same fail-open as no entry at all -- the case `test_every_merge_fact_is_read_by_a_required
    _rule` exists to catch for the merge block.
    """
    for edit in (dict(per_n="300"), dict(seeds="five"), dict(idx278="1700"),
                 dict(idx058="2300"), dict(draws="3000"), dict(n12="90.4"),
                 dict(n16a="95.1"), dict(n16b="89.9"), dict(nstar="32")):
        assert _required_fires(_s16_prose(**edit), "s16-subset-recovery-facts"), (
            f"S16 rule stayed silent on {edit}")


def test_s16_required_fires_on_swapped_context():
    """The round-2 finding: value-only needles pass under a SWAP.

    Exchanging 89.8 and 96.1 leaves every value present in the file while N=12 now exceeds the
    bar and N*=16 is false; same for trading the two N=16 recoveries between runs. The bound
    needles must fire on both swaps -- these probes are what make the binding real rather than
    asserted.
    """
    F = _cn.FACTS
    swap_pct = _s16_prose(n12=f"{F['recov_r0278_n16_pct']:g}",
                          n16a=f"{F['recov_r0278_n12_pct']:g}")
    assert _required_fires(swap_pct, "s16-subset-recovery-facts"), (
        "swapping 89.8 and 96.1 between N=12 and N=16 stayed green")
    swap_run = _s16_prose(n16a=f"{F['recov_r0058_n16_pct']:g}",
                          n16b=f"{F['recov_r0278_n16_pct']:g}")
    assert _required_fires(swap_run, "s16-subset-recovery-facts"), (
        "trading the two N=16 recoveries between r0278 and r0058 stayed green")


def test_s16_required_stays_silent_without_its_trigger():
    """Scoping. Files that discuss subsets or pooled consensus without carrying S16 must pass."""
    for probe in ("The consensus vote pools N-best hypotheses over random subsets of frames.",
                  "Recovery of the all-frame cell improves with the number of pooled frames.",
                  _jungfrau_prose()):
        assert not _required_fires(probe, "s16-subset-recovery-facts"), probe


def test_s16_required_fires_on_narrowed_or_moved_claims():
    """Round-3 findings: two more shapes that value-only or loosely-bound needles let through.

    (a) The both-runs relationship is part of the claim: a section that quietly narrows
    "Both runs cross ... N*=16" to one run keeps a bare N*=16 needle satisfied while the second
    banked N* silently stops being asserted. (b) The N=12 clause must be bound to r0278 through
    a tempered gap: a plain [^.]{0,80} bind lazily scans past an intervening "for r0058" to the
    legitimate "for r0278" later in the sentence, so the moved claim passed.
    """
    narrowed = _s16_prose().replace("Both runs cross", "r0278 crosses")
    assert _required_fires(narrowed, "s16-subset-recovery-facts"), (
        "narrowing the N*=16 claim to one run stayed green -- the r0058 half is unwatched")
    moved = _s16_prose(at12="at $N=12$ for r0058")
    assert _required_fires(moved, "s16-subset-recovery-facts"), (
        "moving the N=12 recovery to r0058 stayed green -- the tempered bind is not tempering")


def test_needles_reject_decimal_extensions():
    """A number needle must not be satisfied by a decimal that merely STARTS with it.

    `_lit` guarded against a following digit but not against `.<digit>`, so "1785.4 indexed
    frames" satisfied the needle for 1785 and a malformed edited count stayed green -- and the
    same hole let the retired N*=32 pattern fire on the legitimate larger values N*=320 and
    N*=32.5 (Copilot review of #163, round 6). Both directions are pinned: the needle must reject
    the decimal, and the retired rule must not claim one.
    """
    for key, field in (("indexed_r0278", "idx278"), ("indexed_r0058", "idx058")):
        stretched = _s16_prose(**{field: f"{_cn.FACTS[key]}.4"})
        assert _required_fires(stretched, "s16-subset-recovery-facts"), (
            f"{key} needle accepted a decimal extension of its value")
    assert _required_fires(_s16_prose(nstar="16.4"), "s16-subset-recovery-facts"), (
        "the N* needle accepted 16.4 as if it were 16")
    for larger in ("Pooling to $N^{\\star}=320$ was never tested.",
                   "The sweep reports $N^{\\star}=32.5$ under interpolation."):
        assert not _fires(larger, "nstar-32-retired"), (
            f"the retired-32 rule claimed a different value: {larger!r}")


def test_s16_needles_bind_values_to_what_they_count():
    """Round-7 findings: three needles held their value loosely enough to accept a wrong claim.

    (a) The indexed counts were only required "shortly after the run ID", so
    "r0278 (1785 shots; 1700 indexed frames)" passed with the guarded count wrong. (b) The N=12
    tempered gap rejected only an intervening `for r0058`, so any other run tag let the regex
    scan on to the legitimate `for r0278`. (c) The both-runs needle took `Both runs` + the bare
    number, so "Both runs used N*=16 as an arbitrary cap" kept the value and replaced the claim.
    """
    mislabelled = _s16_prose().replace(
        f"r0278, with ${_cn.FACTS['indexed_r0278']}$ indexed frames",
        f"r0278 (${_cn.FACTS['indexed_r0278']}$ shots; 1700 indexed frames)")
    assert _required_fires(mislabelled, "s16-subset-recovery-facts"), (
        "the indexed count was accepted without being bound to 'indexed frames'")
    other_prep = _s16_prose(at12="at $N=12$ on r0058")
    assert _required_fires(other_prep, "s16-subset-recovery-facts"), (
        "a run ID written with a different preposition slipped the tempered gap (round 10)")
    only_after = _s16_prose(cross="Both runs cross the $90\\%$ level only after")
    assert _required_fires(only_after, "s16-subset-recovery-facts"), (
        "'cross 90% only after N*=16' satisfied the needle while contradicting the claim")
    other_run = _s16_prose(at12="at $N=12$ for r9999")
    assert _required_fires(other_run, "s16-subset-recovery-facts"), (
        "an unrelated run tag let the N=12 bind scan onward to r0278")
    relabelled = _s16_prose().replace(
        f"and r0058, with ${_cn.FACTS['indexed_r0058']}$.",
        f"and r0058, with ${_cn.FACTS['indexed_r0058']}$ shots.")
    assert _required_fires(relabelled, "s16-subset-recovery-facts"), (
        "the r0058 count was accepted after its 'indexed frames' label was replaced")
    recast = _s16_prose(cross="Both runs used")
    assert _required_fires(recast, "s16-subset-recovery-facts"), (
        "the threshold claim was replaced while the number survived")
    # THE WHOLE EVASION BATTERY, rounds 12-14 plus three I constructed after the round-14 fix
    # to show a blacklist could not converge. The gap before the threshold is now a WHITELIST
    # (article + LaTeX punctuation only), which closes all of them at once and cannot be widened
    # by a synonym -- that is why the list below is allowed to keep growing without the pattern
    # having to.
    for pre_neg in ("Both runs cross anything except the $90\\%$ recovery level by",
                    "Both runs cross, or fail to cross, the $90\\%$ level by",
                    "Both runs cross a threshold below the $90\\%$ recovery level by",
                    "Both runs cross a threshold near the $90\\%$ recovery level by",
                    "Both runs cross a weaker $90\\%$ proxy level by",
                    "Both runs cross roughly half the $90\\%$ recovery level by"):
        assert _required_fires(_s16_prose(cross=pre_neg), "s16-subset-recovery-facts"), (
            f"negation before the threshold satisfied the both-runs needle: {pre_neg!r}")
    # Negation WITHOUT punctuation, which the comma-breaking constraint alone did not stop:
    # in "level not by", the two permitted \w+ tokens are "level" and "not" (round 12).
    for neg in ("Both runs cross the $90\\%$ level not by",
                "Both runs cross the $90\\%$ level never by",
                # CONDITIONAL, not negation -- no negator list would ever have held "whether",
                # which is why this bridge is a whitelist now too (round 15).
                "Both runs cross the $90\\%$ level whether by"):
        assert _required_fires(_s16_prose(cross=neg), "s16-subset-recovery-facts"), (
            f"punctuation-free negation satisfied the both-runs needle: {neg!r}")
    # ...and the bridge must still admit legitimate rewordings, or the guard becomes a style rule.
    for ok_bridge in ("Both runs cross $90\\%$ recovery by",
                      "Both runs cross the $90\\%$ recovery level by"):
        assert not _required_fires(_s16_prose(cross=ok_bridge), "s16-subset-recovery-facts"), (
            f"a legitimate rewording of the bridge was rejected: {ok_bridge!r}")
    negated_by = _s16_prose(cross="Both runs cross the $90\\%$ level, but not by")
    assert _required_fires(negated_by, "s16-subset-recovery-facts"), (
        "'but not by N*=16' satisfied the both-runs needle -- 'by' must be pinned "
        "immediately before N* with only whitespace, not admitted through an arbitrary gap (round 11)")


def test_needles_reject_signs_and_unbounded_digits():
    """Round-8 findings: three ways a needle matched a value it should not have.

    (a) `_lit` excluded a leading digit or period but not a SIGN, so "-1785 indexed frames" and
    "-0.90" satisfied every needle built from the positive literal. (b) The numeric seeds branch
    was unbounded, so "8 random seeds" was found inside "18 random seeds". (c) The retired N*=32
    rule's decimal guard went too far the other way and fell silent on "N*=32.0", which is the
    same retired claim with a trailing zero -- only a NONZERO decimal is a different number.
    """
    assert _required_fires(_s16_prose(idx278=f"-{_cn.FACTS['indexed_r0278']}"),
                           "s16-subset-recovery-facts"), "a sign-flipped count satisfied the needle"
    assert _required_fires(_s16_prose(seeds="18"), "s16-subset-recovery-facts"), (
        "'8 random seeds' was found inside '18 random seeds'")
    assert _fires("The sweep gives $N^{\\star}=32.0$ for r0058.", "nstar-32-retired"), (
        "the retired claim written as 32.0 slipped past the decimal guard")
    assert not _fires("Interpolation puts it at $N^{\\star}=32.5$.", "nstar-32-retired")
    assert not _fires("Pooling to $N^{\\star}=320$ was never tested.", "nstar-32-retired")


def test_nstar32_exemption_is_clause_scoped():
    """Round-3 finding: the 240-char exemption window let one properly retired mention exempt a
    SEPARATE live N*=32 claim in the same paragraph, and full-sentence scope then failed the same
    way through a comma. The exemption is CLAUSE-scoped, so each probe below fires on its live
    claim while a self-contained retirement stays exempt -- and a decimal must not truncate the
    scope (sentence ends are '.', '?' or '!' followed by whitespace, never a bare '.')."""
    mixed = ("The previously quoted $N^{\\star}=32$ for r0058 does not reproduce here. "
             "The reconstructed protocol gives $N^{\\star}=32$ for r0058.")
    assert _fires(mixed, "nstar-32-retired"), (
        "a live N*=32 rode the previous sentence's retirement vocabulary out")
    question = ("Was the previously quoted $N^{\\star}=32$ reproduced? "
                "The protocol gives $N^{\\star}=32$ for r0058.")
    assert _fires(question, "nstar-32-retired"), (
        "a '?' sentence end was read as one sentence -- the live second claim passed (round 4)")
    one_sentence = ("The previously quoted $N^{\\star}=32$ does not reproduce, "
                    "but the reconstructed protocol gives $N^{\\star}=32$ for r0058.")
    assert _fires(one_sentence, "nstar-32-retired"), (
        "a comma joined a retirement and a live claim into one sentence and both were exempted "
        "(round 4) -- the exemption must be clause-scoped, not sentence-scoped")
    conjunction = ("The protocol gives $N^{\\star}=32$ but the previously quoted "
                   "$N^{\\star}=32$ does not reproduce")
    assert _fires(conjunction, "nstar-32-retired"), (
        "two matches in ONE clause shared the retirement phrase and both were exempted -- with "
        "no way to tell which occurrence it qualifies, the guard must refuse (round 7)")
    wrong_side_before = ("The reconstructed protocol gives $N^{\\star}=32$ but the original "
                         "sweep would have correctly reported 16")
    assert _fires(wrong_side_before, "nstar-32-retired"), (
        "a forward-attaching qualifier AFTER the match suppressed a live claim -- "
        "'would have correctly reported' must precede the value it retires (round 10)")
    wrong_object_before = ("The original sweep would have correctly reported 16 but the "
                           "reconstructed protocol gives $N^{\\star}=32$")
    assert _fires(wrong_object_before, "nstar-32-retired"), (
        "the pre-qualifier already attaches to 16 mid-clause -- it must end IMMEDIATELY before "
        "the retired value, not merely somewhere earlier in the clause (round 11)")
    counterfactual = ("a coarser grid whose next tested point after 16 was 32 would have "
                      "correctly reported $N^{\\star}=32$")
    assert not _fires(counterfactual, "nstar-32-retired"), (
        "the manuscript's real counterfactual form must stay exempt")
    unrelated = ("The old recovery does not reproduce but the reconstructed protocol gives "
                 "$N^{\\star}=32$")
    assert _fires(unrelated, "nstar-32-retired"), (
        "an unrelated failure earlier in the clause suppressed a live claim -- the retirement "
        "phrase must FOLLOW the value it retires (round 9)")
    standalone = "The previously quoted $N^{\\star}=32$ for r0058 remains correct."
    assert _fires(standalone, "nstar-32-retired"), (
        "'previously quoted' alone suppressed the rule -- an exempt must RETIRE the value, and "
        "ORed entries made this one a standalone escape (round 6)")
    decimal_span = ("The previously quoted $N^{\\star}=32$ (recovery 93.4\\% at $N=24$) "
                    "does not reproduce here.")
    assert not _fires(decimal_span, "nstar-32-retired"), (
        "a decimal split the sentence and orphaned the retirement vocabulary")


def test_nstar32_rule_fires_live_and_stays_exempt_when_retired():
    """The retired N*=32, in all three states -- the probes #163 shipped without.

    The middle case is the one Copilot's review turned up: "reconstructed protocol" had been put in
    the exempt tuple, and it neither retires the value nor states it counterfactually, so the live
    and WRONG sentence below sat inside the exemption and passed. The exempts that remain are the
    ones that actually retire it, and the manuscript's real sentence carries both.
    """
    assert _fires("The pooled sweep gives $N^{\\star} = 32$ for r0058.", "nstar-32-retired")
    assert _fires("The reconstructed protocol gives $N^{\\star} = 32$ for r0058.",
                  "nstar-32-retired"), "a live wrong claim rode the section's own vocabulary out"
    for retired in (
            "the previously quoted $N^{\\star}=32$ for r0058 does not reproduce here",
            "a coarser grid whose next tested point after 16 was 32 would have correctly "
            "reported $N^{\\star}=32$"):
        assert not _fires(retired, "nstar-32-retired"), retired
    assert not _fires(f"Both runs give $N^{{\\star}} = {_cn.FACTS['n_cross90_r0058']}$.",
                      "nstar-32-retired")


def test_every_test_in_this_file_is_registered():
    """The __main__ runner lists its tests by hand, so a new one is silent until it is added.

    Caught the moment it happened: the four S16 tests above were written, passed under pytest, and
    the `python experiments/...` run -- which is the form CI uses -- reported 21/21 without ever
    calling them. A test that is never called is worse than no test, because the count goes up.
    Reads the tuple out of the source rather than importing it, since it is built inside
    `if __name__ == "__main__"` and does not exist when pytest collects this module. Anchored on
    the INDENTED assignment and taken from the last match: an unanchored split matched the literal
    inside this function first and read its own body as the registry (every test then reported
    unregistered, which is at least a loud way to be wrong).
    """
    src = Path(__file__).read_text(encoding="utf-8")
    body = re.split(r"^ +tests = \(", src, flags=re.M)[-1].split(")\n", 1)[0]
    registered = set(re.findall(r"\btest_\w+", body))
    defined = set(re.findall(r"^def (test_\w+)", src, re.M))
    assert not (defined - registered), (
        f"defined but never run by the __main__ runner: {sorted(defined - registered)}")


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


def test_deprecated_saturating_batch_alias_is_preserved():
    assert _cn.FACTS["saturating_batch"] == _cn.FACTS["driver_default_batch"] == 64




# ---------------------------------------------------------------------------------------------
# Triage 153/216 follow-up (Copilot review of #184): the sf_*/floor_* FACTS must be READ by a rule
# that looks at the deliverable, and the superseded "97% above +3 sigma" pairing must fire.
# Probes are rendered from FACTS with the manuscript's own wording, so a value edited in FACTS
# without the paper following (or vice versa) shows up here as a failing needle.

def _floor_prose(**edit) -> str:
    F = _cn.FACTS
    v = dict(pct=f"{F['floor_above_pct']:d}", n=f"{F['floor_above_of80']:d}",
             den=f"{F['floor_real_frames']:d}", fl=f"{F['floor_at_K']:.1f}",
             meas=f"{F['floor_measured_at_K']:.1f}", msd=f"{F['floor_measured_sd_at_K']:.1f}",
             med=f"{F['floor_median_sigmas']:.1f}", mean=f"{F['floor_mean_sigmas']:.1f}",
             p3=f"{F['floor_above_3sd_pct']:d}", n3=f"{F['floor_above_3sd_of80']:d}",
             r=f"{F['floor_fit_r']:.3f}",
             K="70{,}400")          # the manuscript's LaTeX rendering of FACTS['floor_K']
    v.update(edit)
    return (f"When the real crystal frames are evaluated against the reference cell, "
            f"${v['pct']}\\%$ (${v['n']}/{v['den']}$) score above the extreme-value floor: the "
            f"$\\sqrt{{2\\ln K}}$ fit of Fig.~\\ref{{fig:stat_floor}}(\\emph{{b}}) places that floor at "
            f"${v['fl']}$ inliers at $K={v['K']}$, consistent with the directly measured null maximum "
            f"of ${v['meas']}\\pm{v['msd']}$. The median separation is {v['med']} standard deviations of "
            f"the null-maximum distribution (mean {v['mean']}), and ${v['p3']}\\%$ (${v['n3']}/{v['den']}$) "
            f"lie more than $3\\sigma$ above that distribution's mean. The observed growth is consistent "
            f"with the $\\sqrt{{2\\ln K}}$ look-elsewhere scaling over the tested range ($r={v['r']}$).")


def _split_prose(**edit) -> str:
    F = _cn.FACTS
    v = dict(un=f"{120 - F['sf_strict_of120']:d}", g=f"{F['sf_genmiss_of120']:d}",
             s=f"{F['sf_selmiss_of120']:d}", gp=f"{F['sf_genmiss_rate_pct']:d}",
             sp=f"{F['sf_selmiss_rate_pct']:d}")
    v.update(edit)
    return (f"At the stricter criterion the {v['un']} unaccepted frames divide into {v['g']} generation "
            f"misses, in which no retained candidate matches the reference lattice, and {v['s']} "
            f"selection misses, in which such a candidate is present but is not chosen (${v['gp']}\\%$ "
            f"and ${v['sp']}\\%$ of the 120 frames).")


def _ceiling_prose(**edit) -> str:
    F = _cn.FACTS
    v = dict(lat=f"{F['sf_lattice_of120']:d}",
             lp=f"{F['sf_lattice_rate_pct']:d}",
             c=f"{F['sf_ceiling_of120']:d}",
             p=f"{round(100.0 * F['sf_ceiling_of120'] / 120):d}",
             fp=f"{100 * int(F['sf_ceiling_of120']) // 120:d}")   # the FLOORED stdout rendering
    v.update(edit)
    return (f"Blind indexing on sparse cxidb reaches ${v['lat']}/120 ~ {v['lp']}\\%$ "
            f"at the correct-lattice bar "
            f"(oracle-reachable ceiling ${v['c']}/120 ~ {v['p']}\\%$; "
            f"oracle_blind.py floors that printout to ${v['fp']}\\%$).")


def _negatives_caption(**edit) -> str:
    F = _cn.FACTS
    v = dict(base=f"{F['sf_negatives_lattice_of120']:d}", bstrict=f"{F['sf_negatives_strict_pct']:d}",
             lat=f"{F['sf_lattice_of120']:d}", latp=f"{F['sf_lattice_rate_pct']:d}",
             st=f"{F['sf_strict_of120']:d}", stp=f"{F['sf_strict_rate_pct']:d}",
             best=f"{F['sf_negatives_lattice_pct']:d}")
    v.update(edit)
    return (f"\\caption{{\\label{{tab:negatives}}Single-frame levers explored. Both are a looser bar than "
            f"the $\\geq$25\\%-of-spots gate used in Sec.~\\ref{{sec:ceiling}}. These percentages come "
            f"from one scorer-development sweep on a single A100 and are therefore comparable with one "
            f"another rather than with the shipped rates: the baseline row is ${v['base']}/120$ at this "
            f"bar and ${v['bstrict']}\\%$ at the $\\geq$25\\% gate, where the shipped code now gives "
            f"${v['lat']}/120$ (${v['latp']}\\%$) and ${v['st']}/120$ (${v['stp']}\\%$).}}\n"
            f"\\begin{{tabular}}{{lll}}\nscorer: coverage-gated defect & \\textbf{{{v['best']}\\% (best)}} "
            f"& --- \\\\\n\\end{{tabular}}")


def test_floor_required_passes_on_the_measured_values():
    for name in ("floor-facts", "floor-fit-r"):
        assert not _required_fires(_floor_prose(), name), name
    unicode_sigma = _floor_prose().replace(r"3\sigma", "3σ")
    assert not _required_fires(unicode_sigma, "floor-facts")
    assert not _fires(_floor_prose(), "floor-97-3sigma")


def test_floor_required_fires_on_the_wrong_K():
    """floor_at_K is 50.6 only AT K=70,400 -- the fit's whole content is that the floor grows with K.

    The rule checked the floor value and never the K it was quoted against, so a manuscript saying
    "50.6 inliers at K=700" satisfied every needle (#184 review, suppressed comment). Both the
    separator-free and plain-comma renderings must still pass, since the needle exists to pin the
    NUMBER, not the manuscript's typography.
    """
    assert not _required_fires(_floor_prose(), "floor-facts")
    for good in ("70{,}400", "70,400", "70400"):
        assert not _required_fires(_floor_prose(K=good), "floor-facts"), f"rejects valid {good!r}"
    for bad in ("700", "7040", "704000", "70{,}401"):
        assert _required_fires(_floor_prose(K=bad), "floor-facts"), f"silent on K={bad!r}"


def test_floor_required_fires_on_every_edited_value():
    for edit in (dict(pct="97"), dict(n="77"), dict(fl="50.7"), dict(meas="50.9"), dict(msd="3.9"),
                 dict(med="7.9"), dict(mean="10.4"), dict(p3="80"), dict(n3="64")):
        assert _required_fires(_floor_prose(**edit), "floor-facts"), f"floor-facts silent on {edit}"
    assert _required_fires(_floor_prose(r="0.990"), "floor-fit-r")


def test_superseded_97_above_3sigma_sentence_fires():
    old = ("When the real crystal frames are evaluated against the reference cell, 97\\% lie above the "
           "$+3\\sigma$ null level, with a median separation of 7.5 null standard deviations.")
    assert _fires(old, "floor-97-3sigma")
    assert _fires(old.replace(r"\sigma", "σ"), "floor-97-3sigma")
    assert _fires(old.replace("97", "98"), "floor-97-3sigma")   # same misattribution, new number
    # ...and the corrected sentence, which carries both a 98% and a 3 sigma, must stay clean.
    assert not _fires(_floor_prose(), "floor-97-3sigma")


def test_floor_97_rule_stops_at_the_next_percentage():
    ok = "98\\% exceed the fitted floor, while 75\\% exceed the null mean +3\\sigma."
    assert not _fires(ok, "floor-97-3sigma")
    assert not _fires(ok.replace(r"\sigma", "σ"), "floor-97-3sigma")


def test_split_required_passes_and_fires():
    assert not _required_fires(_split_prose(), "sf-oracle-split")
    for edit in (dict(un="40"), dict(g="30"), dict(s="9"), dict(gp="25"), dict(sp="8")):
        assert _required_fires(_split_prose(**edit), "sf-oracle-split"), f"silent on {edit}"


def test_ceiling_required_passes_and_fires():
    assert not _required_fires(_ceiling_prose(), "sf-ceiling-facts")
    # fp="76" is the #184 finding: a stale FLOORED printout beside a correct rounded 72%. The rule
    # pinned only the rounded value, so that text passed.
    for edit in (dict(lp="70"), dict(c="85"), dict(p="76"), dict(fp="76"), dict(fp="72")):
        assert _required_fires(_ceiling_prose(**edit), "sf-ceiling-facts"), f"silent on {edit}"


def test_negatives_required_passes_and_fires():
    for name in ("negatives-caption-facts", "negatives-69-is-the-lattice-bar"):
        assert not _required_fires(_negatives_caption(), name), name
    for edit in (dict(base="79"), dict(bstrict="69"), dict(lat="83"), dict(latp="69"),
                 dict(st="85"), dict(stp="71")):
        assert _required_fires(_negatives_caption(**edit), "negatives-caption-facts"), f"silent on {edit}"
    # The finding itself: a "69% (best)" cell whose caption no longer anchors it to the lattice bar.
    bare = "\\caption{Single-frame levers explored; all fail.}\n\\begin{tabular}{lll}\nscorer & \\textbf{69\\% (best)} & --- \\\\\n\\end{tabular}"
    assert _required_fires(bare, "negatives-69-is-the-lattice-bar")


def test_new_required_rules_stay_silent_without_their_trigger():
    quiet = "A paragraph about something else entirely, with 79/120 and 98\\% in it but no trigger."
    for name in ("floor-facts", "floor-fit-r", "sf-oracle-split", "sf-ceiling-facts", "negatives-caption-facts",
                 "negatives-69-is-the-lattice-bar"):
        assert not _required_fires(quiet, name), name


if __name__ == "__main__":
    tests = (test_closed_form_mle_matches_brute_force,
             test_bounds_ordered_and_bracket_the_estimate,
             test_zero_discordant_pairs_is_not_certainty,
             test_coverage_is_at_least_nominal,
             test_every_new_rule_fires_on_its_own_trigger,
             test_no_new_rule_fires_on_corrected_text,
             test_rule_names_are_unique_and_carry_replacements,
             test_stream_band_rule_crosses_a_latex_hard_wrap,
             test_fps29_statcard_covers_the_deck_shape_fps29_cannot_see,
             test_hl_rendering_fires_on_marks_and_spares_comments_and_the_macro_def,
             test_consensus_117_accepts_the_form_its_own_advice_requests,
             test_guard_advice_is_not_itself_retired,
             test_no_rule_text_anywhere_is_itself_retired,
             test_advice_check_covers_the_exempt_rules,
             test_every_merge_fact_is_read_by_a_required_rule,
             test_required_merge_rows_pass_on_the_measured_values,
             test_required_merge_rows_fire_on_an_edited_cell,
             test_required_whole_file_form_catches_a_value_leaving_the_file,
             test_required_stays_silent_without_its_trigger,
             test_superseded_files_are_exempt_from_required_rules_only,
             test_oracle_ceiling_is_not_asserted_as_a_sum,
             test_blind_pair_rules_fire_and_stay_silent,
             test_s16_required_passes_on_the_measured_values,
             test_s16_required_fires_on_every_edited_claim,
             test_s16_required_fires_on_swapped_context,
             test_s16_required_fires_on_narrowed_or_moved_claims,
             test_needles_reject_decimal_extensions,
             test_s16_needles_bind_values_to_what_they_count,
             test_needles_reject_signs_and_unbounded_digits,
             test_nstar32_exemption_is_clause_scoped,
             test_s16_required_stays_silent_without_its_trigger,
             test_nstar32_rule_fires_live_and_stays_exempt_when_retired,
             test_every_test_in_this_file_is_registered,
             test_submission_files_are_default_targets,
             test_gate_tie_catches_a_severed_functional_use,
             test_check_arithmetic_is_green_on_the_shipped_table,
             test_each_arithmetic_guard_fires_when_its_facts_are_perturbed,
             test_perturbations_are_restored,
             test_main_fails_on_missing_requested_targets,
             test_main_fails_on_missing_or_unreadable_in_repo_defaults,
             test_deprecated_saturating_batch_alias_is_preserved,
             test_floor_required_passes_on_the_measured_values,
             test_floor_required_fires_on_the_wrong_K,
             test_floor_required_fires_on_every_edited_value,
             test_superseded_97_above_3sigma_sentence_fires,
             test_floor_97_rule_stops_at_the_next_percentage,
             test_split_required_passes_and_fires,
             test_ceiling_required_passes_and_fires,
             test_negatives_required_passes_and_fires,
             test_new_required_rules_stay_silent_without_their_trigger,
             test_bare_86_of_120_needs_to_say_whose_it_is,
             test_fused_row_91_pairs_the_pipeline_yield_with_the_engine_speed,
)
    ok = 0
    for t in tests:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(tests)} passed")
    raise SystemExit(0 if ok == len(tests) else 1)
