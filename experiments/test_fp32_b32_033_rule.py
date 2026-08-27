"""Pin the fp32-b32-0.33 drift rule in BOTH directions.

0.33 is the awkward value in this project: LIVE for fp64 indexing at B=32 and for fused
box-integration, RETIRED for fp32 indexing at B=32 (0.31 since #165). So it cannot be retired
outright, and an exemption list cannot express "fp64 owns this number" -- only "fp64 is nearby".

Two earlier versions of the rule leaked, both found by Copilot review on #167, and both are pinned
below as MUST_FIRE cases 1 and 2:
  * requiring "ms" let the bare form "0.33 fp32 indexing" through;
  * exempt=("fp64",) was a +/-240-char proximity test, so an fp64 in the NEXT SENTENCE exempted a
    genuinely stale claim.
The MUST_NOT cases are what stops the obvious over-correction: a legitimate fp32-vs-fp64 contrast
puts both tokens next to a 0.33 that fp64 owns.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_numbers import AMBIGUOUS, RETIRED, _normalize, scan                                 # noqa: E402

RULE = next(r for r in RETIRED if r.name == "fp32-b32-0.33")

MUST_FIRE = [
    "fp32 indexing at B=32 is 0.33 ms. fp64 results are discussed next.",  # exempt-window leak
    "The fp32 indexing figure at B=32 is 0.33 per frame.",                 # the no-"ms" leak
    "0.33 fp32 indexing",                                                  # inverted form
    "known-cell fp32 0.33 ms/hit at batch 32",
    "at B=32 the fp32 engine measures 0.33",
    # number BEFORE its attribution, 41 chars apart -- the 15-char reverse window missed this
    # entirely (Copilot, round 4). Natural wording, and exactly the claim the rule exists to kill.
    "0.33 ms per frame for known-cell indexing in fp32 at B=32",
]
MUST_NOT_FIRE = [
    # Integration is the OTHER live meaning of 0.33 and is not precision-tagged, so an fp32 sentence
    # about integration is correct and must not be condemned (Copilot, round 3).
    "in fp32, fused box-integration reaches 0.33 ms per frame",
    "at B=32 fp32 is 0.31 ms against fp64's 0.33 ms",         # fp64 owns the 0.33
    "fp64 is 0.33 ms at B=32, fp32 is 0.31 ms",               # same, other order
    "a fused GPU box-integration reaches 0.33 ms per frame",  # the other live meaning
    "fp64 indexing at B=32 is 0.33 ms",                       # currently true
    "fp32 indexing at B=32 is 0.31 ms",                       # currently true
    "B=120 fp32 is 0.14 ms and fp64 0.17 ms",                 # unrelated
]


# The fp32 B=120 figure has the same shape: 0.16 was fp32-at-B=120 (now 0.14), but 0.16 is ALSO
# predict's live per-frame cost, so it is scoped to the fp32 pairing the same way. Adding this rule
# immediately caught two stale sites the 0.26/0.45 sweep had missed, which is why it is pinned here.
RULE16 = next(r for r in RETIRED if r.name == "fp32-b120-0.16")

MUST_FIRE_16 = [
    "GPU in fp32, measured on A100: 0.16 ms at B=120",       # the real stale site, verbatim
    "fp32 at B=120 is 0.16 ms/hit",
    "0.16 ms per hit for known-cell indexing in fp32 at B=120",   # reverse order, 39 chars apart
]
MUST_NOT_FIRE_16 = [
    "peakfind at 1.16 ms is 7.3x predict (0.16 ms)",         # predict's LIVE cost
    "predict 0.16 ms; fp32 indexing is 0.14 ms at B=120",    # both present, correctly attributed
    # ⚠ the SAME sentence in the OTHER order. Round 2 tested only the line above, and the forward
    # branch condemned this one -- testing one ordering is testing half a rule (Copilot, round 3).
    "fp32 indexing is 0.14 ms at B=120; predict 0.16 ms",
    "fp64 0.17 ms at B=120, fp32 0.14",                      # currently true
]


# ---------------------------------------------------------------------------------------------
# The LaTeX-tie battery. The retired fused family originally anchored its unit with `\s*`, and the
# manuscript writes "$0.26$~ms" / "$3.8$~kHz". `~` is not whitespace and _normalize() deliberately
# preserves it, so the guard reported glint_rewrite_JAC_refined.tex CLEAN while its INTRODUCTION,
# its tab:summary row and its streaming section all still said 0.26. Seven live sites, invisible.
# Copilot did not catch this either; an adversarial sweep on #167 did.
RULE26 = next(r for r in RETIRED if r.name == "fused-b120-0.26")
RULE38 = next(r for r in RETIRED if r.name == "fused-fps-3800")

MUST_FIRE_TIE = [
    "batched known-cell registration requires 0.26~ms per frame",   # the introduction, verbatim
    "the tabulated 0.26~ms is measured at B=120",
    "known-cell registration operates at 0.26 ms per frame",        # plain space still works
    '_snum(xr+0.28,5.12,"26 -> 0.26","ms/f index once locked")',    # across a string boundary
    "& known-cell & 76% (91/120) & --- & " + chr(92) + "textbf{0.26} & " + chr(92) + "textbf{3800}",  # unit in the header
]
MUST_NOT_FIRE_TIE = [
    "dot(s,x+0.26,cy+0.24,0.14,col)",                    # a LAYOUT COORDINATE, not a timing
    "box(s,x+0.26,cy+0.52,cw-0.4,0.68)",                 # ditto
    "38.3/79.3/79.7 (0.26%) -- same lattice either way", # a PERCENTAGE in multishot.py
    "batch $32" + chr(92) + "to120$ (occupancy, same kernels) & 0.45 & 0.26 & $1.7" + chr(92) + "times$",  # lever-table BEFORE
]
MUST_FIRE_38 = [
    "corresponding to approximately 3.8~kHz",
    "about 3800 frames~s$^{-1}$",
]

# The AMBIGUOUS rules had the identical tie blind spot one screen below the RETIRED ones. Fixing a
# class means fixing every instance of it, so this pins the ambiguity side too.
RULE_AMB = next(r for r in AMBIGUOUS if r.name == "subms-no-batch")
MUST_FIRE_AMB = [
    "the known-cell path costs 0.17~ms per frame",   # LaTeX tie, no batch qualifier
    "the known-cell path costs 0.17 ms per frame",   # plain space, same defect
]
MUST_NOT_FIRE_AMB = [
    "0.17~ms/hit at B=120",                          # throughput unit and batch both named
    "0.17 ms amortized at batch 120",                 # equivalent explicit qualification
    "Known-cell indexing costs 0.17 ms/hit at B=120.",
    "0.33~ms per frame of integration",              # other quantity named
]

MUST_FIRE_RATIO = [
    "registration is 100x cheaper when batched at B=120",
]
MUST_NOT_FIRE_RATIO = [
    "registration is 151x cheaper when batched at B=120",
    # numeric boundary: the rule was written without the (?<![\d.]) every other rule here uses,
    # so an unrelated 1100x read as the retired 100x (Copilot, round 6).
    "the new engine is 1100x cheaper than before",
]

# ffbidx-12x matched only the ordering "12x ... ffbidx"; with the comparator FIRST the retired
# speedup was invisible. Same one-directional gap as the fp32 pairing rules (Copilot, round 6).
RULE12 = next(r for r in RETIRED if r.name == "ffbidx-12x")
MUST_FIRE_12 = [
    "this is 12x faster than pipelined ffbidx",
    "ffbidx is 12x slower than the fused engine",
]
MUST_NOT_FIRE_12 = ["the anneal is 12x faster than the numpy loop"]


def _check_via_scan(fire, silent, label):
    """Rules with `needs` conditions must be tested through scan(): they are applied by scan/_near,
    not by the pattern, so matching the regex alone proves nothing about whether the file fails.
    Testing the regex for such a rule tests the wrong layer -- the first version of this battery
    did exactly that and reported two false failures."""
    from pathlib import Path
    bad = []
    for s_ in fire:
        if not any(label in f for f in scan(Path("probe.md"), s_)):
            bad.append(f"  [{label}] MISSED (should fire): {s_}")
    for s_ in silent:
        if any(label in f for f in scan(Path("probe.md"), s_)):
            bad.append(f"  [{label}] FALSE POSITIVE (should not fire): {s_}")
    return bad


def _check(rule, fire, silent, label):
    bad = []
    for s in fire:
        if not rule._rx.search(_normalize(s)):
            bad.append(f"  [{label}] MISSED (should fire): {s}")
    for s in silent:
        if rule._rx.search(_normalize(s)):
            bad.append(f"  [{label}] FALSE POSITIVE (should not fire): {s}")
    return bad


def main():
    bad = (_check(RULE, MUST_FIRE, MUST_NOT_FIRE, "fp32-b32-0.33")
           + _check(RULE16, MUST_FIRE_16, MUST_NOT_FIRE_16, "fp32-b120-0.16")
           + _check(RULE26, MUST_FIRE_TIE, MUST_NOT_FIRE_TIE, "fused-b120-0.26")
           + _check(RULE38, MUST_FIRE_38, [], "fused-fps-3800")
           + _check_via_scan(MUST_FIRE_AMB, MUST_NOT_FIRE_AMB, "subms-no-batch")
           + _check_via_scan(MUST_FIRE_RATIO, MUST_NOT_FIRE_RATIO, "ratio-100x")
           + _check(RULE12, MUST_FIRE_12, MUST_NOT_FIRE_12, "ffbidx-12x"))
    if bad:
        print(f"{len(bad)} failure(s)"); print("\n".join(bad)); return 1
    nf = (len(MUST_FIRE)+len(MUST_FIRE_16)+len(MUST_FIRE_TIE)+len(MUST_FIRE_38)
          + len(MUST_FIRE_AMB)+len(MUST_FIRE_RATIO)+len(MUST_FIRE_12) + len(MUST_FIRE_RATIO))
    ns = (len(MUST_NOT_FIRE)+len(MUST_NOT_FIRE_16)+len(MUST_NOT_FIRE_TIE)
          + len(MUST_NOT_FIRE_AMB)+len(MUST_NOT_FIRE_RATIO)+len(MUST_NOT_FIRE_12) + len(MUST_NOT_FIRE_RATIO))
    print(f"drift-rule battery OK -- {nf} fire, {ns} stay silent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
