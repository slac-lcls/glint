"""same_lattice must be SYMMETRIC: same_lattice(A,B) == same_lattice(B,A), for every pair.

WHY THIS EXISTS. `same_lattice` is used two different ways and only one of them made the old
asymmetry visible. As the paper's accuracy gate (`glint_fast.py`: "a frame counts as indexed iff
same_lattice(M, truth)") the reference is ALWAYS the second argument, so normalising the relative
tolerances by argument 2 read as "within rtol of the truth" and looked deliberate. But the same
predicate is the grouping relation in `_consensus_exact`, in `glint_fast.index_blind_nbest`'s dedup
loop, in the StreamDriver voter collection and in `hybrid_stream` -- and there BOTH arguments are
candidates, so group membership depended on which member happened to become the representative,
and therefore on arrival order.

The old body normalised by argument 2:

    abs(|det M1| - |det M2|) > vtol * |det M2|        ->  reject
    |l1 - l2| <= rtol * l2                            ->  accept

Measured before the fix: argument order decided the verdict for 1.87% of perturbed-cell pairs
overall, and 4.5% in the 2-4% perturbation band where refine drift actually lives. Worked case at
the volume gate: |det| 1.000 vs 0.905 returned False as (A,B) and True as (B,A).

WHAT IT CHECKS. Three things, in increasing order of what they would catch:

  1. SYMMETRY, on 20 000 perturbed LYSO pairs spanning the bands where the relation is actually
     undecided. Any single asymmetric verdict fails the test. This is the contract.
  2. The two documented worked cases, pinned literally, so a future renormalisation that happens
     to be symmetric but reintroduces edge behaviour still has to confront them.
  3. That the test is NOT VACUOUS -- the sample must contain both accepted and rejected pairs, and
     must contain pairs near the boundary. A perturbation range that put every pair safely inside
     or outside the tolerance would pass symmetry trivially and check nothing.

WHAT IT DELIBERATELY DOES NOT CHECK. Transitivity. A tolerance relation is never transitive:
a~b and b~c with a!~c is reachable for any positive rtol, and the test asserts that this is STILL
reachable rather than pretending the fix bought more than it did. Symmetry makes the relation
independent of argument order; it does not make greedy grouping order-independent. Well-defined
group membership needs a canonical key or a fixed-order clustering, and that is a separate change
with its own measurement.

Pure numpy, no torch, no device, no data files -- runs anywhere in about a second.
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint.lattice import cell_to_Ar                      # noqa: E402
from glint.lattice import reduced_params                  # noqa: E402
from glint.multishot import _grp_reduced, same_lattice    # noqa: E402
from glint.running_consensus import RunningConsensus      # noqa: E402

LYSO_A, LYSO_C = 79.02, 37.98


def _perturbed(rng, scale):
    return cell_to_Ar(LYSO_A * (1 + rng.normal(0, scale)),
                      LYSO_A * (1 + rng.normal(0, scale)),
                      LYSO_C * (1 + rng.normal(0, scale)), 90, 90, 90)


def _fp(M):
    l, c = reduced_params(M)
    return l, c, abs(np.linalg.det(M))


def _grp_count(*cells):
    RP = [((l, c), d) for l, c, d in map(_fp, cells)]
    return _grp_reduced(list(enumerate([1] * len(cells))), RP, 0.05, 0.06, 0.10)[0][1]


def test_symmetry():
    """same_lattice(A,B) == same_lattice(B,A) on 20 000 pairs across the undecided bands."""
    rng = np.random.default_rng(0)
    # 0.02-0.04 is where the old relation disagreed with itself most often; the wider span keeps
    # clearly-same and clearly-different pairs in the sample so the check is not vacuous.
    bands = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08]
    base, rem = divmod(20000, len(bands))
    counts = [base + (i < rem) for i in range(len(bands))]
    n_true = n_false = 0
    failures = []
    for scale, n in zip(bands, counts):
        for _ in range(n):
            A, B = _perturbed(rng, scale), _perturbed(rng, scale)
            ab, ba = same_lattice(A, B), same_lattice(B, A)
            if ab != ba:
                failures.append((scale, ab, ba))
            n_true += ab
            n_false += not ab
    assert not failures, (
        f"{len(failures)} asymmetric verdicts, e.g. {failures[:3]} "
        "-- a relative tolerance is being normalised by one argument rather than by both")
    # non-vacuity: the sample must actually exercise both outcomes
    assert n_true > 1000, f"only {n_true} accepted pairs -- bands too wide to test anything"
    assert n_false > 1000, f"only {n_false} rejected pairs -- bands too narrow to test anything"
    return n_true, n_false


def test_worked_cases():
    """The two cases from the bug report, pinned literally."""
    # Volume gate: |det| ratio 0.905. Under the old rule this was False one way, True the other.
    s = (0.905) ** (1.0 / 3.0)
    A = cell_to_Ar(LYSO_A, LYSO_A, LYSO_C, 90, 90, 90)
    B = cell_to_Ar(LYSO_A * s, LYSO_A * s, LYSO_C * s, 90, 90, 90)
    assert same_lattice(A, B), "volume gate should accept the documented 0.905 ratio (A,B)"
    assert same_lattice(B, A), "volume gate should accept the documented 0.905 ratio (B,A)"

    # Length gate: one-axis perturbations around the 5% relative step.
    for f, expected in ((1.05, True), (1.10, False), (0.95, False), (0.90, False)):
        P = cell_to_Ar(LYSO_A * f, LYSO_A, LYSO_C, 90, 90, 90)
        assert same_lattice(A, P) == expected, f"unexpected length-gate result for (A,P) at f={f}"
        assert same_lattice(P, A) == expected, f"unexpected length-gate result for (P,A) at f={f}"


def test_consensus_paths_use_mean_normalisation():
    """The same boundary cases must drive the batch and streaming consensus gates identically."""
    s = (1.0 / 0.905) ** (1.0 / 3.0)
    A = cell_to_Ar(LYSO_A, LYSO_A, LYSO_C, 90, 90, 90)
    B = cell_to_Ar(LYSO_A * s, LYSO_A * s, LYSO_C * s, 90, 90, 90)
    P = cell_to_Ar(LYSO_A * 0.95, LYSO_A, LYSO_C, 90, 90, 90)
    rc = RunningConsensus(min_support=2, adaptive=False)
    la, ca, da = _fp(A)
    lb, cb, db = _fp(B)
    lp, cp, dp = _fp(P)

    assert _grp_count(A, B) == 2 and _grp_count(B, A) == 2, "_grp_reduced split the accepted boundary pair"
    assert _grp_count(A, P) == 1 and _grp_count(P, A) == 1, "_grp_reduced merged the rejected boundary pair"

    assert rc._match(la, ca, da, [B, lb, cb, db, 1]) and rc._match(lb, cb, db, [A, la, ca, da, 1])
    assert not rc._match(la, ca, da, [P, lp, cp, dp, 1]) and not rc._match(lp, cp, dp, [A, la, ca, da, 1])

    rc._push(A, la, ca, da)
    assert tuple(rc._candidates(lb, cb, db)) == (0,)
    assert tuple(rc._candidates(lp, cp, dp)) == ()

    rc = RunningConsensus(min_support=2, adaptive=False)
    rc._push(B, lb, cb, db)
    assert tuple(rc._candidates(la, ca, da)) == (0,)
    lm = np.array([la, lb]); cm = np.array([ca, cb]); dm = np.array([da, db])
    assert rc._covered(la, ca, da, (lm, cm, dm)) == 2
    assert rc._covered(lp, cp, dp, (np.array([la, lp]), np.array([ca, cp]), np.array([da, dp]))) == 1


def test_transitivity_is_still_violable():
    """NOT a wish. Symmetry does not buy transitivity, and the fix must not be read as if it did.

    If this ever fails, the relation became transitive by some other change -- which would be a
    real result about grouping, and should be measured and written down rather than silently
    absorbed by a test that was asserting the opposite.
    """
    # ONE axis only. Scaling all three would change the volume by f**3, so a 4.8% length step
    # becomes a 15% volume step and the vtol=10% gate rejects the chain before the length gate
    # is ever consulted -- which would test the volume gate, not transitivity.
    a, b, c = 1.00, 1.048, 1.096          # ~4.7% apart pairwise, ~9.2% end to end (of the mean)
    A = cell_to_Ar(LYSO_A * a, LYSO_A, LYSO_C, 90, 90, 90)
    B = cell_to_Ar(LYSO_A * b, LYSO_A, LYSO_C, 90, 90, 90)
    C = cell_to_Ar(LYSO_A * c, LYSO_A, LYSO_C, 90, 90, 90)
    assert same_lattice(A, B) and same_lattice(B, C), "chain no longer links -- retune the step"
    assert not same_lattice(A, C), (
        "a~b, b~c AND a~c: the relation appears transitive here. That is a finding about grouping, "
        "not a passing test -- measure it and write it down.")


if __name__ == "__main__":
    n_true, n_false = test_symmetry()
    print(f"symmetry      OK  (20000 pairs; {n_true} accepted / {n_false} rejected, non-vacuous)")
    test_worked_cases()
    print("worked cases  OK  (volume ratio 0.905; length steps 0.90-1.10)")
    test_consensus_paths_use_mean_normalisation()
    print("consensus     OK  (_grp_reduced, _match, _candidates and _covered agree on the boundaries)")
    test_transitivity_is_still_violable()
    print("transitivity  still violable, as documented (a~b, b~c, a!~c)")
    print("\nPASS")
