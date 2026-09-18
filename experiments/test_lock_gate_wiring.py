"""CPU test that the blind driver's consensus gate is WIRED, not merely implemented.

Everything else about `pool_switch` is tested against `RunningConsensus` directly
(test_running_consensus.py), which leaves the thing the change actually ships -- the production
DEFAULT -- living in exactly two places no test touched: `StreamDriver.__init__`'s signature and
the one call site that forwards it. Drop the three kwargs from that call, or reset the defaults to
the inert values, and the entire rest of the suite still passes.

Builds a real StreamDriver on CPU (`use_gpu=False`), which is cheap: the blind branch only
constructs the vote histogram and resolves the blind indexer lazily, so no GPU, no pixels and no
peak-finding are involved in reading back what it was given.

That lazy `from glint.glint_fast import index_blind_nbest` is the one thing standing between this
file and a torch-free machine, and it is also the seam the other driver tests inject through. It is
stubbed here UNCONDITIONALLY rather than only when torch is missing: the first version of this file
passed locally and failed CI with ModuleNotFoundError, which is precisely the divergence a
conditional stub preserves. Nothing here exercises the indexer, so nothing is lost by never
importing the real one.

Run: `python experiments/test_lock_gate_wiring.py` or `pytest`. Wired into the CPU CI job
alongside test_running_consensus.py, which covers the rule this covers the wiring of.
"""
import contextlib
import inspect
import sys
import types

import numpy as np

from glint.stream_driver import StreamDriver

N = 32
PANELS = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=1e4,
               cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5), coffset=0.0,
               min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
M_KC = np.diag([1 / 79.0, 1 / 79.0, 1 / 38.0])   # RECIPROCAL basis; a direct-space one here builds
                                                 # an HKL grid of absurd size


@contextlib.contextmanager
def _no_torch_needed():
    """Stand in for glint.glint_fast so the blind branch's lazy import resolves without torch."""
    stub = types.ModuleType("glint.glint_fast")
    stub.index_blind_nbest = lambda q, k: []          # never called: nothing here indexes a frame
    had = "glint.glint_fast" in sys.modules
    prev = sys.modules.get("glint.glint_fast")
    sys.modules["glint.glint_fast"] = stub
    try:
        yield
    finally:                                          # leave the interpreter as we found it, or a
        if had:                                       # later test in the same pytest process would
            sys.modules["glint.glint_fast"] = prev    # silently inherit the stub
        else:
            sys.modules.pop("glint.glint_fast", None)


def _driver(**kw):
    with _no_torch_needed():
        return StreamDriver(None, PANELS, 0.1, 1.3, (N, N), dtype=np.uint16, B=8, dmin=3.0,
                            use_gpu=False, **kw)


def test_blind_driver_gets_the_pool_keyed_gate_by_default():
    """The measured setting (0.02 / 1.5 / 72) reaches the histogram without being asked for."""
    rc = _driver()._rc
    assert (rc.min_frac, rc.min_lead, rc.pool_switch) == (0.02, 1.5, 72), \
        (rc.min_frac, rc.min_lead, rc.pool_switch)
    assert (rc.min_support, rc.base_gap) == (3, 2), (rc.min_support, rc.base_gap)


def test_overrides_are_forwarded_including_the_inert_one():
    """lock_frac=0 / lock_lead=1 / lock_pool_switch=0 restores the bare gap rule -- the documented
    escape hatch, which is only real if it is actually plumbed."""
    rc = _driver(lock_frac=0.0, lock_lead=1.0, lock_pool_switch=0)._rc
    assert (rc.min_frac, rc.min_lead, rc.pool_switch) == (0.0, 1.0, 0), \
        (rc.min_frac, rc.min_lead, rc.pool_switch)
    rc2 = _driver(lock_frac=0.05, lock_lead=2.0, lock_pool_switch=128)._rc
    assert (rc2.min_frac, rc2.min_lead, rc2.pool_switch) == (0.05, 2.0, 128), \
        (rc2.min_frac, rc2.min_lead, rc2.pool_switch)


def test_known_cell_driver_builds_no_vote_histogram():
    """A supplied cell means there is nothing to vote on; the gate must not appear there at all."""
    dk = StreamDriver(M_KC, PANELS, 0.1, 1.3, (N, N), dtype=np.uint16, B=8, dmin=3.0, use_gpu=False)
    assert getattr(dk, "_rc", None) is None, "known-cell driver should not create a vote histogram"


def test_new_options_stay_at_the_end_of_the_signature():
    """They sit at the very end rather than beside the other lock_* options, and that is deliberate:
    this constructor is NOT keyword-only, so inserting a parameter mid-signature silently rebinds
    every positional argument after it (review of glint#122 caught exactly that -- adaptive_relock
    onward had shifted a slot). A future tidy-up grouping them by topic would reintroduce it.

    The expected tail GROWS as options are appended (retry_cascade/retry_nbest from glint#75,
    bg_mode from glint#131, laue/ops from glint#180) -- that is the rule being obeyed, not broken. What must never happen is
    one of these moving inward, or a new option landing anywhere but after them, and pinning the
    whole tail in order still catches both. If you add an option and this fails, the fix is to add
    its name to the END here; if the name you added is not last in `params`, that is the bug this
    test is for. `bg_mode` was appended rather than filed next to the other integration options
    where `half`/`gap`/`ring` live, for exactly the reason above."""
    params = list(inspect.signature(StreamDriver.__init__).parameters)
    expect = ["self", "Mc", "panels", "clen_m", "wavelength_A", "shape", "dtype", "mask", "B",
              "dmin", "tol", "half", "gap", "ring", "min_peaks", "snr_bins", "pf_kw", "use_gpu",
              "lock_support", "lock_gap", "adaptive_gap", "warmup_nbest", "adaptive_relock",
              "min_inliers", "min_inlier_frac", "warm_topk", "warm_floor", "double_hit",
              "geom_refine", "geom_refine_kw", "rescue_buffer", "fanout", "alias_gate",
              "lock_probe", "probe_null", "lock_min_z", "warmup_rescue", "qc_frac_threshold",
              "stream_out", "stream_geom_text", "stream_image", "stream_symmetry", "stream_peaks",
              "lock_frac", "lock_lead", "lock_pool_switch", "retry_cascade", "retry_nbest",
              "bg_mode", "laue", "ops",
              "roster", "events", "on_event", "cell_window"]          # glint#199 (cell registry), appended
    assert params == expect, (params, expect)


def test_live_gate_defaults_and_window_are_pinned():
    """The LIVE gate (_fits: min_inliers count AND min_inlier_frac fraction, counting peaks at the
    _inliers near-integer window) shipped with min_inlier_frac=0.15, min_inliers=0 and a 0.15
    window, and until now nothing pinned any of the three: reset the defaults or nudge the window
    and the whole suite stayed green. Same defect this file exists for, one signature over.

    The window is asserted through BOTH names on purpose: stream_driver.HKL_TOL is the gate's own
    constant, spurious_meter.HKL_TOL is the meters' declared mirror of it, and the paper's 0.15 is
    what both must equal. stream_driver also asserts the pair equal at import, so if that module-
    scope check is ever deleted, this test still catches a split."""
    p = inspect.signature(StreamDriver.__init__).parameters
    assert p["min_inlier_frac"].default == 0.15, p["min_inlier_frac"].default
    assert p["min_inliers"].default == 0, p["min_inliers"].default
    from glint.spurious_meter import HKL_TOL
    from glint.stream_driver import HKL_TOL as DRIVER_HKL_TOL
    assert HKL_TOL == 0.15, HKL_TOL
    assert DRIVER_HKL_TOL == HKL_TOL, (DRIVER_HKL_TOL, HKL_TOL)
    # ...and the BEHAVIOR, not just the declarations: comparing the two constants leaves this
    # green if _inliers grows its own literal again (Copilot review of #170, round 2). With M the
    # identity, each q row's residual is its own fractional part, so the window is probed
    # directly from both sides, and the max(1) rule is exercised -- a row inside the window on
    # one component and outside on another must NOT count.
    #
    # THE EXACT EDGE IS TESTABLE, and it is the row that pins the COMPARISON OPERATOR: without it
    # `<` and `<=` are indistinguishable, since every other row is strictly inside or strictly
    # outside. It cannot be written as a decimal literal -- 1.15 stores as 1.1499999... and
    # silently probes the wrong side -- but HKL_TOL ITSELF works: round() subtracts exactly zero
    # from it, so the residual comes back bit-identical to the constant, whatever float that is,
    # and `residual < HKL_TOL` is False for the shipped strict `<` and True for `<=` (Copilot
    # review of #170, round 3, correcting this comment's earlier claim that the edge was
    # untestable -- it was untestable the way I first tried, not in general).
    #
    # It must be HKL_TOL BARE, not 1 + HKL_TOL: measured, 1.15 - round(1.15) = 0.1499999999999999,
    # which is strictly BELOW the constant and counts as an inlier, so the offset form silently
    # tests the wrong side exactly like the decimal literal did. Only round()-subtracts-zero
    # preserves the bits.
    M = np.eye(3)
    q = np.array([[1.149, 2.0, 3.0],      # max residual 0.149  -> in
                  [1.151, 2.0, 3.0],      # 0.151               -> out
                  [HKL_TOL, 2.0, 3.0],    # residual == HKL_TOL -> out under strict <, in under <=
                  [1.10, 2.149, 2.851],   # all inside          -> in
                  [1.149, 2.151, 3.0]])   # mixed: max rules    -> out
    got = StreamDriver._inliers(None, q, M)
    assert got == 2, f"_inliers counted {got} of the straddle set, expected 2"

    # ...and finally that _inliers READS the constant rather than merely agreeing with it. Every
    # probe above is fixed at the current 0.15, so re-inlining the original `< 0.15` literal --
    # the exact regression this PR exists to prevent -- leaves them all green (Copilot review of
    # #170, round 4). Only moving the constant and watching the behavior follow can tell the two
    # apart: with the window widened to 0.20 every row's residual (0.149, 0.151, 0.15, 0.149,
    # 0.151) falls inside, so the count must rise 2 -> 5. Restored in a finally, so a failure
    # here cannot leak a bogus tolerance into the rest of the suite.
    import glint.stream_driver as _sd
    _saved = _sd.HKL_TOL
    try:
        _sd.HKL_TOL = 0.20
        widened = StreamDriver._inliers(None, q, M)
    finally:
        _sd.HKL_TOL = _saved
    assert widened == 5, (
        f"_inliers counted {widened} at a widened window, expected 5 -- it is not reading "
        "HKL_TOL, so the canonical constant is decorative and a re-inlined literal would pass")


if __name__ == "__main__":
    tests = (test_blind_driver_gets_the_pool_keyed_gate_by_default,
             test_overrides_are_forwarded_including_the_inert_one,
             test_known_cell_driver_builds_no_vote_histogram,
             test_new_options_stay_at_the_end_of_the_signature,
             test_live_gate_defaults_and_window_are_pinned)
    ok = 0
    for t in tests:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(tests)} passed")
    # The repo's script contract, stated in .github/workflows/ci.yml: print, then exit non-zero on
    # failure. Without this the file reports "2/6 passed" and still exits 0, so a CI step running it
    # goes green on a red suite -- which matters now that CI runs this file.
    raise SystemExit(0 if ok == len(tests) else 1)
