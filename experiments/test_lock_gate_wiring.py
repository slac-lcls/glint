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
    onward had shifted a slot). A future tidy-up grouping them by topic would reintroduce it."""
    params = list(inspect.signature(StreamDriver.__init__).parameters)
    assert params[-3:] == ["lock_frac", "lock_lead", "lock_pool_switch"], params[-3:]


if __name__ == "__main__":
    tests = (test_blind_driver_gets_the_pool_keyed_gate_by_default,
             test_overrides_are_forwarded_including_the_inert_one,
             test_known_cell_driver_builds_no_vote_histogram,
             test_new_options_stay_at_the_end_of_the_signature)
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
