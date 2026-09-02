"""The pooled-vote acceptance gate has ONE home: glint.multishot.CONSENSUS_MIN_FRAC / _MIN_LEAD.

WHY THIS EXISTS. The pair used to be defined in glint/hybrid_stream.py, which imports torch at
module level, so the shipped gate could not be read (or tested) anywhere torch is absent -- including
the CPU CI job, where every consensus test runs. They now live in glint.multishot next to
consensus_cell, and hybrid_stream RE-EXPORTS them so `from glint.hybrid_stream import
CONSENSUS_MIN_FRAC` keeps working (experiments/test_consensus_gate.py imports and rebinds them there).

Two halves:

* numpy half -- always runs. The shipped defaults are (0.02, 1.5) when the GLINT_CONSENSUS_MIN_*
  env overrides are unset (if they ARE set, the constants must equal the override -- the env hook is
  part of the contract); hybrid_stream.py's source no longer defines them from os.environ (a second
  definition would silently drift from the first, which is the dedup this test exists for) and its
  one consensus_cell() call still passes them by name.

* re-export half -- runs where torch is importable and skips otherwise (the pattern
  test_gate_constants_dedup.py's import half uses): hybrid_stream.CONSENSUS_MIN_FRAC IS
  multishot.CONSENSUS_MIN_FRAC (same object, not merely equal), and likewise for _MIN_LEAD.

  PYTHONPATH=. python experiments/test_consensus_gate_constants.py     # exit 0 = all pass
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from glint import multishot                                              # noqa: E402

fails = []


def check(name, cond, msg=""):
    print(f"  {name:70s}: {'PASS' if cond else 'FAIL'}   {msg}")
    if not cond:
        fails.append(name)


def numpy_half():
    print("numpy half (no torch needed)")
    pair = (multishot.CONSENSUS_MIN_FRAC, multishot.CONSENSUS_MIN_LEAD)
    env_frac, env_lead = (os.environ.get("GLINT_CONSENSUS_MIN_FRAC"),
                          os.environ.get("GLINT_CONSENSUS_MIN_LEAD"))
    if env_frac is None and env_lead is None:
        check("multishot ships (CONSENSUS_MIN_FRAC, CONSENSUS_MIN_LEAD) == (0.02, 1.5)",
              pair == (0.02, 1.5), repr(pair))
    else:
        # An ambient override is legitimate (that is what the env hook is for), but then the
        # DEFAULT cannot be observed here; check the hook did what it says instead.
        want = (float(env_frac) if env_frac is not None else 0.02,
                float(env_lead) if env_lead is not None else 1.5)
        check(f"env override honoured: constants == {want}", pair == want,
              f"{pair!r}  (GLINT_CONSENSUS_MIN_FRAC={env_frac!r}, GLINT_CONSENSUS_MIN_LEAD={env_lead!r})")
    check("both are floats", all(isinstance(x, float) for x in pair), repr(pair))

    # SOURCE: hybrid_stream.py must not carry a second os.environ definition of either name.
    with open(os.path.join(ROOT, "glint", "hybrid_stream.py")) as f:
        tree = ast.parse(f.read(), filename="glint/hybrid_stream.py")
    defined_here = [n.targets[0].id for n in tree.body
                    if isinstance(n, ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id in ("CONSENSUS_MIN_FRAC", "CONSENSUS_MIN_LEAD")]
    check("hybrid_stream.py defines neither constant itself (single home = multishot)",
          not defined_here, f"defined: {defined_here}")
    imported = [a.name for n in tree.body if isinstance(n, ast.ImportFrom)
                and n.module == "glint.multishot" for a in n.names]
    check("hybrid_stream.py imports both from glint.multishot",
          {"CONSENSUS_MIN_FRAC", "CONSENSUS_MIN_LEAD"} <= set(imported), f"imports: {imported}")
    # ...and the one consensus_cell() call there still passes them BY NAME (a literal 0.02 / 1.5
    # re-inlined at the call would pass the two checks above and drift anyway).
    kw = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "consensus_cell":
            kw = {k.arg: getattr(k.value, "id", None) for k in node.keywords}
    check("hybrid_stream's consensus_cell(...) passes min_frac=CONSENSUS_MIN_FRAC, "
          "min_lead=CONSENSUS_MIN_LEAD",
          kw.get("min_frac") == "CONSENSUS_MIN_FRAC" and kw.get("min_lead") == "CONSENSUS_MIN_LEAD",
          repr(kw))


def reexport_half():
    print("re-export half")
    try:
        import torch                                                     # noqa: F401
    except ImportError:
        print("  SKIP -- no torch: glint.hybrid_stream cannot import here (the numpy half above "
              "already pinned its source to the multishot import)")
        return
    from glint import hybrid_stream as hs
    check("hybrid_stream.CONSENSUS_MIN_FRAC is multishot.CONSENSUS_MIN_FRAC",
          hs.CONSENSUS_MIN_FRAC is multishot.CONSENSUS_MIN_FRAC,
          f"{hs.CONSENSUS_MIN_FRAC!r} vs {multishot.CONSENSUS_MIN_FRAC!r}")
    check("hybrid_stream.CONSENSUS_MIN_LEAD is multishot.CONSENSUS_MIN_LEAD",
          hs.CONSENSUS_MIN_LEAD is multishot.CONSENSUS_MIN_LEAD,
          f"{hs.CONSENSUS_MIN_LEAD!r} vs {multishot.CONSENSUS_MIN_LEAD!r}")


if __name__ == "__main__":
    numpy_half()
    reexport_half()
    print()
    if fails:
        print(f"FAIL: {len(fails)} check(s): {fails}")
        sys.exit(1)
    print("ALL PASS")
