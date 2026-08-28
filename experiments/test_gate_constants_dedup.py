"""One strict gate, imported everywhere -- no re-inlined literal copies.

glint#170 canonicalized the published acceptance gate -- same_lattice AND matched/len >= GATE_FRAC
AND matched >= GATE_MIN, with matched counting |q @ M - round(q @ M)| < GATE_TOL componentwise --
into glint/glint_fast.py. The follow-up PR converted the remaining independent literal copies
(replica_gpu's self-benchmark, four experiment scripts, both gate scorers) into imports of those
names. This test keeps them converted: re-inline a 0.15 / 0.25 / 10 into any of the converted
files and it fails.

Two halves:

* SOURCE half -- always runs, including on the torch-less CI runner (glint_fast imports torch
  unconditionally, so the canonical values are read from its AST, the same route
  check_numbers.py's gate tie uses). For each converted file it asserts (a) every module-level
  gate constant is BOUND to the canonical name -- `from glint.glint_fast import GATE_...` or
  `NAME = gf.GATE_...` -- never to a numeric literal, (b) no comparison in the file tests against
  the raw gate values, and (c) the strict_gate() helpers call matched_strict(), never the
  configurable matched(), whose QDIST=1 mode applies a different matching rule to the strict
  thresholds (the defect Copilot found inside gpass() on glint#170).

* IMPORT half -- runs where torch is importable and skips otherwise (the pattern
  test_gate_project.py uses for its GPU half): imports the import-safe converted modules and
  asserts their module-level constants equal glint.glint_fast's at runtime.

Scope guard: the live gate's window (stream_driver.HKL_TOL / spurious_meter.HKL_TOL) is a
DIFFERENT quantity that happens to share the value 0.15; nothing here scans those modules, and
replica_gpu.py's scan is restricted to its __main__ block so its ffbidx score_thr / polish
windows (live-side internals, deliberately not converted) stay out of reach.

  python experiments/test_gate_constants_dedup.py
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

CANON = os.path.join(ROOT, "glint", "glint_fast.py")
CANON_NAMES = ("GATE_TOL", "GATE_FRAC", "GATE_MIN")

# (path relative to ROOT, {local name: canonical name} for module-level gate constants,
#  strict-gate helper name or None, scan_scope: "module" or "main")
FILES = [
    ("experiments/test_streamdriver_vs_offline.py",
     {"GATE_FRAC": "GATE_FRAC", "GATE_MIN": "GATE_MIN"}, "strict_gate", "module"),
    ("experiments/test_watchdog_miss_reason.py",
     {"GATE_FRAC": "GATE_FRAC", "GATE_MIN": "GATE_MIN"}, "strict_gate", "module"),
    ("experiments/gap_at_scale.py",
     {"GATE_FRAC": "GATE_FRAC", "GATE_MIN": "GATE_MIN"}, None, "module"),
    ("experiments/gap_on_real.py",
     {"GATE_MIN": "GATE_MIN"}, None, "module"),
    ("experiments/score_glint_gate.py",
     {"TOL": "GATE_TOL", "MIN_FRAC": "GATE_FRAC", "MIN_REFL": "GATE_MIN"}, None, "module"),
    ("experiments/xgandalf/score_xg_gate.py",
     {"TOL": "GATE_TOL", "GATE_FRAC": "GATE_FRAC", "GATE_MIN": "GATE_MIN"}, None, "module"),
    # The gate here lives in the __main__ self-benchmark, not in module constants; what must hold
    # is that the canonical names are imported and no gate literal is compared against in that
    # block. matched_strict is required in the import so the residual test cannot be re-inlined.
    ("glint/replica_gpu.py",
     {}, None, "main"),
]

fails = []


def check(name, cond, msg=""):
    print(f"  {name:64s}: {'PASS' if cond else 'FAIL'}   {msg}")
    if not cond:
        fails.append(name)


def parse(path):
    with open(os.path.join(ROOT, path)) as f:
        return ast.parse(f.read(), filename=path)


def canonical_values():
    """GATE_TOL/GATE_FRAC/GATE_MIN read from glint_fast.py's AST (no torch needed)."""
    vals = {}
    for node in parse("glint/glint_fast.py").body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in CANON_NAMES \
                and isinstance(node.value, ast.Constant):
            vals[node.targets[0].id] = node.value.value
    return vals


def glint_fast_imports(tree):
    """{bound name: canonical name} over every `from ...glint_fast import ...` in the module."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.split(".")[-1] == "glint_fast":
            for a in node.names:
                out[a.asname or a.name] = a.name
    return out


def module_assigns(tree):
    """{name: value node} for top-level assignments, tuple targets unpacked element-wise."""
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                out[tgt.id] = node.value
            elif isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple) \
                    and len(tgt.elts) == len(node.value.elts):
                for t, v in zip(tgt.elts, node.value.elts):
                    if isinstance(t, ast.Name):
                        out[t.id] = v
    return out


def bound_to_canon(local, canon, assigns, imports):
    """True iff `local` is the canonical constant: gf.<canon> attribute, a name already bound to
    it by an import (aliased or not), or absent from assigns because the ImportFrom itself binds
    it. A numeric literal on the right-hand side is exactly what must fail."""
    v = assigns.get(local)
    if v is None:
        return imports.get(local) == canon
    if isinstance(v, ast.Attribute):
        return v.attr == canon
    if isinstance(v, ast.Name):
        return imports.get(v.id) == canon
    return False            # ast.Constant (a re-inlined literal) and anything else


def compare_literals(scope_node, forbidden):
    """Every raw numeric comparator inside `scope_node` whose value is a gate value."""
    hits = []
    for node in ast.walk(scope_node):
        if not isinstance(node, ast.Compare):
            continue
        for cmp_ in node.comparators:
            if isinstance(cmp_, ast.Constant) and not isinstance(cmp_.value, bool) \
                    and cmp_.value in forbidden:
                hits.append(f"line {cmp_.lineno}: compares against literal {cmp_.value!r}")
    return hits


def main_block(tree):
    """The `if __name__ == "__main__":` block, or None."""
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            left = node.test.left
            if isinstance(left, ast.Name) and left.id == "__name__":
                return node
    return None


def calls_in(func_node):
    names = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            f = node.func
            names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None))
    return names


def source_half():
    canon = canonical_values()
    check("glint_fast declares all three gate constants", set(canon) == set(CANON_NAMES),
          f"found {sorted(canon)}")
    if set(canon) != set(CANON_NAMES):
        return
    forbidden = set(canon.values())

    for path, aliases, gate_fn, scope in FILES:
        tree = parse(path)
        imports = glint_fast_imports(tree)
        assigns = module_assigns(tree)

        for local, cname in aliases.items():
            check(f"{path}: {local} is glint_fast.{cname}",
                  bound_to_canon(local, cname, assigns, imports))

        if scope == "main":
            blk = main_block(tree)
            check(f"{path}: has a __main__ block", blk is not None)
            hits = compare_literals(blk, forbidden) if blk is not None else []
            check(f"{path}: no gate literal compared in __main__", not hits, "; ".join(hits))
            need = {"matched_strict", "GATE_FRAC", "GATE_MIN"}
            check(f"{path}: imports matched_strict/GATE_FRAC/GATE_MIN",
                  need <= set(imports), f"imported: {sorted(imports)}")
        else:
            hits = compare_literals(tree, forbidden)
            check(f"{path}: no gate literal in any comparison", not hits, "; ".join(hits))

        if gate_fn is not None:
            fn = next((n for n in tree.body
                       if isinstance(n, ast.FunctionDef) and n.name == gate_fn), None)
            check(f"{path}: {gate_fn}() exists", fn is not None)
            if fn is not None:
                calls = calls_in(fn)
                check(f"{path}: {gate_fn}() uses matched_strict, not matched",
                      "matched_strict" in calls and "matched" not in calls,
                      f"calls: {sorted(c for c in calls if c)}")


def import_half():
    try:
        import torch                                            # noqa: F401
    except ImportError:
        print("  import half: SKIP (no torch -- glint_fast cannot import; the SOURCE half above "
              "already pinned the bindings)")
        return
    import importlib
    import glint.glint_fast as gf
    for modname, aliases in [
            ("test_streamdriver_vs_offline", {"GATE_FRAC": "GATE_FRAC", "GATE_MIN": "GATE_MIN"}),
            ("test_watchdog_miss_reason", {"GATE_FRAC": "GATE_FRAC", "GATE_MIN": "GATE_MIN"}),
            ("gap_at_scale", {"GATE_FRAC": "GATE_FRAC", "GATE_MIN": "GATE_MIN"}),
            ("gap_on_real", {"GATE_MIN": "GATE_MIN"}),
            ("score_glint_gate", {"TOL": "GATE_TOL", "MIN_FRAC": "GATE_FRAC",
                                  "MIN_REFL": "GATE_MIN"})]:
        # score_xg_gate.py is excluded: its scoring body runs at module level (no __main__
        # guard), so it cannot be imported for inspection; its SOURCE checks above stand alone.
        mod = importlib.import_module(modname)
        for local, cname in aliases.items():
            got, want = getattr(mod, local), getattr(gf, cname)
            check(f"import {modname}.{local} == glint_fast.{cname}", got == want,
                  f"{got!r} vs {want!r}")


if __name__ == "__main__":
    source_half()
    import_half()
    print()
    if fails:
        print(f"FAIL: {len(fails)} check(s): {fails}")
        sys.exit(1)
    print("ALL PASS")
