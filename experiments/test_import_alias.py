"""`import fftindex` must hand control back to the caller (review s7-04).

fftindex/__init__.py is the back-compat alias: it imports every glint submodule so that
`fftindex.<mod> is glint.<mod>`, and it catches Exception so optional-dependency modules (cupy, torch,
numba) can fail quietly. SystemExit is not an Exception. glint/test_first_index_per_group.py used to
run its checks at module scope and end in sys.exit(...), so wherever torch was installed `import
fftindex` printed that module's report and ended the program with status 0. A legacy script that
imported the alias and then failed still exited 0.

Three layers, so the test can fail on both CI runners:

  STATIC   No module in glint/ or fftindex/ calls sys.exit / os._exit / exit / quit, or raises
           SystemExit, at import time, meaning outside a function body and outside
           `if __name__ == "__main__":`. Reads source only. This is the layer that fails on the
           torch-less CPU runner, where the old module's `import torch` raised first and hid its exit.
  MODULES  In a child process every glint submodule either imports or raises an Exception (an
           optional dependency missing is fine). None may end the process. With torch installed
           (the torch-CPU job, a dev box) this is the runtime form of STATIC.
  ALIAS    Each in a child process:
           - `import fftindex; print(...)` reaches the print.
           - `fftindex is glint` and `fftindex.geom is glint.geom`.
           - A script that imports the alias and then exits 3 still exits 3.
           - `import glint` alone is unaffected.

The child processes are not run under run_ci_locally.py's import block, so on a machine with torch
they import the real thing. That is the stronger case, not a weaker one.

    PYTHONPATH=. python experiments/test_import_alias.py
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join([ROOT] + [p for p in os.environ.get(
    "PYTHONPATH", "").split(os.pathsep) if p]), PYTHONDONTWRITEBYTECODE="1")
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + detail}")
    if not ok:
        FAILS.append(name)


def child(code, timeout=600):
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=ENV, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def tail(s, n=2):
    lines = [l for l in s.splitlines() if l.strip()]
    return " | ".join(lines[-n:])


# ------------------------------------------------------------------------------------- STATIC
def _is_main_guard(node):
    """`if __name__ == "__main__":`, either operand order."""
    if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)):
        return False
    sides = [node.test.left, node.test.comparators[0]]
    return (any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
            and any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides))


def import_time_exits(src, filename):
    """[(line, what)] for every exit reachable at import time in this module's source."""
    tree = ast.parse(src, filename)
    mods = {"sys": "sys", "os": "os"}                       # local name -> module, for `import sys as _s`
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in ("sys", "os"):
                    mods[a.asname or a.name] = a.name
    bare = {"exit", "quit"}
    for node in ast.walk(tree):                             # `from sys import exit as bye`
        if isinstance(node, ast.ImportFrom) and node.module in ("sys", "os"):
            for a in node.names:
                if a.name in ("exit", "_exit"):
                    bare.add(a.asname or a.name)
    functions = {node.name: node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    found = []
    visited_functions = set()

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return                                          # runs when called, not when imported
        if _is_main_guard(node):
            for n in node.orelse:                           # the else branch DOES run on import
                visit(n)
            return
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                    and mods.get(f.value.id) in ("sys", "os") and f.attr in ("exit", "_exit")):
                found.append((node.lineno, f"{f.value.id}.{f.attr}(...)"))
            elif isinstance(f, ast.Name) and f.id in bare:
                found.append((node.lineno, f"{f.id}(...)"))
            elif isinstance(f, ast.Name) and f.id in functions and f.id not in visited_functions:
                visited_functions.add(f.id)
                for statement in functions[f.id].body:
                    visit(statement)
        if isinstance(node, ast.Raise) and node.exc is not None:
            e = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(e, ast.Name) and e.id == "SystemExit":
                found.append((node.lineno, "raise SystemExit"))
        for c in ast.iter_child_nodes(node):
            visit(c)

    visit(tree)
    return sorted(found)


print("STATIC: no import-time exit anywhere in glint/ or fftindex/")
# The detector has to be able to fail, or a pass means nothing.
_probe = ("import sys as _s\nfrom os import _exit\nX = 1\nfor i in range(1):\n    _s.exit(0)\n"
          "def f():\n    _s.exit(1)\nclass C:\n    raise SystemExit\n"
          "if __name__ == '__main__':\n    _s.exit(2)\nelse:\n    _exit(3)\n"
          "def main():\n    _s.exit(4)\nmain()\n")
check("detector sees exits in a loop, class body, main-guard else and called main, not an uncalled def or the guard",
      [ln for ln, _ in import_time_exits(_probe, "<probe>")] == [5, 9, 13, 15],
      str(import_time_exits(_probe, "<probe>")))
n_files = 0
for pkg in ("glint", "fftindex"):
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, pkg)):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ROOT)
            n_files += 1
            hits = import_time_exits(open(path, encoding="utf-8").read(), path)
            if hits:
                check(f"{rel} does not exit at import", False,
                      ", ".join(f"line {ln}: {w}" for ln, w in hits))
print(f"  scanned {n_files} files")
check("found the package sources to scan", n_files > 40, str(n_files))

# ------------------------------------------------------------------------------------ MODULES
print("\nMODULES: each glint submodule imports or raises; none ends the process")
_WALK = r'''
import importlib, pkgutil
import glint
names = sorted(m.name for m in pkgutil.iter_modules(glint.__path__))
print("NAMES " + " ".join(names), flush=True)
for name in names:
    print("BEGIN " + name, flush=True)
    try:
        importlib.import_module("glint." + name)
    except SystemExit as e:                 # report it and go on, so every offender is named
        print("EXIT %s %r" % (name, e.code), flush=True)
        continue
    except Exception as e:                  # optional dependency missing etc.: allowed
        msg = (str(e).splitlines() or [""])[0][:100]
        print("RAISED %s %s: %s" % (name, type(e).__name__, msg), flush=True)
        continue
    print("END " + name, flush=True)
print("DONE", flush=True)
'''
rc, out, err = child(_WALK)
lines = out.splitlines()
names = next((l.split()[1:] for l in lines if l.startswith("NAMES ")), [])
ended = {l.split()[1] for l in lines if l.startswith("END ")}
raised = [l[len("RAISED "):] for l in lines if l.startswith("RAISED ")]
exited = [l[len("EXIT "):] for l in lines if l.startswith("EXIT ")]
begun = [l.split()[1] for l in lines if l.startswith("BEGIN ")]
check("the walk ran to the end (no module killed the process outright)",
      rc == 0 and "DONE" in lines,
      f"rc={rc}; last module begun: {begun[-1] if begun else None}; {tail(out + err)}")
check(f"{len(names)} submodules listed", len(names) > 40, str(len(names)))
check("no submodule raises SystemExit on import", not exited, "; ".join(exited))
print(f"  imported {len(ended)}, raised an Exception {len(raised)} (allowed: optional dependencies)")
for r in raised:
    print(f"        {r}")

# -------------------------------------------------------------------------------------- ALIAS
print("\nALIAS: the fftindex shim returns control")
SENT = "ALIVE_AFTER_IMPORT"
rc, out, err = child(f"import glint; print({SENT!r})")
check("`import glint` reaches the next statement", rc == 0 and SENT in out, f"rc={rc} {tail(out + err)}")
rc, out, err = child(f"import fftindex; print({SENT!r})")
check("`import fftindex` reaches the next statement", rc == 0 and SENT in out, f"rc={rc} {tail(out + err)}")
check("`import fftindex` prints nothing of its own", out.strip() == SENT, tail(out, 3))
rc, out, err = child("import sys, fftindex, glint, fftindex.geom\n"
                     "assert fftindex is glint, 'fftindex is not the glint package'\n"
                     "assert sys.modules['fftindex.geom'] is glint.geom, 'fftindex.geom is a copy'\n"
                     f"print({SENT!r})")
check("fftindex is glint and fftindex.geom is glint.geom", rc == 0 and SENT in out,
      f"rc={rc} {tail(out + err)}")
rc, out, err = child("import fftindex\nraise SystemExit(3)")
check("a script that imports fftindex and then exits 3 exits 3", rc == 3, f"rc={rc} {tail(out + err)}")
# The shim skips glint's in-package test scripts, so a future one that forgets its main guard cannot
# do this again. Without torch that module fails to import anyway, so this bites only where torch is.
rc, out, err = child("import sys, fftindex\n"
                     "print('TESTMODS', sorted(k for k in sys.modules if '.test_' in k))")
check("the shim imports no test_* module", rc == 0 and "TESTMODS []" in out, f"rc={rc} {tail(out + err)}")

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
