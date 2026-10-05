"""glint/__init__.py loads its public names on first use, so a numpy-only submodule imports without SciPy.

Runs each case in a fresh interpreter with `scipy` blocked (sys.modules['scipy'] = None):
  - `import glint.geom` and `import glint.predict` succeed (on the eager __init__ both failed on scipy.optimize);
  - with SciPy available, every name in glint.__all__ still resolves, and an unknown name raises AttributeError.
No torch, no GPU, no data.

  PYTHONPATH=. python experiments/test_lazy_exports.py
"""
import os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def run(code):
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1:] or [""]


def check(name, ok, detail=""):
    print(("ok    " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


for mod in ("glint.geom", "glint.predict"):
    rc, last = run(f"import sys; sys.modules['scipy'] = None; import {mod}")
    check(f"import {mod} without SciPy", rc == 0, last[0])

try:
    import scipy  # noqa: F401
    rc, last = run("import glint\n"
                   "for n in glint.__all__: getattr(glint, n)\n"
                   "try:\n    glint.no_such_name\nexcept AttributeError:\n    print('attr-ok')\n")
    check("every name in glint.__all__ resolves; unknown names raise AttributeError", rc == 0 and last[0] == "attr-ok", last[0])
except ImportError:
    print("SKIP  __all__ resolution (needs SciPy)")

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
