"""glint_cli must not drop --cell (or the other sparse-only options) silently when dense mode runs.

WHY THIS EXISTS. The dense front end (hybrid_stream.dense_index) takes no cell, no N-best list, no
cascade and no escalation. The CLI picked it with an explicit --mode dense, or with --mode auto (the
default) once the median frame had >= CLUSTER_MIN peaks, and then threw --cell away without a word:
on 30 real cxidb-17 frames with --cell, --mode dense wrote 7 of 29 crystals on another lattice, every
one labelled `indexed_by = file` (review s8-04). Now:
  * --mode dense with --cell, --escalate, --cascade, a non-default --nbest or --select is refused by
    argparse (exit 2) before any frame is loaded;
  * --mode auto with --cell runs the sparse (known-cell) front end even on dense frames, and says so on
    stderr;
  * --mode auto that picks dense without --cell names any other sparse-only option it is ignoring, on
    stderr;
  * everything else dispatches as before (sparse frames, --mode sparse, default --nbest/--select).

The CLI's own main() runs in-process on --qframes input. glint.glint_fast and glint.hybrid_stream
import torch, which the CPU job does not install, so both are replaced in sys.modules by stand-ins:
CLUSTER_MIN = 3000 (the shipped default), a frame loader, and call-recording front ends. What runs for
real is glint_cli's argument handling and dispatch, and glint.stream.write_stream.

Run: `PYTHONPATH=. python experiments/test_cli_dense_flags.py` (exit 0/1). numpy only.
"""
import contextlib
import io
import os
import sys
import tempfile
import types

import numpy as np

from glint.lattice import cell_to_Ar

CLUSTER_MIN = 3000
CELL = ["79.1", "79.1", "37.9", "90", "90", "90"]
FRAMES = {}                     # path -> list of (n, 3) q arrays, served by the stand-in loader
CALLS = []                      # (front end, kwargs) per call


def _stub_modules():
    rng = np.random.default_rng(0)

    def load(path):
        CALLS.append(("load", {"path": path}))
        return FRAMES[path]

    def _results(frames, images):
        return [{"image": im["image"], "event": im["event"], "M": None, "q": q, "hkl": None}
                for q, im in zip(frames, images)]

    def hybrid_index(frames, images=None, Mc_known=None, nbest=3, cascade=None, escalate=None,
                     select="first", **kw):
        CALLS.append(("hybrid_index", dict(Mc_known=Mc_known, nbest=nbest, cascade=cascade,
                                           escalate=escalate, select=select)))
        return _results(frames, images), {"n": len(frames), "n_idx": 0}

    def dense_index(frames, images=None, warmup=True):
        CALLS.append(("dense_index", {}))
        return _results(frames, images), {"n": len(frames), "n_idx": 0, "mode": "dense"}

    gf = types.ModuleType("glint.glint_fast")
    gf.CLUSTER_MIN, gf.load = CLUSTER_MIN, load
    hs = types.ModuleType("glint.hybrid_stream")
    hs.hybrid_index, hs.dense_index, hs._report = hybrid_index, dense_index, lambda stats, out: None
    sys.modules["glint.glint_fast"], sys.modules["glint.hybrid_stream"] = gf, hs
    # glint.cascade.external_cascade is imported only for --cascade on the sparse path
    cs = types.ModuleType("glint.cascade")
    cs.external_cascade = lambda path: ("cascade", path)
    sys.modules["glint.cascade"] = cs
    return rng


RNG = _stub_modules()
from glint import glint_cli  # noqa: E402  (after the stand-ins are in place)

fails = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        fails.append(name)


def run_cli(argv):
    """main() with this argv: (exit code, stderr text, front-end calls)."""
    del CALLS[:]
    err = io.StringIO()
    old = sys.argv
    sys.argv = ["glint"] + argv
    code = 0
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            glint_cli.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    finally:
        sys.argv = old
    return code, err.getvalue(), [c for c in CALLS if c[0] != "load"], [c for c in CALLS if c[0] == "load"]


with tempfile.TemporaryDirectory() as d:
    dense, sparse = os.path.join(d, "dense.txt"), os.path.join(d, "sparse.txt")
    FRAMES[dense] = [RNG.normal(size=(3500, 3)) * 0.2, RNG.normal(size=(3502, 3)) * 0.2]
    FRAMES[sparse] = [RNG.normal(size=(50, 3)) * 0.2 for _ in range(3)]
    out = os.path.join(d, "o.stream")
    base = lambda src: ["--qframes", src, "-o", out]

    print("--mode auto, dense frames, --cell: the cell is used (sparse front end), and that is said")
    code, err, calls, _ = run_cli(base(dense) + ["--cell"] + CELL)
    check("exit 0", code == 0, (code, err))
    check("hybrid_index ran, dense_index did not", [c[0] for c in calls] == ["hybrid_index"], calls)
    check("hybrid_index got the --cell matrix",
          calls and calls[0][1].get("Mc_known") is not None
          and np.allclose(calls[0][1]["Mc_known"], cell_to_Ar(*map(float, CELL))), calls)
    check("stderr says the dense front end was skipped for --cell",
          "--cell" in err and "sparse" in err and "CLUSTER_MIN" in err, err)
    calls = run_cli(base(dense) + ["--cell", " ".join(CELL)])[2]
    check("one quoted string works the same", [c[0] for c in calls] == ["hybrid_index"], calls)

    print("\n--mode auto, dense frames, no --cell: dense as before; other sparse-only options are named")
    code, err, calls, _ = run_cli(base(dense))
    check("plain: dense_index, nothing on stderr", code == 0 and [c[0] for c in calls] == ["dense_index"]
          and err == "", (code, calls, err))
    for extra, flag in ((["--escalate"], "--escalate"), (["--cascade", "drv"], "--cascade"),
                        (["--nbest", "5"], "--nbest 5"), (["--select", "matched"], "--select matched")):
        code, err, calls, _ = run_cli(base(dense) + extra)
        check(f"{flag}: dense_index ran and stderr says {flag} is IGNORED",
              code == 0 and [c[0] for c in calls] == ["dense_index"] and flag in err and "IGNORED" in err,
              (code, calls, err))
    code, err, calls, _ = run_cli(base(dense) + ["--nbest", "3", "--select", "first"])
    check("default --nbest/--select spelled out: not flagged", code == 0 and err == "", err)

    print("\n--mode dense: sparse-only options are refused before any frame is loaded")
    for extra, flag in ((["--cell"] + CELL, "--cell"), (["--escalate"], "--escalate"),
                        (["--cascade", "drv"], "--cascade"), (["--nbest", "1"], "--nbest 1"),
                        (["--select", "matched"], "--select matched")):
        code, err, calls, loads = run_cli(base(dense) + ["--mode", "dense"] + extra)
        check(f"{flag}: exit 2, named, no front end, no frames loaded",
              code == 2 and flag in err and "--mode dense" in err and not calls and not loads,
              (code, calls, loads, err[-200:]))
    code, err, calls, _ = run_cli(base(dense) + ["--mode", "dense", "--cell"] + CELL + ["--escalate"])
    check("several at once: all named", code == 2 and "--cell" in err and "--escalate" in err, err[-200:])
    code, err, calls, _ = run_cli(base(dense) + ["--mode", "dense", "--nbest", "3"])
    check("--mode dense alone (default --nbest spelled out): runs dense, silent",
          code == 0 and [c[0] for c in calls] == ["dense_index"] and err == "", (code, calls, err))

    print("\nunchanged: sparse frames and --mode sparse")
    code, err, calls, _ = run_cli(base(sparse) + ["--cell"] + CELL + ["--escalate", "--nbest", "5"])
    check("auto on sparse frames with --cell: hybrid_index with every option, nothing on stderr",
          code == 0 and [c[0] for c in calls] == ["hybrid_index"] and err == ""
          and calls[0][1].get("Mc_known") is not None and calls[0][1].get("escalate")
          and calls[0][1].get("nbest") == 5,
          (code, calls, err))
    code, err, calls, _ = run_cli(base(dense) + ["--mode", "sparse", "--cell"] + CELL)
    check("--mode sparse on dense frames with --cell: hybrid_index, nothing on stderr",
          code == 0 and [c[0] for c in calls] == ["hybrid_index"] and err == "", (code, calls, err))

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + str(len(fails)) + '  ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
