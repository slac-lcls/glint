"""End-to-end test of the envbridge-driven path, WITHOUT psana (torch env only).

Drives the exact flow glint_xtc.py uses -- envbridge.call(reader_env, "module:fn") to get q-frames,
then GLINT's real hybrid_index -- but points envbridge at a SYNTHETIC reader in the SAME env (via
Env.shell, no conda activation) whose q come from a planted lysozyme cell. So it proves: the envbridge
call marshals a list of numpy arrays back correctly, and the driver's indexing half recovers the cell.

    PYTHONPATH=<envbridge checkout>:$PYTHONPATH python end_to_end_test.py     # needs envbridge + torch

The psana half (xtc_qreader: reading xtc, per-pixel-coord q, zdist/cframe geometry) is validated
separately on real data -- the gate in the README.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # repo root for `glint`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # experiments/ for test_geom_bridge

try:
    import envbridge
except ImportError:
    sys.exit("envbridge not found; pip install "
             "'git+https://github.com/slac-lcls/drp-benchmarks.git#subdirectory=envbridge' "
             "(or put a checkout on PYTHONPATH)")

from test_geom_bridge import LYSO

ok = True


def check(name, cond):
    global ok
    ok = ok and cond
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


# envbridge worker runs in THIS env (Env.shell no-op activation); it must import _sim_qreader, so put
# this dir on the worker's PYTHONPATH -- exactly how glint_xtc adds the bridge dir for xtc_qreader.
bridge_dir = str(Path(__file__).resolve().parent)
env = envbridge.Env.shell("true", pythonpath=[bridge_dir])

out = envbridge.call(env, "_sim_qreader:run_to_qframes_sim", 30)
frames = [np.asarray(q, float) for q in out["qframes"]]
check("30 q-frames returned across the bridge", out["n_sent"] == 30 and len(frames) == 30)
check("frames are (N,3) float arrays", all(f.ndim == 2 and f.shape[1] == 3 for f in frames))

from glint.hybrid_stream import hybrid_index
from glint.stream import write_stream
from glint.multishot import same_lattice

images = [{"image": "sim://e2e", "event": e} for e in out["events"]]
results, stats = hybrid_index(frames, images, nbest=3)

with tempfile.TemporaryDirectory() as d:
    p = str(Path(d) / "e2e.stream")
    write_stream(results, p)
    check("stream written, non-empty", Path(p).exists() and Path(p).stat().st_size > 0)

indexed = sum(1 for r in results if r.get("M") is not None and same_lattice(np.asarray(r["M"]), LYSO))
check("consensus recovered the planted cell",
      stats.get("Mc") is not None and same_lattice(np.asarray(stats["Mc"]), LYSO))
check(f"planted cell indexed on a majority ({indexed}/{len(frames)})", indexed >= 0.6 * len(frames))

print("ALL PASS" if ok else "FAILURES ABOVE")
raise SystemExit(0 if ok else 1)
