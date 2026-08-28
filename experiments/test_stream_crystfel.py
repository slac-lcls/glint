"""Can CrystFEL actually READ what GLINT writes?

This is the check that was missing. `glint/stream.py` emitted a file that reported crystals and
that no CrystFEL tool could open -- for three independent reasons, none of which the error message
named. A round-trip through GLINT's own reader would have passed the whole time; only the real
binary catches it.

Two layers, so it is useful with or without CrystFEL installed:

  ALWAYS   structural assertions on the bytes -- the geometry block, `--- Begin crystal`, an
           indexed_by CrystFEL will accept, a panel name that exists in the geometry.
  IF FOUND run `process_hkl` on the output and require it to find the crystals. Set CRYSTFEL_BIN,
           or it is looked up on PATH and under the S3DF install root.

Run:  python experiments/test_stream_crystfel.py
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint.stream import write_stream, panel_at, panel_bounds, _origin_panel

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + detail}")
    if not ok:
        FAILS.append(name)


# A two-panel geometry whose FIRST panel does not contain the origin, so a writer that just grabs
# panels[0] is distinguishable from one that resolves the origin properly.
GEOM = """clen = 0.1
res = 5000
adu_per_photon = 1
photon_energy = 9400
q1/min_fs = 512
q1/max_fs = 1023
q1/min_ss = 0
q1/max_ss = 1023
q1/corner_x = 0.5
q1/corner_y = -512.5
q1/fs = x
q1/ss = y
q0/min_fs = 0
q0/max_fs = 511
q0/min_ss = 0
q0/max_ss = 1023
q0/corner_x = -512.5
q0/corner_y = -512.5
q0/fs = x
q0/ss = y
"""


def make_results(n=6):
    rng = np.random.default_rng(0)
    M = np.diag([1 / 38.0, 1 / 79.0, 1 / 79.0])          # lysozyme-ish A^-1 reciprocal basis
    out = []
    for i in range(n):
        hkl = rng.integers(-8, 9, size=(25, 3))
        out.append({"image": "t.cxi", "event": i,
                    "M": None if i % 3 == 2 else np.linalg.inv(M),   # one unindexed in three
                    "q": hkl @ M, "hkl": hkl})
    return out


print("panel resolution")
b = panel_bounds(GEOM)
check("both panels parsed", len(b) == 2, str(b))
check("origin resolves to q0, not the first panel listed", _origin_panel(GEOM) == "q0",
      _origin_panel(GEOM))
check("panel_at picks by position", panel_at(b, 700, 10) == "q1", panel_at(b, 700, 10))
check("panel_at falls back off-detector", panel_at(b, 9999, 9999, "zz") == "zz")
check("no geometry -> p0", _origin_panel(None) == "p0")

print("\nstructure of the written stream")
td = tempfile.mkdtemp()
path = os.path.join(td, "g.stream")
res = make_results()
n_idx = write_stream(res, path, geom_text=GEOM)
txt = open(path).read()
check("returns the indexed count", n_idx == 4, str(n_idx))
check("geometry block present -- without it CrystFEL cannot open the file at all",
      "----- Begin geometry file -----" in txt and "----- End geometry file -----" in txt)
check("geometry block carries OUR panels", "q0/min_fs" in txt and "q1/min_fs" in txt)
check("crystal marker is CrystFEL's", txt.count("--- Begin crystal") == 4)
check("no bare 'Begin crystal' left", "\nBegin crystal" not in txt)
check("end marker matches", txt.count("--- End crystal") == 4)
check("indexed_by is a name CrystFEL accepts (not 'glint')",
      "indexed_by = file" in txt and "indexed_by = glint" not in txt)
check("unindexed frames say so", txt.count("indexed_by = none") == 2)
check("peak-list block present even when empty (else the chunk reads as incomplete)",
      txt.count("Peaks from peak search") == 6 and txt.count("End of peak list") == 6)
check("reflection rows name a panel that EXISTS in the geometry",
      " q0\n" in txt and " p0\n" not in txt)
check("chunks balance", txt.count("----- Begin chunk") == txt.count("----- End chunk") == 6)

print("\nno .geom -> still a readable file, with the fallback detector")
p2 = os.path.join(td, "nogeom.stream")
write_stream(res, p2)
t2 = open(p2).read()
check("geometry block still written", "----- Begin geometry file -----" in t2)
check("falls back to p0", " p0\n" in t2)

print("\nheader is shared with the integrated writer, so the two cannot drift")
from glint.predict import _header as pheader
from glint.stream import _header as sheader
check("both emit the geometry block",
      "----- Begin geometry file -----" in pheader(GEOM) and
      "----- Begin geometry file -----" in sheader(GEOM))
check("the integrated writer still names itself", "predict+integrate" in pheader(GEOM))


def find_crystfel():
    if os.environ.get("CRYSTFEL_BIN"):
        return os.environ["CRYSTFEL_BIN"]
    p = shutil.which("process_hkl")
    if p:
        return os.path.dirname(p)
    for c in sorted(glob.glob("/sdf/group/lcls/ds/tools/crystfel/*/bin/process_hkl"), reverse=True):
        return os.path.dirname(c)
    return None


print("\nreal CrystFEL")
cf = find_crystfel()
if cf is None:
    print("  SKIP  process_hkl not found (set CRYSTFEL_BIN); structural checks above still ran")
else:
    print(f"  using {cf}")
    r = subprocess.run([os.path.join(cf, "process_hkl"), "-i", path,
                        "-o", os.path.join(td, "o.hkl"), "-y", "1"],
                       capture_output=True, text=True, timeout=300)
    err = (r.stderr or "") + (r.stdout or "")
    check("stream opens", "Too much audit information" not in err and "Failed to open" not in err,
          err.strip().splitlines()[:2])
    check("indexer name accepted", "Bad list of indexing methods" not in err,
          err.strip().splitlines()[:2])
    check("no unknown-panel warnings", "No such panel" not in err,
          err.strip().splitlines()[:2])
    check("crystals found -- 4 patterns", " 4 " in err or "4 patterns" in err or
          os.path.getsize(os.path.join(td, "o.hkl")) > 0, err.strip()[-200:])

shutil.rmtree(td, ignore_errors=True)
print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
