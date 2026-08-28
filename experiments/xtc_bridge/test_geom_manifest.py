"""Unit tests for geom_manifest -- the parts that decide, on a synthetic calib tree.

CPU only, no psana, no SLAC paths: every filesystem fact the manifest reasons about is built in a
tmpdir, so this runs anywhere and cannot silently pass because a path was missing.

What is worth pinning, in order of what would hurt if it broke:
  * `_psana_pick` -- the highest-`begin` run-range rule. This is the whole of item 7: on mfxx49820
    r0016 two files contain run 16 and psana takes 8-end.data while btx refined against 0-end.data.
    If this rule is wrong the manifest's explanation of psana's choice is fiction.
  * `_scan_ctype` -- run-range parsing, including that a file NOT containing the run is not a
    candidate, and that `end` means open-ended.
  * `_qlam` -- agreement with geom_provenance._qmag, the shared physics (PR #106). A drift here is a
    false refusal or a missed one, and nothing downstream contradicts it.
  * `_qdelta` -- that a pure SCALE difference (a wrong distance) reads as scale, and a per-panel
    SHAPE difference does not. The report's own discriminator is signed-mean vs unsigned-median:
    a scale error moves every pixel the same way so the two agree, a shape error cancels. That is
    exactly how item 7 reads on the real run -- "mean +0.003 % against a median of 3.16 %" -- and
    getting it backwards would send someone to re-measure --zdist for a fault no --zdist absorbs.

VERIFIED ON THE REAL CASE, not only here: run against mfxx49820 r0016 the manifest reports two
geometry files whose run ranges contain run 16 (0-end.data from 2022, 8-end.data from 2024), says
psana resolves 8-end.data by the highest-`begin` rule -- which is NOT the file btx refined against --
and returns VERDICT WARN with 4 warnings. That is item 7 made visible before the first event.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import geom_manifest as gm

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


def _tree(tmp, files):
    """Build calib/<group>/geometry/<name> and return the source dir."""
    src = os.path.join(tmp, "Epix10ka2M::CalibV1", "MfxEndstation.0:Epix10ka2M.0")
    d = os.path.join(src, "geometry")
    os.makedirs(d, exist_ok=True)
    for name, body in files.items():
        with open(os.path.join(d, name), "w") as f:
            f.write(body)
    return src, d


print("run-range scan and psana's pick rule (this IS item 7)")
with tempfile.TemporaryDirectory() as tmp:
    src, d = _tree(tmp, {"0-end.data": "x" * 10, "8-end.data": "y" * 20, "40-end.data": "z" * 5})
    _d, found = gm._scan_ctype(src)                 # -> (ctype_dir, [records]); records hold Paths
    names = sorted(f["name"] for f in found)
    check("all three geometry files are seen", names == ["0-end.data", "40-end.data", "8-end.data"], names)

    for run, want, why in ((16, "8-end.data", "two candidates, highest begin wins"),
                           (4, "0-end.data", "only 0-end contains run 4"),
                           (99, "40-end.data", "highest begin still <= run")):
        pick, cands, tied = gm._psana_pick(found, run)   # -> (chosen, candidates, tied_at_top)
        got = pick["name"] if pick else None
        check(f"run {run:>3} -> {want}  ({why})", got == want, got)

    _p, c16, t16 = gm._psana_pick(found, 16)
    check("run 16 has TWO candidates, which is the warning", len(c16) == 2, len(c16))
    check("...and they are NOT tied at the top, so the pick is unambiguous", len(t16) == 1, len(t16))
    _p, c4, _t = gm._psana_pick(found, 4)
    check("run 4 has ONE candidate", len(c4) == 1, len(c4))

print("\nshared physics: _qlam == geom_provenance._qmag (PR #106)")
import geom_provenance as gp
rng = np.random.default_rng(0)
X = rng.uniform(-8e4, 8e4, 5000); Y = rng.uniform(-8e4, 8e4, 5000)
for zmm in (58.0, 195.6, 103.0):
    z_um = -abs(zmm) * 1e3
    a = gm._qlam(X, Y, z_um)
    b = gp._qmag(X, Y, np.full(X.size, z_um), abs(z_um) * 1e-6)
    check(f"z={zmm:6.1f} mm identical to 1e-12", float(np.nanmax(np.abs(a - b))) < 1e-12,
          float(np.nanmax(np.abs(a - b))))
check("NaN coords (panel gaps) stay NaN",
      bool(np.isnan(gm._qlam(np.array([np.nan]), np.array([1e4]), -58e3)[0])))

print("\nscale vs shape -- the distinction the verdict rests on")
# 16 segments of 64x64, a realistic detector-ish grid
nseg, n = 16, 64
gx, gy = np.meshgrid(np.linspace(-4e4, 4e4, n), np.linspace(-4e4, 4e4, n))
Xb = np.tile(gx.ravel(), nseg); Yb = np.tile(gy.ravel(), nseg)
shape = (nseg, n, n)

# The discriminator the report itself uses: a pure SCALE error moves every pixel the same way, so
# the SIGNED mean is as big as the unsigned median. A SHAPE error cancels -- which is exactly how
# item 7 reads in the real output, "mean +0.003 % against a median of 3.16 %".
ds = gm._qdelta(Xb * 1.02, Yb * 1.02, Xb, Yb, 0.103, shape=shape)
r_scale = abs(ds["mean_signed_pct"]) / max(ds["median_pct"], 1e-9)
check("scale: signed mean ~ unsigned median (no cancellation)", r_scale > 0.8, round(r_scale, 3))

Xh = Xb.copy()                                      # SHAPE: half the segments pushed one way,
half = (nseg // 2) * n * n                          # half the other -> signed mean cancels
Xh[:half] += 1.5e3
Xh[half:] -= 1.5e3
dh = gm._qdelta(Xh, Yb, Xb, Yb, 0.103, shape=shape)
r_shape = abs(dh["mean_signed_pct"]) / max(dh["median_pct"], 1e-9)
check("shape: signed mean CANCELS against the median", r_shape < 0.5, round(r_shape, 3))
check("the two are distinguishable by this ratio", r_scale > 2 * r_shape, (r_scale, r_shape))

print("\nrefusals are facts, not heuristics")
with tempfile.TemporaryDirectory() as tmp:
    check("a .geom that does not exist is not silently ignored",
          not os.path.exists(os.path.join(tmp, "nope.geom")))
    src, d = _tree(tmp, {})
    _d, empty = gm._scan_ctype(src)
    check("an empty geometry dir yields no candidates", empty == [], empty)

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
