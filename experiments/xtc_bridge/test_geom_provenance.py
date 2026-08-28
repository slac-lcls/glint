"""Does the geometry-provenance check separate a benign distance error from a real shape error?

STATUS.md item 7. Runs on synthetic coordinates -- no psana, no cupy, no GPU -- because the property
under test is arithmetic, not data: given two coordinate sets, does `compare` say the right thing?

The three cases that matter, and the middle one is why this check exists at all:

  IDENTICAL        must be CORROBORATED. A check that flags a good geometry gets switched off.
  PURE DISTANCE    the detector is at a different clen. `--zdist` sets the distance, so this is
                   benign -- it must NOT be reported as a shape error, or every run with a slightly
                   wrong --zdist cries wolf.
  PER-QUADRANT     tilts/offsets, the term a refinement fits. `--zdist` cannot absorb it and blind
                   indexing does not converge through it. This is the real failure and must be
                   called as such.

The per-quadrant case is built to mimic the measured mfxx49820 r0016 signature: +-2% alternating by
quadrant, 8 panels positive and 8 negative, a global scale explaining almost none of it.
"""
from __future__ import annotations

import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import geom_provenance as gp

FAILS = []
NSEG, H, W = 16, 352, 384
PITCH_UM, ZDIST = 100.0, 0.1


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + detail}")
    if not ok:
        FAILS.append(name)


def base_coords():
    """A plausible tiled detector: 16 panels in a 4x4 grid, beam through the middle."""
    yy, xx = np.mgrid[0:H, 0:W]
    X = np.empty((NSEG, H, W)); Y = np.empty((NSEG, H, W))
    for p in range(NSEG):
        qx, qy = p % 4, p // 4
        X[p] = (xx + (qx - 2) * W) * PITCH_UM
        Y[p] = (yy + (qy - 2) * H) * PITCH_UM
    Z = np.full((NSEG, H, W), -ZDIST * 1e6)
    return X, Y, Z


X0, Y0, Z0 = base_coords()
shape = (NSEG, H, W)

print("IDENTICAL geometries")
c = gp.compare(X0, Y0, Z0, X0.copy(), Y0.copy(), Z0.copy(), shape, ZDIST)
st, why = gp.verdict(c)
check("median |rel| is 0", c["median_rel"] == 0.0, f"{c['median_rel']:.3e}")
check("verdict CORROBORATED", st == "CORROBORATED", f"{st}: {why}")

print("\nPURE DISTANCE error (detector 3% further away) -- benign, --zdist absorbs it")
# A distance change is a radial scale on the transverse coords at fixed zdist: every panel moves the
# same way, so the signs must NOT split.
Xd, Yd = X0 * 1.03, Y0 * 1.03
c = gp.compare(X0, Y0, Z0, Xd, Yd, Z0, shape, ZDIST)
st, why = gp.verdict(c)
check("detected as a difference", c["median_rel"] > gp.AGREE_REL, f"{c['median_rel']:.3e}")
check("panels move TOGETHER (low dispersion)",
      c["dispersion_ratio"] < gp.SHAPE_DISPERSION, f"ratio {c['dispersion_ratio']:.2f}")
check("signs do NOT split (all panels one way)",
      c["n_pos"] == 0 or c["n_neg"] == 0, f"{c['n_pos']} pos / {c['n_neg']} neg")
check("verdict names DISTANCE, not shape", "DISTANCE" in why, why)

print("\nPER-QUADRANT error (+-2% alternating) -- the real mfxx49820 failure mode")
Xq, Yq = X0.copy(), Y0.copy()
for p in range(NSEG):
    f = 1.02 if (p // 4) % 2 else 0.98        # quadrants alternate, as measured
    Xq[p] *= f; Yq[p] *= f
c = gp.compare(X0, Y0, Z0, Xq, Yq, Z0, shape, ZDIST)
st, why = gp.verdict(c)
check("detected as a difference", c["median_rel"] > gp.AGREE_REL, f"{c['median_rel']:.3e}")
check("panels disagree with EACH OTHER (high dispersion)",
      c["dispersion_ratio"] >= gp.SHAPE_DISPERSION, f"ratio {c['dispersion_ratio']:.2f}")
check("scale_explains would NOT have caught it -- why dispersion is the discriminator",
      c["scale_explains"] >= gp.SCALE_EXPLAINS_ENOUGH,
      f"scale explains only {100*c['scale_explains']:.1f}%, so this assumption changed")
check("signs split 8/8", c["n_pos"] == 8 and c["n_neg"] == 8,
      f"{c['n_pos']} pos / {c['n_neg']} neg")
check("verdict names SHAPE and says --zdist cannot fix it",
      "SHAPE" in why and "CANNOT" in why, why)
print(f"    (median {100*c['median_rel']:.3f}%, dispersion ratio {c['dispersion_ratio']:.2f}, "
      f"spread {100*c['spread']:.3f}% -- measured real case: 1.750% / ~1.26 / 9.019%)")

print("\nthe two cases must be DISTINGUISHABLE, not merely both flagged")
cd = gp.compare(X0, Y0, Z0, Xd, Yd, Z0, shape, ZDIST)
cq = gp.compare(X0, Y0, Z0, Xq, Yq, Z0, shape, ZDIST)
check("dispersion ratio separates them by an order of magnitude",
      cq["dispersion_ratio"] / max(cd["dispersion_ratio"], 1e-9) > 5,
      f"distance {cd['dispersion_ratio']:.2f} vs quadrant {cq['dispersion_ratio']:.2f}")
check("scale_explains does NOT separate them (the discarded discriminator)",
      not (cd["scale_explains"] - cq["scale_explains"] > 0.5),
      f"distance {cd['scale_explains']:.3f} vs quadrant {cq['scale_explains']:.3f} -- if this now "
      f"separates, revisit which quantity the verdict keys on")
check("sign split separates them",
      (cd["n_pos"] == 0 or cd["n_neg"] == 0) and cq["n_pos"] > 0 and cq["n_neg"] > 0)

print("\nbeam-centre pixels must not dominate")
# |q| -> 0 at the beam centre makes a RELATIVE difference unbounded. Put the beam ON a pixel and
# check the statistic stays finite and close to the intended 2%.
Xc, Yc, Zc = base_coords()
Xc[0, 0, 0] = 0.0; Yc[0, 0, 0] = 0.0
Xq2, Yq2 = Xc * 1.02, Yc * 1.02
c = gp.compare(Xc, Yc, Zc, Xq2, Yq2, Zc, shape, ZDIST)
check("median stays finite", np.isfinite(c["median_rel"]), str(c["median_rel"]))
check("p99 is reported instead of max, and is sane",
      np.isfinite(c["p99_rel"]) and c["p99_rel"] < 1.0, f"p99={c['p99_rel']:.3e}")

print("\nrun-range file resolution")
import tempfile, pathlib
with tempfile.TemporaryDirectory() as td:
    d = pathlib.Path(td) / "Epix10ka2M::CalibV1" / "MfxEndstation.0:Epix10ka2M.0" / "geometry"
    d.mkdir(parents=True)
    (d / "0-end.data").write_text("x")
    (d / "8-end.data").write_text("x")
    cands = gp.calib_candidates(td, "MfxEndstation.0:Epix10ka2M.0")
    check("finds both candidates", len(cands) == 2, str(len(cands)))
    # psana takes the HIGHEST start covering the run -- the trap that hands you 8-end when btx
    # built on 0-end.
    check("run 16 resolves to 8-end (psana's rule)",
          gp.resolves_to(cands, 16)["lo"] == 8, str(gp.resolves_to(cands, 16)))
    check("run 3 resolves to 0-end", gp.resolves_to(cands, 3)["lo"] == 0)
    st, lines = gp.report("MfxEndstation.0:Epix10ka2M.0", 16, calib_dir=td, out=None)
    check("no --geom -> UNVERIFIED", st == "UNVERIFIED", st)
    check("report warns about the multi-candidate choice",
          any("candidates" in l for l in lines))
    check("UNVERIFIED report does not imply the geometry is fine",
          any("absence of any check" in l for l in lines))

print("\npath failures psana accepts SILENTLY")
with tempfile.TemporaryDirectory() as td:
    st, lines = gp.report("det", 16, calib_dir=str(pathlib.Path(td) / "nope"), out=None)
    check("nonexistent --calib-dir is called out",
          any("DOES NOT EXIST" in l for l in lines), " | ".join(lines))
    d2 = pathlib.Path(td) / "Epix10ka2M::CalibV1" / "det" / "geometry"
    d2.mkdir(parents=True)
    (d2 / "20-end.data").write_text("x")
    st, lines = gp.report("det", 16, calib_dir=td, out=None)
    check("no range covering the run is called out",
          any("NO deployed range covers" in l for l in lines), " | ".join(lines))
    st, lines = gp.report("det", 16, calib_dir="rel/ative", out=None)
    check("relative --calib-dir is called out", any("RELATIVE" in l for l in lines))

print("\n--zdist independence")
Zn = np.full((NSEG, H, W), -0.1234 * 1e6)
st, lines = gp.report("det", 1, calib_dir=None, coords_psana=(X0, Y0, Zn), zdist=0.1234, out=None)
check("zdist taken from psana coords_z is flagged as not independent",
      any("NOT an independent" in l for l in lines), " | ".join(lines[-2:]))
st, lines = gp.report("det", 1, calib_dir=None, coords_psana=(X0, Y0, Zn), zdist=0.0999, out=None)
check("a genuinely independent zdist is NOT flagged",
      not any("NOT an independent" in l for l in lines))

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
