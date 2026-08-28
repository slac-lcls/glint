"""Which geometry did this run actually get, and does anything corroborate it? -- STATUS.md item 7.

THE FAILURE THIS EXISTS FOR. psana's deployed geometry is often the UNREFINED starting calibration,
while the refinement downstream actually trusts lives only in a `.geom` and is never written back.
Nothing complains. On mfxx49820 r0016 that let blind indexing lock a wrong doubled-*c* cell at
support 23/6294 and REPORT SUCCESS; supplying btx's refined `.geom` took support to 832 and the cell
to [38.3 79.1 80.3] against a truth of [38.4 79.3 79.5]. A wrong geometry does not crash and does not
look wrong -- it quietly moves every q, and blind indexing converges on something else.

WHAT THIS MODULE REFUSES TO DO IS PASS SILENTLY. It reports one of three states, and the third is
the dangerous one that used to be indistinguishable from success:

    CORROBORATED   a .geom was supplied and agrees with psana
    DISAGREE       a .geom was supplied and does NOT agree -- with the disagreement characterised
    UNVERIFIED     no .geom; nothing to compare against. Says so, loudly, rather than saying nothing

THE DISCRIMINATOR, and why it is not an invented threshold. Two geometries can differ two ways, and
only one of them matters:

  * a GLOBAL SCALE / distance term. Benign: `--zdist` exists precisely to set the distance, so this
    is absorbed. It moves EVERY panel the same way.
  * a per-panel SHAPE term -- the tilts and offsets a detector refinement fits. `--zdist` cannot
    touch it, and it is what stops blind indexing converging. Panels disagree with EACH OTHER.

The quantity that separates them is INTER-PANEL DISPERSION relative to the size of the error:
`dispersion / median`, where dispersion is the median absolute deviation of the per-panel signed
medians about their own median. Measured on synthetic geometries where the answer is known:

    pure distance +3%          scale explains 91.6 %   dispersion/median  0.07
    per-quadrant +-2%          scale explains 54.2 %   dispersion/median  1.01
    small per-panel shifts     scale explains 18.1 %   dispersion/median  0.91

Note what that table kills. The obvious discriminator -- "does one global scale explain it?" -- does
NOT work: a real per-quadrant error is still 54 % absorbable, which would have been waved through as
a distance problem. Dispersion separates the same three cases by an order of magnitude, so the 0.3
cut below is the middle of a gap rather than a tuned value.

Measured on the known-bad mfxx49820 r0016 case (S3DF job 34279882, all 2,162,688 pixels, psana
0-end.data vs btx r0016.geom):

    median |rel| in |q|            1.750 %
    mean    rel                   +0.093 %      ~0, so not a scale error
    best global scale k            0.998961
    median |rel| after removing k  1.746 %      -> scale explains 0.2 % of it
    per-quadrant signed median     -2.17 / +2.19 / +2.30 / -2.15 %
    SIGN SPLIT                     8 panels positive, 8 negative
    panel-median spread            9.019 %

The scale term explains essentially nothing and the signs split exactly evenly -- the refinement
term, unmistakably.

Two numbers here differ from the ones in `geom_coords.py`'s docstring (median 3.16 %, max 22.6 %),
and both are right: that measurement pushed btx's PEAK pixels through, this one every pixel. Peaks
sit at higher |q| where the relative error is larger, so the peak-weighted median is higher. The
all-pixel `max` is worse for the opposite reason -- near the beam centre |q| -> 0 and a relative
error means nothing there, which is why this module reports a high PERCENTILE and never the max.

The comparison needs no wavelength: lambda is a common factor in q = (s - kin)/lambda, so it cancels
out of every relative difference. One less argument, one less way to be wrong.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

# Two independently-computed coordinate arrays for the SAME geometry agree to float32 rounding,
# ~1e-7 relative. The control run (psana coords against themselves) came back at exactly 0.0, so
# anything above this is a real difference and not numerical noise. 1e-4 sits ~1000x above the noise
# floor and ~175x below the observed 1.75 % failure, so it separates the two without tuning.
AGREE_REL = 1e-4

# Inter-panel dispersion / median error. Measured 0.07 for a pure distance error against 0.91-1.01
# for three different shape errors, so 0.3 is the middle of an order-of-magnitude gap, not a tuned
# number. Above it, panels disagree with each other and no single distance can reconcile them.
SHAPE_DISPERSION = 0.3

# Reported, but NOT used to decide: a genuine per-quadrant error measured 54% absorbable by one
# global scale, so keying on this would have called the real failure a distance problem.
SCALE_EXPLAINS_ENOUGH = 0.5

_RANGE = re.compile(r"^(\d+)-(\d+|end)\.data$")


def calib_candidates(calib_dir, det_name, ctype="geometry"):
    """Every deployed `<start>-<end>.data` for this detector's `ctype`, and which one covers a run.

    psana's layout is <calib_dir>/<Class>::CalibV1/<DetSource>/<ctype>/<start>-<end>.data, and psana
    resolves the HIGHEST start whose range covers the run. That choice is invisible and is a real
    trap: mfxx49820 deploys both `0-end.data` and a later tilted `8-end.data`, and the btx geometry
    downstream trusts was built on 0-end -- so psana silently hands you the one btx did NOT use.
    Returning every candidate is the point; a single file is a fact worth stating too, because it
    means the geometry has never been re-deployed since the experiment started.
    """
    if not calib_dir:
        return []
    root = Path(calib_dir)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.glob(f"*/*/{ctype}")):
        src = d.parent.name
        # Match loosely: callers pass either psana's full source ("MfxEndstation.0:Epix10ka2M.0") or
        # a DAQ alias ("jungfrau4M"), and the directory is always named for the full source.
        if det_name and not (det_name in src or src in det_name):
            continue
        for f in sorted(d.iterdir()):
            m = _RANGE.match(f.name)
            if not m:
                continue
            lo = int(m.group(1))
            hi = np.inf if m.group(2) == "end" else int(m.group(2))
            try:
                mtime = os.path.getmtime(f)
            except OSError:
                mtime = None
            out.append({"path": str(f), "src": src, "lo": lo, "hi": hi, "mtime": mtime})
    return out


def resolves_to(cands, run):
    """The candidate psana would pick for `run`: covering the run, highest start wins."""
    cover = [c for c in cands if c["lo"] <= run <= c["hi"]]
    return max(cover, key=lambda c: c["lo"]) if cover else None


def _qmag(Xf, Yf, Zf, zdist):
    """|q| per pixel, up to the 1/lambda that cancels in every ratio below.

    Mirrors xtc_core.prep_geometry exactly -- micrometres to metres, Z magnitude from `zdist` with
    psana's sign, kin along that sign -- but without building peak finders, so this can run at
    startup with no cupy and no GPU.
    """
    X = np.asarray(Xf, float).reshape(-1) * 1e-6
    Y = np.asarray(Yf, float).reshape(-1) * 1e-6
    Zc = np.sign(np.nanmean(np.asarray(Zf, float))) * float(zdist)
    r = np.stack([X, Y, np.full(X.shape, Zc)], axis=-1)
    s = r / np.linalg.norm(r, axis=-1, keepdims=True)
    kin = np.array([0.0, 0.0, np.sign(Zc)])
    return np.linalg.norm(s - kin, axis=-1)


def compare(Xp, Yp, Zp, Xg, Yg, Zg, shape, zdist):
    """psana coords vs .geom coords -> how much they differ, and whether it is scale or shape."""
    qp, qg = _qmag(Xp, Yp, Zp, zdist), _qmag(Xg, Yg, Zg, zdist)
    # Drop the beam-centre pixels: |q| -> 0 there, so a RELATIVE difference is unbounded and says
    # nothing about geometry. The 10th percentile is a cut on radius, not on the statistic.
    ok = np.isfinite(qp) & np.isfinite(qg) & (qp > 0)
    if not ok.any():
        return None
    ok &= qp > np.percentile(qp[ok], 10)
    rel = np.full(qp.shape, np.nan)
    rel[ok] = (qg[ok] - qp[ok]) / qp[ok]
    r = rel[ok]

    med = float(np.median(np.abs(r)))
    k = float(np.median(qg[ok] / qp[ok]))                       # best single global scale
    res = (qg[ok] / k - qp[ok]) / qp[ok]
    med_after = float(np.median(np.abs(res)))
    explains = 0.0 if med <= 0 else max(0.0, 1.0 - med_after / med)

    nseg = shape[0] if len(shape) == 3 else 1
    per = []
    rel_s = rel.reshape((nseg, -1))
    for p in range(nseg):
        v = rel_s[p][np.isfinite(rel_s[p])]
        per.append(float(np.median(v)) if v.size else np.nan)
    per = np.array(per)
    fin = per[np.isfinite(per)]

    # How much the panels disagree with EACH OTHER, robustly (MAD about their own median). A pure
    # distance change moves them all together and leaves this near zero; any per-panel structure --
    # quadrants, tilts, shifts -- makes it comparable to the error itself.
    common = float(np.median(fin)) if fin.size else float("nan")
    disp = float(np.median(np.abs(fin - common))) if fin.size else float("nan")

    return {
        "median_rel": med,
        "p99_rel": float(np.percentile(np.abs(r), 99)),
        "mean_rel": float(np.mean(r)),
        "scale_k": k,
        "median_rel_after_scale": med_after,
        "scale_explains": explains,
        "per_panel": per,
        "panel_common": common,
        "panel_dispersion": disp,
        "dispersion_ratio": (disp / med) if med > 0 else 0.0,
        "n_pos": int(np.sum(fin > 0)),
        "n_neg": int(np.sum(fin < 0)),
        "spread": float(np.nanmax(per) - np.nanmin(per)) if fin.size else float("nan"),
        "npix": int(ok.sum()),
    }


def verdict(cmp_):
    """(state, one-line reason) from a compare() result."""
    if cmp_ is None:
        return "UNVERIFIED", "no usable pixels to compare"
    if cmp_["median_rel"] < AGREE_REL:
        return "CORROBORATED", (f"the two geometries agree to {100*cmp_['median_rel']:.4f}% in |q|, "
                                f"below the {100*AGREE_REL:.2f}% float-noise floor")
    # Inter-panel dispersion decides, NOT scale_explains -- see SHAPE_DISPERSION. A real
    # per-quadrant error is still ~54% absorbable by a global scale, so that test would clear it.
    if cmp_["dispersion_ratio"] >= SHAPE_DISPERSION:
        return "DISAGREE", (f"the panels disagree with EACH OTHER (dispersion "
                            f"{100*cmp_['panel_dispersion']:.3f}% vs median "
                            f"{100*cmp_['median_rel']:.3f}%, ratio "
                            f"{cmp_['dispersion_ratio']:.2f}; signs {cmp_['n_pos']}/"
                            f"{cmp_['n_neg']}) -- a detector SHAPE difference (tilts/offsets) that "
                            f"--zdist CANNOT absorb, whatever it is set to")
    return "DISAGREE", (f"but every panel moves together (dispersion ratio "
                        f"{cmp_['dispersion_ratio']:.2f}, common offset "
                        f"{100*cmp_['panel_common']:+.3f}%) -- consistent with a DISTANCE "
                        f"difference, which --zdist sets; check --zdist before anything else")


def report(det_name, run, *, calib_dir=None, geom=None, coords_psana=None, coords_geom=None,
           shape=None, zdist=None, out=print):
    """The startup report. Returns (state, lines); `out=None` to render nothing.

    Deliberately WARNS rather than raising. The comparison is a heuristic about two coordinate sets,
    and this route has been burned before by a check that silently did nothing (gpu_pool geometry,
    5d6c9e0) -- but a check that aborts a 100k-frame run on a heuristic is the opposite failure. The
    number is printed; the human decides.
    """
    lines = []
    cands = calib_candidates(calib_dir, det_name)
    picked = resolves_to(cands, run)

    lines.append(f"geometry provenance for {det_name} run {run}")
    if not calib_dir:
        lines.append("  calib-dir : psana default (not overridden) -- candidates not enumerated")
    elif not Path(calib_dir).is_dir():
        # The worst of the path failures: psana.setOption accepts a nonexistent path without
        # complaint and quietly falls back to the default calib dir, so the run proceeds on a
        # geometry the user believes they overrode.
        lines.append(f"  calib-dir : {calib_dir}")
        lines.append("  *** THIS PATH DOES NOT EXIST. psana accepts it silently and falls back to "
                     "the DEFAULT calib dir, so --calib-dir has had no effect on this run. ***")
    elif not cands:
        lines.append(f"  calib-dir : {calib_dir}")
        lines.append(f"  *** exists, but has NO geometry files for a detector matching "
                     f"{det_name!r}. psana will fall back to the default. Check the detector source "
                     f"name against the <Class>::CalibV1/<source>/ directory. ***")
    else:
        lines.append(f"  calib-dir : {calib_dir}")
        for c in cands:
            hi = "end" if not np.isfinite(c["hi"]) else int(c["hi"])
            mark = " <- psana resolves this" if picked and c["path"] == picked["path"] else ""
            lines.append(f"    {c['lo']}-{hi}.data{mark}")
        if picked is None:
            lines.append(f"  *** NO deployed range covers run {run} -- psana falls back to the "
                         "default calib dir for geometry. ***")
        elif len(cands) > 1:
            lines.append(f"  NOTE {len(cands)} candidates: psana takes the highest start covering the "
                         "run, which is not necessarily the one a downstream refinement was built on")
        else:
            lines.append("  NOTE one candidate only -- geometry has not been re-deployed since it "
                         "was first laid down, so it may be the unrefined starting calibration")
    if calib_dir and not os.path.isabs(str(calib_dir)):
        lines.append("  NOTE --calib-dir is RELATIVE; psana resolves it against the working "
                     "directory, which the launcher changes. Prefer an absolute path.")
    if geom and geom != "<self>" and not Path(str(geom)).is_file():
        lines.append(f"  *** --geom {geom} DOES NOT EXIST as a file. ***")

    # --zdist is the single source of truth for the distance, so where it CAME from matters. If it
    # was read off psana's own coords_z then nothing independent constrains the scale, and a global
    # |q| error rides along invisibly -- there is no second opinion to disagree with it. This is the
    # cxilu8823 r0226 situation exactly.
    if zdist and coords_psana is not None:
        zp = np.asarray(coords_psana[2], float)
        znom = abs(float(np.nanmedian(zp))) * 1e-6
        if znom > 0 and abs(znom - float(zdist)) / float(zdist) < 0.01:
            lines.append(f"  NOTE --zdist {zdist:.6f} m matches psana's own nominal coords_z "
                         f"({znom:.6f} m) to <1%, so it is NOT an independent measurement of the "
                         f"distance -- a global |q| scale error would be invisible here.")

    state = "UNVERIFIED"
    if geom and coords_psana is not None and coords_geom is not None and zdist:
        cmp_ = compare(*coords_psana, *coords_geom, shape, zdist)
        state, why = verdict(cmp_)
        lines.append(f"  --geom    : {geom}")
        if cmp_ is not None:
            lines.append(f"  |q| difference over {cmp_['npix']} pixels (beam-centre 10% excluded):")
            lines.append(f"    median |rel| {100*cmp_['median_rel']:8.3f} %"
                         f"     p99 {100*cmp_['p99_rel']:7.3f} %"
                         f"     mean {100*cmp_['mean_rel']:+7.3f} %")
            lines.append(f"    inter-panel dispersion {100*cmp_['panel_dispersion']:7.3f} %"
                         f"  ratio {cmp_['dispersion_ratio']:.2f}"
                         f"  (>= {SHAPE_DISPERSION} means SHAPE, not distance)")
            lines.append(f"    per-panel signs {cmp_['n_pos']} positive / {cmp_['n_neg']} negative, "
                         f"spread {100*cmp_['spread']:.3f} %")
            lines.append(f"    best global scale {cmp_['scale_k']:.6f} would explain "
                         f"{100*cmp_['scale_explains']:.1f}% -- reported, NOT used to decide")
        lines.append(f"  VERDICT   : {state} -- {why}")
        if state == "DISAGREE":
            lines.append("    GLINT is using the --geom coordinates. psana's differ, so anything "
                         "reading psana's geometry downstream will not agree with this stream.")
    else:
        lines.append("  --geom    : NOT SUPPLIED")
        lines.append("  VERDICT   : UNVERIFIED -- nothing corroborates psana's geometry. This is not "
                     "a claim that it is wrong; it is the absence of any check.")
        lines.append("    On mfxx49820 r0016 the deployed geometry was the unrefined 2021 start and "
                     "blind indexing locked a WRONG cell at support 23/6294 and reported success.")
        lines.append("    Pass --geom with the refined geometry to turn this into a measurement.")

    if out:
        for ln in lines:
            out(ln)
    return state, lines
