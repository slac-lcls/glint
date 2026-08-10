"""STARTUP GEOMETRY MANIFEST for the psana1 xtc route -- metadata only, no frames, always runs.

WHY. STATUS.md item 7: on mfxx49820 r0016 psana's deployed geometry is the UNREFINED 2021 starting
calibration; blind indexing locked a wrong doubled-c cell at support 23/6294 and REPORTED SUCCESS.
Supplying btx's refined `.geom` took support to 832 and the cell to [38.3 79.1 80.3] against a truth
of [38.4 79.3 79.5]. The enabling fault is not the metrology -- it is that NOTHING RECORDS WHICH
GEOMETRY A RUN USED. The reader branches between the `.geom` and psana at xtc_qreader_psana1.py:198
and prints nothing; glint_xtc.py prints only frame counts; the `.stream` header carries no geometry
fingerprint. Two runs that differ ONLY in geometry are indistinguishable from their logs.

This module prints a MANIFEST before the first event: which source supplies X/Y, which calib dir
psana is really using, which detector SOURCE it resolved, every `<begin>-<end>.data` in that source's
`geometry/` ctype with size/mtime/sha1, which one psana picked and by what rule, and how that
geometry's deployment date compares with the sibling ctypes beside it. It costs a handful of `stat`
calls and two small text reads. It runs unconditionally, because "it printed nothing" is exactly how
item 7 stayed silent.

WHAT IT FLAGS, all from metadata:
  * more than one candidate file whose run range contains the run -- the 0-end.data / 8-end.data case
    the reader's own comment describes (xtc_qreader_psana1.py:129-133), where psana takes the higher
    `begin` and btx built on the lower one;
  * a `0-end.data`-only deployment: a starting calibration that was never superseded (item 7's shape);
  * geometry older than every sibling ctype (`pedestals`, `pixel_gain`, ...) in the same source dir --
    the in-data evidence that everything about this detector was redeployed except its metrology;
  * a calib dir that does not exist, is relative, is empty of `*::CalibV1`, or disagrees with what
    psana reports through `env.calibDir()`;
  * a `--det` alias that resolves to a source the calib tree is not keyed on;
  * a resolution replica that disagrees with psana's own `det.geometry(run).path`;
  * `--geom` absent -- i.e. no second opinion exists and the geometry is unverifiable by any means.

Optionally (`cross_check=True`, still frame-free) it MEASURES the |q| disagreement between the two
geometries that are in memory at the same instant -- `.geom` vs psana, and psana's pick vs the other
candidate files -- which is the exact quantity item 7 was diagnosed from, computed at startup instead
of never.

NUMERIC ANCHORS. Exactly three numbers here are thresholds, and all three are measured, not chosen:

  MEDIAN_AGREE_PCT = 0.025   README.md:105, the xtc2/psana2 leg of this repo's real-data gate: on a
                             run whose deployed psana geometry IS the trusted refined fit, per-pixel
                             |q| reproduces it to max 0.025 %, median 0.011 %. That is what "the same
                             metrology" measures as, end to end, in this codebase.
  MEDIAN_BREAK_PCT = 3.16    geom_coords.py docstring / STATUS.md item 7: psana's 0-end.data against
                             btx's refined .geom is median |rel| 3.16 % in |q| (max 22.6 %, mean rel
                             +0.003 %), and AT THAT LEVEL blind indexing demonstrably locked the wrong
                             cell on this repo's own reference run.
  SEG_BREAK_PCT    = 3.0     The SAME measurement, per segment: the disagreement is signed per
                             QUADRANT, segments 0-3 and 12-15 one way and 4-11 the other, +-3-5 %.
                             3 is the LOW END of that measured range, used as the per-segment anchor.

Between the anchors there is no measurement either way, so this module PRINTS THE NUMBER AND DECLINES
TO JUDGE rather than inventing a cut.

WHY THE VERDICT IS PER SEGMENT AND NOT POOLED, which is not a stylistic choice. Item 7's fault is
organised by QUADRANT, so a pooled median dilutes it by exactly the fraction of the detector that is
correct: a fixture in which 8 of 16 segments are displaced by 3 % and the other 8 are exact measures
a pooled median of 0.000 % -- clean, on a geometry that would wreck indexing. The reference case
happened to have EVERY quadrant displaced, which is the only reason its pooled median (3.16 %) was
representative. So the verdict runs on `max |per-segment signed median|` whenever the segment axis is
known, and falls back to the pooled outer-half median only when it is not -- and the report says
which statistic decided. Both are printed either way, together with the segment spread: a pure scale
error gives spread ~0 with a nonzero mean, item 7's fault gives the opposite (mean +0.003 % against a
median of 3.16 %), and the reader can see which one they have.

POPULATION CAVEAT, stated because it is load-bearing: the 3.16 % reference was measured on btx's own
PEAK pixels. This module measures over detector pixels -- a different population. An outer-half
statistic (pixels above the reference geometry's own median |q|, where indexable peaks live and where
|q| is not diverging toward the beam centre) is reported as the closer analogue, and the comparison
is offered as an order of magnitude, not an identity.

STATUS VOCABULARY, and the rule that makes it worth anything:
    OK       checked, and it agrees with a measured reference
    WARN     checked, and it is suspicious or unverifiable FOR THIS RUN -- the run continues
    UNKNOWN  could not be checked -- printed AS LOUDLY AS A WARN, and counts against `ok`
    REFUSE   provably broken -- raises GeometryManifestError
    NOTE     an unconditional property of this ROUTE, true of every run, printed but NOT counted
A check that could not run must never read as healthy. (gpu_pool.py already learned this lesson with
`_geom_init_error`; see `reuse_note()` at the bottom for why its monitor cannot be reused here.)

NOTE exists because a check that cries WARN on every single run teaches people to ignore it, and
then item 7 is silent again for a different reason. The route-invariant hazards -- Z is flattened to
--zdist; the mask comes from psana even under --geom; setOption is process-sticky; three .geom
parsers disagree -- are real and are printed in full every time, but they say nothing about THIS run,
so they do not move the verdict. Anything that distinguishes this run from a healthy one is a WARN.
One interaction is deliberate: `never-redeployed` and `geometry-stale` are heuristics about a file's
age, and a positive |q| measurement against a trusted .geom SUPERSEDES them -- when a cross-check
comes back OK they are demoted to NOTE, because a measurement beats a mtime.

REFUSALS -- only facts, never heuristics. Four, and they are named in the output:
  R1  `--geom` names a path that does not exist, or that parses to zero panels.
  R2  `--calib-dir` names a path that does not exist. psana.setOption() accepts any string, so the
      override the user explicitly asked for is a silent no-op.
  R3  `--calib-dir` exists but holds no `*::CalibV1` directory at all -- likewise provably a no-op.
  R4  No geometry is reachable from ANY source: no `.geom`, and no calib file whose run range
      contains this run. det.coords_x/y/z return None and prep_geometry dies mid-run; failing here is
      the same failure, an hour earlier, with a reason.
Everything else -- one candidate or five, a 3 % disagreement, a stale mtime -- WARNS.

CALL SITE (experiments/xtc_bridge/xtc_qreader_psana1.py, in run_to_qframes_psana1, immediately after
the `psana.Detector(det)` try/except and before the event loop):

    import geom_manifest
    geom_manifest.check_geometry(detector=detector, run=run, exp=exp, det_name=det,
                                 calib_dir=calib_dir, geom=geom, env=ds.env(), zdist=zdist,
                                 rank=rank, nranks=nranks)

Standalone triage (answers "was it the geometry?" for a finished run):
    python geom_manifest.py --exp cxilu8823 --run 226 --det Jungfrau --zdist 0.246
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

CTYPE = "geometry"
_FN = re.compile(r"^(\d+)-(\d+|end)\.data$")
_GROUP = re.compile(r"^(.+)::CalibV\d+$")

# --- the only three thresholds in this file; all measured, see the module docstring ---
MEDIAN_AGREE_PCT = 0.025      # README.md:105  -- psana2 leg; deployed geometry IS the refined fit
MEDIAN_BREAK_PCT = 3.16       # geom_coords.py -- mfxx49820 r0016; pooled median that broke indexing
SEG_BREAK_PCT = 3.0           # same measurement per QUADRANT: +-3-5 %; 3 is the low end of the range

OK, WARN, UNKNOWN, REFUSE, NOTE = "OK", "WARN", "UNKNOWN", "REFUSE", "NOTE"


class GeometryManifestError(RuntimeError):
    """Raised for R1-R4 -- provably broken geometry inputs. Never raised on a heuristic."""


# ---------------------------------------------------------------- small helpers

def _iso(ts):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "?"


def _sha1(path):
    """Short content fingerprint. Geometry files are kilobytes, so this is free -- and it is what
    makes two runs' logs comparable at all: a path can be identical while the file behind it was
    mutated in place (experiment calib areas are group-writable and get updated during a beamtime)."""
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError as e:
        return f"unreadable({e.errno})"
    return h.hexdigest()[:12]


def _rng(f):
    return f"[{f['begin']}, {'end' if f['end'] == np.inf else int(f['end'])}]"


def _calib_groups(calib_dir):
    """-> [(group_type, source, src_dir)] for every `<type>::CalibV1/<source>` under calib_dir.

    The real layout, measured on S3DF for mfxx49820:
        calib/Epix10ka2M::CalibV1/MfxEndstation.0:Epix10ka2M.0/geometry/0-end.data
    """
    out = []
    root = Path(calib_dir)
    if not root.is_dir():
        return out
    try:
        groups = sorted(root.iterdir())
    except OSError:
        return out
    for g in groups:
        m = _GROUP.match(g.name)
        if not (m and g.is_dir()):
            continue
        try:
            srcs = sorted(g.iterdir())
        except OSError:
            continue
        for s in srcs:
            if s.is_dir():
                out.append((m.group(1), s.name, s))
    return out


def _scan_ctype(src_dir, ctype=CTYPE):
    """-> (ctype_dir_or_None, [file records]) for `<src_dir>/<ctype>/<begin>-<end>.data`."""
    d = Path(src_dir) / ctype
    if not d.is_dir():
        return None, []
    files = []
    try:
        entries = sorted(d.iterdir())
    except OSError:
        return d, []
    for f in entries:
        m = _FN.match(f.name)
        if not m:
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        files.append({"path": f, "name": f.name, "begin": int(m.group(1)),
                      "end": np.inf if m.group(2) == "end" else float(m.group(2)),
                      "mtime": st.st_mtime, "size": st.st_size})
    files.sort(key=lambda f: (f["begin"], f["end"]))
    return d, files


def _psana_pick(files, run):
    """Replicate PSCalib's file finder: among files whose [begin,end] contains `run`, the one with
    the HIGHEST `begin` wins. Returns (chosen, candidates, tied_at_top).

    This replica exists to EXPLAIN psana's answer, not to replace it. When
    `det.geometry(run).path` is available the two are compared and any disagreement is reported with
    both paths, because a replica that silently substitutes itself for the real resolver is just
    another way to be confidently wrong. `tied_at_top` with len > 1 is genuinely ambiguous -- the
    tie-break is undocumented -- and is flagged rather than guessed at.
    """
    cands = [f for f in files if f["begin"] <= run <= f["end"]]
    if not cands:
        return None, [], []
    top = max(f["begin"] for f in cands)
    tied = [f for f in cands if f["begin"] == top]
    chosen = min(tied, key=lambda f: (f["end"], f["name"]))
    return chosen, cands, tied


def _sibling_ctypes(src_dir):
    """-> [(ctype, n_files, newest_mtime)] for every ctype dir in the same source directory.

    The in-data reference point for "geometry was never redeployed", with no invented number: on a
    beamtime where pedestals were redeployed during the run and geometry still dates from detector
    installation, the two mtimes say so by themselves.
    """
    out = []
    try:
        entries = sorted(Path(src_dir).iterdir())
    except OSError:
        return out
    for d in entries:
        if not d.is_dir():
            continue
        newest, n = 0.0, 0
        try:
            for f in d.iterdir():
                if f.is_file():
                    n += 1
                    newest = max(newest, f.stat().st_mtime)
        except OSError:
            pass
        out.append((d.name, n, newest))
    return out


# ------------------------------------------------------- psana / geometry access

def _det_source(detector):
    """psana's RESOLVED source string, e.g. 'MfxEndstation.0:Epix10ka2M.0'.

    The calib tree is keyed on the SOURCE, not on the `--det` alias the user typed (glint_xtc.py
    defaults --det to 'jungfrau'), so an alias mismatch -- or a detector renamed between the run and
    the calibration deployment, leaving its geometry under the old name -- is only visible here.
    """
    try:
        s = str(detector.source)
        m = re.search(r"\(([^)]*)\)", s)
        v = (m.group(1) if m else s).strip()
        if v:
            return v, None
    except Exception:
        pass
    try:
        return str(detector.pyda.source).strip(), None
    except Exception as e:
        return None, f"det.source unavailable: {type(e).__name__}: {e}"


def _psana_geometry_path(detector, run):
    """What psana ITSELF says it loaded: GeometryAccess.path. -> (path|None, note|None).

    Metadata only -- det.geometry(run) takes a run NUMBER on psana1, so no event is needed.
    """
    try:
        ga = detector.geometry(int(run))
    except Exception as e:
        return None, f"det.geometry({run}) raised: {type(e).__name__}: {e}"
    if ga is None:
        return None, f"det.geometry({run}) returned None -- psana resolved NO geometry for this run"
    p = getattr(ga, "path", None)
    if not p:
        return None, "det.geometry() returned an object with no .path (psana too old to tell you)"
    return str(p), None


def _coords_from_calib_file(path):
    """Per-pixel (X, Y) in um from an arbitrary psana `*-end.data`, via PSCalib.

    Still metadata: this reads a few kB of text and evaluates the segment transforms. No event, no
    DAQ, no detector object -- which is what lets the manifest answer "and what would the OTHER
    candidate file have given you?" without a second DataSource.
    """
    from PSCalib.GeometryAccess import GeometryAccess
    ga = GeometryAccess(str(path), 0)
    X, Y, _Z = ga.get_pixel_coords()
    return np.asarray(X, float).ravel(), np.asarray(Y, float).ravel()


def _qlam(X, Y, z_um):
    """|q| * lambda for a flat detector at z.  |q| = |s - kin| / lam with s = r/|r| and
    kin = (0,0,sign(z)), so (|q|*lam)^2 = 2*(1 - |z|/|r|).

    lambda CANCELS in the ratio of two geometries, which is why this check needs no wavelength -- and
    on mfxx49820 that is not a nicety: ebeamPhotonEnergy() returns +-inf on 100 % of that run's
    events, so a check that needed a per-event wavelength would never run there at all.
    """
    r = np.sqrt(X * X + Y * Y + z_um * z_um)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.maximum(2.0 * (1.0 - abs(z_um) / r), 0.0))


def _qdelta(Xa, Ya, Xb, Yb, zdist_m, shape=None):
    """Per-pixel relative |q| difference of geometry A against reference B, with a per-segment signed
    breakdown. Both sides use the SAME z (= --zdist), so a wrong --zdist cannot manufacture a
    disagreement here -- and, equally, this check is blind to one. Said out loud in the report."""
    z_um = -abs(float(zdist_m)) * 1e6
    qa = _qlam(np.asarray(Xa, float).ravel(), np.asarray(Ya, float).ravel(), z_um)
    qb = _qlam(np.asarray(Xb, float).ravel(), np.asarray(Yb, float).ravel(), z_um)
    if qa.shape != qb.shape:
        raise ValueError(f"coord size mismatch: {qa.size} vs {qb.size} -- the two geometries do not "
                         f"describe the same pixel array")
    good = np.isfinite(qa) & np.isfinite(qb) & (qb > 0)
    if int(good.sum()) < 100:
        raise ValueError(f"only {int(good.sum())} pixels comparable -- nothing to measure")
    rel = np.full(qa.shape, np.nan)
    rel[good] = (qa[good] - qb[good]) / qb[good]
    a = np.abs(rel[good]) * 100.0
    # Outer half of the detector by the REFERENCE geometry's own |q| -- a data-derived split, not a
    # chosen radius. This is the population closest to the peak pixels the 3.16 % reference was
    # measured on: near the beam centre |q| -> 0 and a relative difference diverges by construction.
    qmed = float(np.median(qb[good]))
    outer = good & (qb >= qmed)
    ao = np.abs(rel[outer]) * 100.0
    out = {"n_px": int(good.sum()),
           "median_pct": float(np.median(a)),
           "p90_pct": float(np.percentile(a, 90)),
           "max_pct": float(a.max()),
           "mean_signed_pct": float(np.mean(rel[good]) * 100.0),
           "outer_median_pct": float(np.median(ao)),
           "outer_p90_pct": float(np.percentile(ao, 90)),
           "outer_max_pct": float(ao.max()),
           "outer_qlam_min": qmed}
    # A pure scale error (wrong distance, wrong absolute wavelength) is entirely absorbed by one
    # global ratio. The reference case absorbed almost nothing -- refitting a single ratio moved the
    # median 3.161 -> 3.138 %, i.e. 0.7 %. Reported, never thresholded.
    s = float(np.median(qa[good] / qb[good]))
    out["scale_fit"] = s
    out["scale_refit_median_pct"] = float(
        np.median(np.abs(qa[good] / s - qb[good]) / qb[good]) * 100.0)
    out["scale_absorbed_pct"] = (100.0 * (1.0 - out["scale_refit_median_pct"] / out["median_pct"])
                                 if out["median_pct"] > 0 else 0.0)
    if shape is not None and len(shape) == 3 and int(np.prod(shape)) == rel.size:
        nseg = int(shape[0])
        r2 = rel.reshape(nseg, -1)
        with np.errstate(invalid="ignore"):
            seg = [float(np.nanmedian(r2[i]) * 100.0) for i in range(nseg)]
        out["seg_median_pct"] = seg
        fin = [v for v in seg if np.isfinite(v)]
        out["seg_spread_pct"] = (max(fin) - min(fin)) if fin else float("nan")
        # THE VERDICT STATISTIC. Item 7's fault is organised by quadrant, so a pooled median is
        # diluted by every segment that happens to be correct -- 8 of 16 segments off by 3 % pools
        # to a median of 0.000 %. The worst segment is the same quantity computed on the population
        # the fault is actually organised by, and it degrades to the pooled number when the fault is
        # uniform (which is what the 3.16 % reference case was).
        out["seg_absmax_pct"] = max((abs(v) for v in fin), default=float("nan"))
        out["seg_absmax_idx"] = int(np.argmax([abs(v) if np.isfinite(v) else -1 for v in seg]))
    return out


# ---------------------------------------------------------------- the report object

class Manifest:
    def __init__(self, title):
        self.title = title
        self.body = []
        self.checks = []          # (status, key, message)
        self.data = {}
        self.downgraded = False   # refuse=False: refusals were recorded but will not raise

    def say(self, line=""):
        self.body.append(line)

    def check(self, status, key, message):
        self.checks.append((status, key, message))

    def demote(self, key, reason):
        """Turn a heuristic WARN into a NOTE because a MEASUREMENT superseded it."""
        for i, (s, k, m) in enumerate(self.checks):
            if k == key and s == WARN:
                self.checks[i] = (NOTE, k, f"{m}  [SUPERSEDED: {reason}]")

    def counts(self):
        c = {OK: 0, WARN: 0, UNKNOWN: 0, REFUSE: 0, NOTE: 0}
        for s, _, _ in self.checks:
            c[s] = c.get(s, 0) + 1
        return c

    @property
    def refusals(self):
        return [(k, m) for s, k, m in self.checks if s == REFUSE]

    @property
    def ok(self):
        """True only if every run-specific check ran AND passed. UNKNOWN counts against it, by
        design: a check that could not run is not a check that passed. NOTE does not count -- it is
        a property of the route, identical on a healthy run and a broken one."""
        c = self.counts()
        return c[REFUSE] == 0 and c[WARN] == 0 and c[UNKNOWN] == 0

    def text(self):
        c = self.counts()
        bar = "=" * 78
        out = [bar, f"GLINT GEOMETRY MANIFEST -- {self.title}", bar]
        out += self.body
        out.append("")
        out.append("  FINDINGS ABOUT THIS RUN")
        run_specific = [(s, k, m) for s, k, m in self.checks if s != NOTE]
        for s, k, m in run_specific or [(OK, "-", "none")]:
            out.append(f"    [{s:<7}] {k:<22} {m}")
        notes = [(s, k, m) for s, k, m in self.checks if s == NOTE]
        if notes:
            out.append("")
            out.append("  ROUTE-INVARIANT CAVEATS (true of EVERY run on this route; not a verdict)")
            for s, k, m in notes:
                out.append(f"    [{s:<7}] {k:<22} {m}")
        verdict = (REFUSE if c[REFUSE] else WARN if c[WARN] else UNKNOWN if c[UNKNOWN] else OK)
        tail = ("geometry CORROBORATED by a measurement against a trusted reference" if verdict == OK
                else ("REFUSING TO RUN" if not self.downgraded else
                      "PROVABLY BROKEN, but refuse=False -- the run CONTINUES anyway")
                if verdict == REFUSE else
                "geometry is NOT verified -- read the findings above")
        out.append("")
        out.append(f"  VERDICT: {verdict}  ({c[OK]} ok, {c[WARN]} warn, {c[UNKNOWN]} unknown, "
                   f"{c[REFUSE]} refuse, {c[NOTE]} note) -- {tail}")
        out.append(bar)
        return "\n".join(out)


# ---------------------------------------------------------------- the check itself

def check_geometry(detector=None, run=0, exp=None, det_name=None, calib_dir=None, geom=None,
                   env=None, zdist=None, shape=None, cross_check=True, refuse=True,
                   rank=0, nranks=1, stream=None):
    """Print -- loudly, every time -- which geometry this run is about to use, and whether anything
    corroborates it. Metadata only: no events are read.

    Cost: a handful of `stat` calls plus two small text reads (~5 ms). With `cross_check` and two
    geometries available, add two per-pixel coordinate arrays and a median -- 2.2 Mpx on Epix10ka2M,
    4.2 Mpx on Jungfrau4M, tens of milliseconds. Against a run whose per-event cost is ~150 ms
    (det.calib is 97.7 % of it) this is under one event. Cheap enough to run unconditionally, which
    is the point: a check you can switch off is a check that was off when it mattered.

    detector    psana1 Detector, already built. Optional -- without it the psana-side answers become
                UNKNOWN rather than being quietly skipped.
    run         run number. Used for run-range resolution only; no events are read.
    exp         experiment id, for the title and for guessing a calib dir when `env` is absent.
    det_name    the `--det` string the user typed, so an alias-vs-source mismatch is visible.
    calib_dir   the `--calib-dir` argument AS GIVEN (may be None, may be relative).
    geom        the `--geom` argument AS GIVEN (may be None, may be relative).
    env         ds.env(). STRONGLY recommended: env.calibDir() is the only authoritative answer to
                "which calib dir is psana actually using", including psana's own default.
    zdist       --zdist in m. Required for the |q| cross-check (both sides use it, so it cancels).
    shape       (nseg,H,W); taken from detector.shape(run) when omitted.
    cross_check run the coordinate-level |q| comparisons. False leaves the metadata half intact and
                says in the report that the measurement was disabled.
    refuse      raise on R1-R4. False downgrades them to WARN and SAYS SO in the report.
    rank/nranks under MPI, rank 0 prints the full manifest and every other rank prints one line
                carrying the resolved absolute paths and sha1s -- so a relative --geom that resolved
                differently on rank 7 shows up instead of hiding behind 64 identical reports.

    Returns the manifest dict. Raises GeometryManifestError on a refusal.
    """
    stream = stream if stream is not None else sys.stdout
    M = Manifest(f"{exp or '?'} run {run}, det {det_name or '?'}"
                 + (f"  [rank {rank}/{nranks}]" if nranks > 1 else ""))
    d = M.data

    # ---------------------------------------------------------------- 1. the .geom side
    if geom:
        gp = Path(geom)
        geom_abs = str(gp.resolve()) if gp.exists() else str(
            gp if gp.is_absolute() else Path(os.getcwd()) / gp)
        d["geom_arg"], d["geom_abs"] = geom, geom_abs
        if not gp.is_absolute():
            M.check(WARN, "geom-relative",
                    f"--geom {geom!r} is RELATIVE; resolved against cwd {os.getcwd()} -> {geom_abs}. "
                    f"Under MPI every rank calls the reader independently and under LUTE the process "
                    f"starts in the Slurm cwd, so the same argument can resolve differently per rank "
                    f"or per job -- or to nothing.")
        if not gp.exists():
            M.check(REFUSE, "geom-missing", f"R1: --geom {geom_abs} does not exist.")
        else:
            st = gp.stat()
            sha = _sha1(gp)
            d["geom_sha1"], d["geom_mtime"] = sha, st.st_mtime
            npan = None
            try:
                import geom_coords
                npan = len(geom_coords.parse_geom(str(gp)))
            except Exception as e:
                M.check(UNKNOWN, "geom-parse",
                        f"could not parse {geom_abs}: {type(e).__name__}: {e}")
            M.say(f"  X/Y IN FORCE : .geom  {geom_abs}")
            M.say(f"                 sha1 {sha}   mtime {_iso(st.st_mtime)}   {st.st_size} B"
                  + (f"   {npan} panels" if npan is not None else ""))
            if npan == 0:
                M.check(REFUSE, "geom-empty",
                        f"R1: --geom {geom_abs} parses to ZERO panels carrying min_fs+corner_x. "
                        f"coords_from_geom would raise on the first frame that has a wavelength, "
                        f"which on some runs is never.")
            elif npan:
                d["geom_panels"] = npan
                M.check(OK, "geom-source",
                        f"X/Y come from the .geom ({npan} panels); psana's X/Y are NOT used.")
            M.check(NOTE, "geom-parsers",
                    "this tree holds THREE .geom parsers with different inheritance and defaults, "
                    "and one file can yield different geometry depending on the path: "
                    "geom_coords.parse_geom (pass-1 q), glint.lute_bridge.parse_geom (pass-2 "
                    "predict/integrate), glint.geom.parse_geom (peaks/images routes). The panel "
                    "count above is geom_coords'; the other two are not cross-checked here.")
    else:
        M.say("  X/Y IN FORCE : psana deployed calibration   (no --geom)")
        M.check(WARN, "no-geom",
                "no --geom, so there is NO SECOND OPINION: nothing in this run can distinguish a "
                "refined geometry from an unrefined one, and this manifest can only report what was "
                "deployed, not whether it is right. This is exactly cxilu8823 r0226 -- --zdist from "
                "psana coords_z, no .geom, 887/933 frames indexed top-1 but consensus refused at 5 "
                "vectors = 0.2 % of 2659 pooled. With one geometry, 'was it the geometry?' is "
                "unanswerable. LUTE's `geom` field is Field(\"\") with no validator on the exp "
                "route, so a dropped value simply never appears on the command line.")

    # ---------------------------------------------------------------- 2. z, and what is discarded
    if zdist:
        M.say(f"  Z IN FORCE   : --zdist {float(zdist):.6f} m, stamped on EVERY pixel")
        M.check(NOTE, "z-flattened",
                f"per-pixel and per-panel Z is DISCARDED by both geometry paths -- geom_coords "
                f"stamps -zdist everywhere and deliberately ignores the .geom's clen/coffset, and "
                f"prep_geometry takes only sign(nanmean(Zf)) from psana. A physically staggered or "
                f"tilted detector is flattened to one plane at {float(zdist):.6f} m, which "
                f"manufactures a per-panel error of item 7's own shape even from a PERFECT .geom. "
                f"Nothing in this manifest can contradict a wrong --zdist either: it is a uniform "
                f"|q| rescale that blind indexing absorbs into the recovered cell.")
    else:
        M.check(UNKNOWN, "z-unknown",
                "no --zdist given to this check, so the |q| cross-check cannot run and the distance "
                "in force is unrecorded.")

    if geom:
        M.say("  MASK FROM    : psana calib dir (NOT the .geom) -- MIXED PROVENANCE")
        M.check(NOTE, "mask-provenance",
                "--geom overrides X/Y only; the bad-pixel mask still comes from whatever calib dir "
                "psana resolved (xtc_qreader_psana1.py:209-212, unconditional). Refined panel "
                "positions can therefore be paired with a mask from a mismatched or copied calib "
                "dir. The .geom's own bad regions / no_index panels are parsed away and never "
                "applied.")

    # ---------------------------------------------------------------- 3. which calib dir, really
    cdir = None
    if calib_dir:
        cp = Path(calib_dir)
        if not cp.is_absolute():
            M.check(WARN, "calibdir-relative",
                    f"--calib-dir {calib_dir!r} is RELATIVE; psana.setOption stores the string "
                    f"verbatim and every rank/job resolves it against its own cwd ({os.getcwd()}).")
        if not cp.exists():
            M.check(REFUSE, "calibdir-missing",
                    f"R2: --calib-dir {calib_dir!r} does not exist. psana.setOption accepts any "
                    f"string, so the override you asked for is a silent no-op and psana is using "
                    f"its default.")
        elif not _calib_groups(cp):
            M.check(REFUSE, "calibdir-empty",
                    f"R3: --calib-dir {cp} exists but holds NO `*::CalibV1` directory, so psana can "
                    f"read nothing from it. The override is provably a no-op.")
        cdir = str(cp.resolve()) if cp.exists() else str(cp)
        M.check(NOTE, "calibdir-sticky",
                "psana.setOption('psana.calib-dir') is PROCESS-GLOBAL and sticky, and cannot be "
                "unset. It is set here and again inside frames_for_events, so a process that builds "
                "a second DataSource -- pass 2 under --integrate, or any driver that reads twice -- "
                "carries this setting forward.")

    env_dir = None
    if env is not None:
        try:
            env_dir = str(env.calibDir())
        except Exception as e:
            M.check(UNKNOWN, "env-calibdir", f"env.calibDir() raised: {type(e).__name__}: {e}")
    if env_dir:
        d["calib_dir"] = env_dir
        M.say(f"  calib dir    : {env_dir}   [authoritative, from env.calibDir()]")
        if cdir and os.path.realpath(env_dir) != os.path.realpath(cdir):
            M.check(WARN, "calibdir-mismatch",
                    f"--calib-dir resolves to {cdir} but psana reports {env_dir}. setOption must "
                    f"precede DataSource; if it did not, the override was lost.")
        if not Path(env_dir).is_dir():
            M.check(WARN, "calibdir-nonexistent",
                    f"psana's calib dir {env_dir} is not a directory -- nothing can be read from it.")
        cdir = env_dir
    elif cdir:
        d["calib_dir"] = cdir
        M.say(f"  calib dir    : {cdir}   [from --calib-dir; env not supplied, NOT confirmed]")
        M.check(UNKNOWN, "calibdir-unconfirmed",
                "no env= passed, so psana's ACTUAL calib dir was never read back. Pass ds.env().")
    else:
        for guess in (f"/sdf/data/lcls/ds/{(exp or '')[:3]}/{exp}/calib",
                      f"/reg/d/psdm/{(exp or '')[:3]}/{exp}/calib"):
            if exp and Path(guess).is_dir():
                cdir = guess
                d["calib_dir"] = cdir
                M.say(f"  calib dir    : {cdir}   [GUESSED from the exp id -- not psana's answer]")
                M.check(WARN, "calibdir-guessed",
                        "the calib dir was guessed from the experiment id, not read from psana. "
                        "Pass env=ds.env() so this is authoritative.")
                break
        else:
            M.check(UNKNOWN, "calibdir-unknown",
                    "no --calib-dir, no env, and no default path exists on this host: the calib "
                    "tree could not be enumerated AT ALL. Every psana-side finding below is missing, "
                    "not passing.")

    # ---------------------------------------------------------------- 4. which source key
    src, src_err = (None, "no detector object passed")
    if detector is not None:
        src, src_err = _det_source(detector)
    if src:
        d["source"] = src
        M.say(f"  source key   : {src}"
              + (f"   (--det {det_name!r})" if det_name and det_name not in src else ""))
        if det_name and det_name.lower() not in src.lower():
            M.check(WARN, "det-alias",
                    f"--det {det_name!r} is an ALIAS: psana resolved it to source {src!r}. The "
                    f"calib tree is keyed on the SOURCE, so a detector renamed between the run and "
                    f"the calibration deployment leaves its geometry under the OLD name and the "
                    f"lookup for the new one misses.")
    else:
        M.check(UNKNOWN, "det-source",
                f"psana's resolved source string is unavailable: {src_err}. The calib enumeration "
                f"below cannot be matched to this detector.")

    # ---------------------------------------------------------------- 5. enumerate the ctype
    files, chosen = [], None
    if cdir and Path(cdir).is_dir():
        groups = _calib_groups(cdir)
        d["n_calib_groups"] = len(groups)
        matches = [g for g in groups if src and g[1] == src]
        if src and not matches:
            key = src.split(":")[-1].split(".")[0] if ":" in src else src
            loose = [g for g in groups if key and key.lower() in g[1].lower()]
            if loose:
                matches = loose
                M.check(WARN, "source-loose-match",
                        f"no calib group is keyed EXACTLY on {src!r}; matched loosely on {key!r} -> "
                        f"{[g[1] for g in loose]}. psana's own lookup is exact, so it may be finding "
                        f"nothing in this tree.")
        if not src:
            matches = groups
            if groups:
                M.check(UNKNOWN, "source-unmatched",
                        f"detector source unknown, so the {len(groups)} calib group(s) present "
                        f"could not be matched to this detector: "
                        f"{[f'{g[0]}::CalibV1/{g[1]}' for g in groups]}")
        if src and not matches:
            M.check(WARN, "source-absent",
                    f"calib dir {cdir} has NO group directory for source {src!r} (types present: "
                    f"{sorted({g[0] for g in groups})}). psana will find no geometry here.")
        if len(matches) > 1:
            M.check(WARN, "source-ambiguous",
                    f"{len(matches)} calib group directories carry this source: "
                    f"{[f'{g[0]}::CalibV1/{g[1]}' for g in matches]}. A hand-assembled or partially "
                    f"copied calib dir can leave the same source under two detector TYPES (e.g. "
                    f"Epix10ka::CalibV1 as well as Epix10ka2M::CalibV1). Only the first is "
                    f"enumerated below.")

        for gtype, gsrc, sdir in matches:
            cd, fl = _scan_ctype(sdir)
            M.say(f"  calib group  : {gtype}::CalibV1/{gsrc}")
            if cd is None:
                M.check(WARN, "ctype-absent",
                        f"{sdir} has no `{CTYPE}/` sub-directory at all -- this source carries other "
                        f"calibrations but no metrology.")
                continue
            files = fl
            chosen, cands, tied = _psana_pick(fl, int(run))
            M.say(f"  {CTYPE} dir : {cd}")
            if not fl:
                M.check(WARN, "ctype-empty",
                        f"{cd} holds no `<begin>-<end>.data` files -- the directory exists and is "
                        f"empty, which reads as 'geometry deployed' to anything that only checks "
                        f"for the directory.")
            for f in fl:
                mark = ("   <== psana resolves THIS" if f is chosen else
                        "   (candidate, NOT chosen)" if f in cands else
                        "   (run out of range)")
                M.say(f"      {f['name']:<18} runs {_rng(f):<12} {f['size']:>8} B  "
                      f"{_iso(f['mtime'])}  sha1 {_sha1(f['path'])}{mark}")
            d["calib_files"] = [{"name": f["name"], "begin": f["begin"],
                                 "end": (None if f["end"] == np.inf else int(f["end"])),
                                 "mtime": f["mtime"], "size": f["size"],
                                 "path": str(f["path"]), "sha1": _sha1(f["path"])} for f in fl]

            # ---- the flags this enumeration exists for ----
            if len(cands) > 1:
                M.check(WARN, "multiple-candidates",
                        f"{len(cands)} geometry files' run ranges contain run {run}: "
                        f"{[c['name'] for c in cands]}. psana takes the HIGHEST `begin`, i.e. "
                        f"{chosen['name']} -- which is NOT necessarily the one your reference was "
                        f"built on. mfxx49820 elsewhere deploys BOTH 0-end.data and a later "
                        f"~16-deg-tilted 8-end.data, and btx refined against 0-end. If a downstream "
                        f"refinement is your truth, pass --calib-dir (or --geom) rather than "
                        f"letting the run-range rule choose for you.")
            if len(tied) > 1:
                M.check(WARN, "ambiguous-pick",
                        f"{len(tied)} files share begin={tied[0]['begin']}: "
                        f"{[t['name'] for t in tied]}. psana's tie-break is undocumented; this "
                        f"replica took {chosen['name']}. Do not trust either answer -- name the "
                        f"file you want with --calib-dir.")
            if len(fl) == 1 and fl[0]["begin"] == 0 and fl[0]["end"] == np.inf:
                M.check(WARN, "never-redeployed",
                        f"the ONLY geometry file is {fl[0]['name']} -- a 0-to-end deployment that "
                        f"has never been superseded, i.e. the calibration this detector was "
                        f"installed with is still in force. This is the exact shape of STATUS.md "
                        f"item 7: on mfxx49820 that file is the UNREFINED 2021 starting calibration, "
                        f"and blind indexing on it locked a wrong doubled-c cell at support 23/6294 "
                        f"while reporting success. It is NOT proof of a fault -- a refined geometry "
                        f"can also be deployed as 0-end.data, and the psana2 leg of this repo's "
                        f"real-data gate is exactly that case -- but without a --geom you cannot "
                        f"tell which of the two you have.")
            if not cands:
                if geom:
                    M.check(WARN, "no-psana-geometry",
                            f"NO geometry file's run range contains run {run} (present: "
                            f"{[f['name'] for f in fl]}). --geom supplies X/Y so indexing survives, "
                            f"but det.coords_* would return None and the MASK lookup uses this same "
                            f"tree.")
                else:
                    M.check(REFUSE, "no-geometry",
                            f"R4: NO geometry file's run range contains run {run} (present: "
                            f"{[f['name'] for f in fl]}) and no --geom was given. det.coords_x/y/z "
                            f"return None and prep_geometry dies mid-run; this is the same failure, "
                            f"before the first det.calib instead of after thousands.")

            # ---- geometry against its own siblings: 'never refined', with no invented number ----
            sib = _sibling_ctypes(sdir)
            if sib:
                M.say("  ctype deployment dates in this source dir:")
                for name, n, mt in sib:
                    M.say(f"      {name:<18} {n:>3} file(s)   newest {_iso(mt)}")
                gm = max((f["mtime"] for f in fl), default=0.0)
                others = [mt for name, _n, mt in sib if name != CTYPE and mt > 0]
                if gm and others:
                    lag_days = (max(others) - gm) / 86400.0
                    d["geometry_lag_days"] = lag_days
                    if lag_days > 0:
                        M.check(WARN, "geometry-stale",
                                f"geometry is {lag_days:.0f} days OLDER than the newest sibling "
                                f"calibration in the same source dir. Everything else about this "
                                f"detector was redeployed after its metrology was; that is what a "
                                f"never-refined geometry looks like from the filesystem. Reported, "
                                f"not thresholded -- the number is the finding.")
                    else:
                        M.check(OK, "geometry-recent",
                                f"geometry is the newest deployment in this source dir, "
                                f"{-lag_days:.0f} days after the next-newest sibling ctype.")
            break
    elif cdir:
        M.check(UNKNOWN, "calibdir-unreadable",
                f"calib dir {cdir} is not a readable directory; nothing could be enumerated.")

    # ---------------------------------------------------------------- 6. psana's own answer
    psana_path, ppath_err = (None, "no detector object passed")
    if detector is not None:
        psana_path, ppath_err = _psana_geometry_path(detector, run)
    if psana_path:
        d["psana_geometry_path"] = psana_path
        d["psana_geometry_sha1"] = _sha1(psana_path)
        M.say(f"  psana loaded : {psana_path}")
        M.say(f"                 sha1 {d['psana_geometry_sha1']}")
        if chosen is not None:
            if os.path.realpath(psana_path) == os.path.realpath(str(chosen["path"])):
                M.check(OK, "resolution-agrees",
                        f"psana loaded {Path(psana_path).name}, exactly what the highest-`begin` "
                        f"rule predicts -- so the enumeration above is a faithful explanation of "
                        f"psana's choice, not a guess.")
            else:
                M.check(WARN, "resolution-disagrees",
                        f"psana loaded {psana_path} but the highest-`begin` rule predicts "
                        f"{chosen['path']}. One of the two is wrong about this tree; trust psana's "
                        f"path and treat the enumeration as incomplete (a symlink, a second calib "
                        f"dir in psana's search chain, or a file this scanner did not match).")
    else:
        M.check(UNKNOWN, "psana-path",
                f"psana would not say which geometry file it loaded: {ppath_err}. The enumeration "
                f"above is a REPLICA of psana's rule, not its answer.")
        if not files and not geom:
            M.check(REFUSE if refuse else WARN, "no-geometry-at-all",
                    "R4: no --geom, no enumerable calib geometry, and psana will not name a file. "
                    "This run has no geometry from any source.")

    # ---------------------------------------------------------------- 7. the measurement
    if cross_check and zdist:
        if shape is None and detector is not None:
            try:
                shape = tuple(int(v) for v in detector.shape(int(run)))
            except Exception as e:
                M.check(UNKNOWN, "shape",
                        f"detector.shape({run}) failed: {type(e).__name__}: {e}; the per-segment "
                        f"breakdown below is unavailable.")
        d["shape"] = tuple(shape) if shape else None

        Xp = Yp = None
        corroborated = False          # set by any cross-check that comes back OK
        if detector is not None:
            try:
                Xp, Yp = detector.coords_x(int(run)), detector.coords_y(int(run))
            except Exception as e:
                M.check(UNKNOWN, "psana-coords",
                        f"det.coords_x/y({run}) failed: {type(e).__name__}: {e}")
            if Xp is None or Yp is None:
                M.check(WARN if geom else UNKNOWN, "psana-coords-none",
                        f"det.coords_x/y({run}) returned None -- psana resolved no usable geometry "
                        f"for this run.")
                Xp = Yp = None

            # DEGENERATE Z. prep_geometry does `Zc = sign(nanmean(Zf)) * zdist` and
            # `kin = [0,0,sign(Zc)]`. A stub or placeholder calibration whose z is all zeros makes
            # that sign 0 -- kin collapses to [0,0,0], Zc to 0, and EVERY |q| degenerates to
            # 1/lambda. An all-NaN coords_z gives NaN and poisons the lot. Both are silent today,
            # and both are visible here for the price of one more coords call. This is a FACT about
            # the array, not a heuristic -- but it is not refused, because --geom is a legitimate
            # way to proceed past it (geom_coords stamps its own -zdist and never consults psana's
            # Z, so under --geom this is harmless; without it, it is fatal-but-silent).
            try:
                Zp = detector.coords_z(int(run))
            except Exception as e:
                Zp = None
                M.check(UNKNOWN, "psana-coords-z",
                        f"det.coords_z({run}) failed: {type(e).__name__}: {e}; the degenerate-Z "
                        f"check could not run.")
            if Zp is not None:
                zm = float(np.nanmean(np.asarray(Zp, float))) if np.size(Zp) else float("nan")
                sgn = np.sign(zm)
                d["psana_mean_z_um"] = zm
                if not np.isfinite(zm) or sgn == 0:
                    M.check(WARN, "degenerate-z",
                            f"psana's mean coords_z is {zm!r}, so prep_geometry's "
                            f"sign(nanmean(Zf)) is {sgn!r}: Zc becomes 0 (or NaN) and kin collapses "
                            f"to [0,0,0]. Every |q| would degenerate to 1/lambda -- a detector with "
                            f"no depth. "
                            + ("--geom is in force and geom_coords stamps its own -zdist, so this "
                               "run is not affected; the deployed calibration is still a stub."
                               if geom else
                               "This run IS affected: without --geom the sign comes straight from "
                               "this array."))
                else:
                    M.check(OK, "z-sign",
                            f"psana mean coords_z {zm:.1f} um -> sign {sgn:+.0f}; kin is well "
                            f"defined and --zdist {float(zdist):.6f} m keeps that sign.")

        # 7a. .geom vs psana -- item 7's own measurement, at startup instead of never. Both
        #     geometries are in memory at the same instant; today nothing compares them.
        if geom and Xp is not None and shape:
            try:
                import geom_coords
                Xg, Yg, _Zg = geom_coords.coords_from_geom(str(geom), tuple(shape), float(zdist))
                m = _qdelta(Xp, Yp, Xg, Yg, zdist, shape)
                d["qdelta_psana_vs_geom"] = m
                if _report_qdelta(M, m, "psana deployed", "the .geom", "qdelta-geom") == OK:
                    corroborated = True
            except Exception as e:
                M.check(UNKNOWN, "qdelta-geom",
                        f"could not compare psana's coords with the .geom: {type(e).__name__}: {e}. "
                        f"NOTE this failure is itself a finding: coords_from_geom raises when the "
                        f"panel map does not tile the calib array, i.e. when the .geom describes a "
                        f"different detector.")

        # 7b. psana's pick vs the OTHER candidate files. Needs no .geom at all, so it fires on the
        #     no-second-opinion case too -- and it turns 'more than one candidate' from a flag into
        #     a number: this is precisely how much choosing 8-end over 0-end would move |q|.
        if chosen is not None and shape:
            others = [f for f in files
                      if f is not chosen and f["begin"] <= int(run) <= f["end"]]
            Xa = Ya = None
            for other in others:
                try:
                    if Xa is None:                       # load psana's pick once, not per candidate
                        Xa, Ya = _coords_from_calib_file(chosen["path"])
                    Xb, Yb = _coords_from_calib_file(other["path"])
                    m = _qdelta(Xa, Ya, Xb, Yb, zdist, shape)
                    d.setdefault("qdelta_between_candidates", {})[other["name"]] = m
                    _report_qdelta(M, m, f"psana's pick {chosen['name']}", other["name"],
                                   f"qdelta-{other['name']}")
                except Exception as e:
                    M.check(UNKNOWN, f"qdelta-{other['name']}",
                            f"could not compare {chosen['name']} with {other['name']}: "
                            f"{type(e).__name__}: {e}")

        # A MEASUREMENT SUPERSEDES A MTIME. `never-redeployed` and `geometry-stale` are heuristics
        # about a file's age; if the deployed geometry has just been shown to agree with a trusted
        # .geom to within the level this repo measured for "the same metrology", its age stops being
        # evidence of anything. Demoted, not deleted -- still printed, under the caveats.
        if corroborated and geom:
            for k in ("never-redeployed", "geometry-stale"):
                M.demote(k, f"the deployed geometry was MEASURED to agree with {Path(geom).name} "
                            f"to within {MEDIAN_AGREE_PCT} %")
    elif cross_check:
        M.check(UNKNOWN, "qdelta",
                "the |q| cross-check did NOT run (no --zdist reached this check). Only metadata was "
                "checked; no geometry was measured.")
    else:
        M.check(UNKNOWN, "qdelta",
                "the |q| cross-check was DISABLED by the caller (cross_check=False). Only metadata "
                "was checked.")

    # ---------------------------------------------------------------- 8. emit
    ref = M.refusals
    if ref and not refuse:
        M.downgraded = True
        M.check(WARN, "refusal-downgraded",
                f"{len(ref)} refusal(s) downgraded to warnings by refuse=False: "
                f"{[k for k, _ in ref]}. The run continues on geometry inputs that are provably "
                f"broken, not merely suspicious.")
    d["ok"], d["counts"], d["text"] = M.ok, M.counts(), M.text()

    if rank == 0 or nranks == 1:
        print(M.text(), file=stream, flush=True)
    else:
        c = M.counts()
        print(f"[geom rank {rank}] geom={d.get('geom_abs')} sha1={d.get('geom_sha1')} "
              f"psana={d.get('psana_geometry_path')} psana_sha1={d.get('psana_geometry_sha1')} "
              f"calib={d.get('calib_dir')} ok={M.ok} warn={c[WARN]} unknown={c[UNKNOWN]} "
              f"refuse={c[REFUSE]}", file=stream, flush=True)

    if ref and refuse:
        raise GeometryManifestError(
            "geometry manifest REFUSED to start this run --\n  "
            + "\n  ".join(f"{k}: {m}" for k, m in ref)
            + "\nThese are facts, not heuristics. Fix the input, or pass refuse=False "
              "(--geom-check-warn-only) to downgrade them to warnings.")
    return d


def _report_qdelta(M, m, a_name, b_name, key):
    """Turn a _qdelta() result into manifest lines plus ONE status, using only the measured anchors.

    Returns the status it recorded, so the caller can let a measurement supersede a heuristic.
    """
    M.say(f"  |q| CROSS-CHECK   {a_name}   vs   {b_name}")
    M.say(f"      {m['n_px']} pixels pooled:  median |rel| {m['median_pct']:.3f} %   "
          f"p90 {m['p90_pct']:.3f} %   max {m['max_pct']:.3f} % (beam-centre-dominated: |q| -> 0 "
          f"there, so the pooled max is not a metrology number)")
    M.say(f"      outer half of |q|:  median {m['outer_median_pct']:.3f} %   "
          f"p90 {m['outer_p90_pct']:.3f} %   max {m['outer_max_pct']:.3f} %")
    M.say(f"      mean SIGNED rel {m['mean_signed_pct']:+.4f} %")
    M.say(f"      best global scale {m['scale_fit']:.6f} -> refit median "
          f"{m['scale_refit_median_pct']:.3f} % ({m['scale_absorbed_pct']:.1f} % of the "
          f"disagreement is a pure scale term that --zdist could absorb; the mfxx49820 reference "
          f"absorbed 0.7 %, 3.161 -> 3.138 %)")
    if "seg_median_pct" in m:
        M.say("      per-segment signed median (%):  "
              + "  ".join(f"{i}:{v:+.2f}" for i, v in enumerate(m["seg_median_pct"])))
        M.say(f"      worst segment {m['seg_absmax_idx']} at {m['seg_absmax_pct']:.3f} %;  "
              f"spread {m['seg_spread_pct']:.2f} pp.  A pure scale error gives spread ~0 with a "
              f"nonzero mean; item 7's fault is the opposite -- mean +0.003 % against a median of "
              f"3.16 %, quadrants +-3-5 %, segments 0-3 and 12-15 against 4-11.")

    # Verdict statistic: the worst SEGMENT when the segment axis is known, because a pooled median
    # is diluted by every quadrant that is correct. See the module docstring.
    if "seg_absmax_pct" in m and np.isfinite(m["seg_absmax_pct"]):
        v, brk, what = m["seg_absmax_pct"], SEG_BREAK_PCT, \
            f"worst-segment (seg {m['seg_absmax_idx']}) signed median"
        anchor = (f"{SEG_BREAK_PCT} %, the LOW END of the +-3-5 % per-quadrant range measured on "
                  f"mfxx49820 r0016")
    else:
        v, brk, what = m["outer_median_pct"], MEDIAN_BREAK_PCT, "outer-half pooled median"
        anchor = (f"{MEDIAN_BREAK_PCT} %, the pooled median measured on mfxx49820 r0016")
        M.say("      (no segment axis available -- the verdict falls back to the POOLED median, "
              "which is diluted whenever only some quadrants are wrong)")

    if v >= brk:
        st, msg = WARN, (
            f"{what} |q| disagreement {v:.3f} % is AT OR ABOVE {anchor} -- the geometry error at "
            f"which blind indexing locked the WRONG cell (support 23/6294) while reporting success. "
            f"These two geometries are not the same metrology and no --zdist reconciles them "
            f"({m['scale_absorbed_pct']:.1f} % of it is a global scale term). If one of them is a "
            f"trusted refinement, USE IT (--geom / --calib-dir); if neither is, do not quote the "
            f"cell this run reports.")
    elif v <= MEDIAN_AGREE_PCT:
        st, msg = OK, (
            f"{what} |q| disagreement {v:.3f} % is within {MEDIAN_AGREE_PCT} %, the agreement this "
            f"repo measured on a run whose deployed psana geometry IS the trusted refined fit "
            f"(max 0.025 %, median 0.011 %). Same metrology, corroborated by measurement.")
    else:
        st, msg = WARN, (
            f"{what} |q| disagreement {v:.3f} % falls BETWEEN the two measured anchors "
            f"({MEDIAN_AGREE_PCT} % = same metrology; {anchor} = known on this repo's reference run "
            f"to break blind indexing). Nothing measured here says which side of that gap this is "
            f"on, so NO VERDICT is offered -- the number is above, judge it. (The reference was "
            f"measured on btx's PEAK pixels, a different population from these detector pixels; "
            f"treat it as an order of magnitude, not an identity.)")
    M.check(st, key, msg)
    return st


def reuse_note():
    """Can gpu_pool.py's fast-radial-integration geometry monitor be reused for item 7?

    NO -- and reusing it would REPRODUCE item 7's silence rather than cure it. Three structural
    reasons, none of them fixable by tuning:

    1. It is DIFFERENTIAL; item 7's fault is CONSTANT. `_maybe_validate` seeds
       `self._reference_profile` from the FIRST injected reference frame and thereafter reports a
       Pearson correlation against that seed (gpu_pool.py:232-247). A geometry that was wrong from
       frame 1 seeds a wrong baseline and correlates ~1.0 with itself for the rest of the run --
       reported as stability='stable'. It can only ever see a geometry that CHANGES mid-run.
    2. It needs FRAMES, and this check must precede them. It takes an `image` and integrates it, so
       it cannot run at startup -- and on mfxx49820 it might never run at all, because 100 % of that
       run's events return +-inf from ebeamPhotonEnergy() and without --wavelength no frame ever
       reaches the geometry build.
    3. Its thresholds are invented and incommensurate. corr > 0.95 'stable' / < 0.90 'drift' is not
       derived from any measurement in this repo, and a Pearson correlation of a radial profile has
       no conversion to the 3.16 % |q| error that is the actual fault. Additionally `_drp_radial` is
       an optional import from two hard-coded paths, so on a host without drp-benchmarks the monitor
       is simply absent.

    What IS reused is its LESSON, not its code -- the fix already applied there: `_geom_init_error`
    is recorded and 'error' is excluded from `all_agree`, so a check that was configured and could
    not run does not read as healthy. That is exactly this module's UNKNOWN status, which prints as
    loudly as a WARN and counts against `ok`.

    Where that monitor IS the right tool and this one is not: a detector that MOVES mid-run, or
    per-event coords that change. This manifest is a startup snapshot and is blind to both. They are
    complements, not substitutes -- run both.
    """
    return reuse_note.__doc__


# ---------------------------------------------------------------- standalone triage CLI

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="GLINT startup geometry manifest (metadata only; no events are read)")
    ap.add_argument("--exp", default=os.environ.get("GLINT_EXP"))
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--det", default="jungfrau")
    ap.add_argument("--zdist", type=float, default=0.0)
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--geom", default=None)
    ap.add_argument("--no-cross-check", action="store_true")
    ap.add_argument("--no-refuse", action="store_true")
    a = ap.parse_args(argv)

    detector = env = None
    try:
        import psana
        if a.calib_dir:
            psana.setOption("psana.calib-dir", str(a.calib_dir))
        ds = psana.DataSource(f"exp={a.exp}:run={a.run}:smd")
        env = ds.env()
        detector = psana.Detector(a.det)
    except Exception as e:
        print(f"[geom_manifest] psana unavailable or DataSource failed ({type(e).__name__}: {e}); "
              f"running the FILESYSTEM half only -- psana-side answers will read UNKNOWN.",
              file=sys.stderr, flush=True)
    try:
        check_geometry(detector=detector, run=a.run, exp=a.exp, det_name=a.det,
                       calib_dir=a.calib_dir, geom=a.geom, env=env, zdist=a.zdist,
                       cross_check=not a.no_cross_check, refuse=not a.no_refuse)
    except GeometryManifestError as e:
        print(str(e), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
