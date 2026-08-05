"""LUTE task-parameters model for the GLINT GPU indexer -- a drop-in alternative to CrystFELIndexer
in the SFX DAG:  PeakFinderSFX -> [IndexGLINT] -> StreamFileConcatenator -> PartialatorMerger.
(PeakFinderSFX is optional: set `images` and GLINT peak-finds the raw .cxi on the GPU itself.)

MERGEABILITY: the default GLINT stream is ORIENTATION-ONLY -- every reflection carries placeholder
I=0/sigma=0. To merge you must pick one of `integrate: true` (GLINT predicts and box-integrates its
own reflections, writing real I/sigma) or the `tofile` -> indexamajig handoff. Feeding the default
stream straight to partialator merges zeros.

Why GLINT in LUTE: none of LUTE's bundled CrystFEL builds (0.10.2 default ... 0.12.0) are compiled
with FFBIDX, so GPU fast-feedback-style indexing is simply unavailable via indexamajig. GLINT fills
that gap -- GPU blind indexing + cross-frame consensus -- and emits the same CrystFEL `.stream` that
ConcatenateStreamFiles / partialator already consume. For the best MERGE, set `tofile` (GLINT hands
CrystFEL the refined-merge solution file).

INSTALL (see lute/README.md): copy to `lute/io/models/glint_index.py`, add
`from .glint_index import *` to `lute/io/models/__init__.py`, and add
`GLINTIndexer: Executor = Executor("IndexGLINT")` to `lute/managed_tasks.py`.
"""
from typing import Any, Dict, Literal, Optional

from pydantic import Field, PositiveFloat, PositiveInt, root_validator, validator

from lute.io.models.base import ThirdPartyParameters

__all__ = ["IndexGLINTParameters"]
__author__ = "GLINT (S. Marchesini)"


class IndexGLINTParameters(ThirdPartyParameters):
    """Parameters for the GLINT GPU blind SFX indexer (peaks + geometry -> CrystFEL .stream)."""

    class Config(ThirdPartyParameters.Config):
        set_result: bool = True
        result_from_params: str = ""

    executable: str = Field(
        "/sdf/home/s/smarches/git/glint/lute/glint_launch.sh",
        description="Launcher that activates the GLINT GPU (torch) env and runs glint.glint_cli.",
        flag_type="",
    )
    peaks: str = Field(
        "", description="CrystFEL peak-search stream from FindPeaksSFX (peakfinder8).",
        flag_type="--", rename_param="peaks",
    )
    images: Optional[str] = Field(
        None,
        description="ALTERNATIVE to `peaks`: raw detector .cxi (or a .list of them). GLINT peak-finds "
                    "on the GPU itself, so PeakFinderSFX can be dropped from the DAG. Mutually "
                    "exclusive with `peaks`.",
        flag_type="--", rename_param="images",
    )
    # ---- THIRD frame source: raw xtc, read in-process (psana1) or over envbridge (psana2) --------
    # PeakFinderSFX stays in the DAG and remains the default route; this is for runs where GLINT
    # should read the xtc itself and no .cxi is wanted. It routes glint_launch.sh to
    # experiments/xtc_bridge/glint_xtc.py instead of glint.glint_cli -- a different program with a
    # different flag set, which is why the launcher whitelists rather than forwarding blindly.
    #
    # NOTE ON PEAK-FINDING, because this source changes who does it. With `peaks` the peaks come from
    # peakfinder8 upstream; with `images` you can set `peakfinder: stored` and REUSE the .cxi's own
    # peakfinder8/Cheetah peaks. Raw xtc carries no stored peak list, so GLINT must find its own
    # (PeakFinderV4). This is the ONE route where GLINT's finder stands in for peakfinder8 rather
    # than deferring to it -- so validate a new detector here before trusting a rate.
    exp: Optional[str] = Field(
        None,
        description="ALTERNATIVE to `peaks`/`images`: LCLS experiment id, read straight from xtc. "
                    "Requires `run` and `zdist`. Mutually exclusive with the other two sources.",
        flag_type="--", rename_param="exp",
    )
    run: Optional[PositiveInt] = Field(
        None, description="Run number; only with `exp`.", flag_type="--", rename_param="run",
    )
    det: Optional[str] = Field(
        None,
        description="psana detector name for the `exp` source, e.g. 'MfxEndstation.0:Epix10ka2M.0' "
                    "or a Jungfrau alias. Get the exact string from psana's DetNames().",
        flag_type="--", rename_param="det",
    )
    zdist: Optional[PositiveFloat] = Field(
        None,
        description="Sample-detector distance in METRES; REQUIRED with `exp`. psana's per-pixel Z is "
                    "nominal, so GLINT replaces it. A wrong value scales every |q|: source it from "
                    "the refined geometry (a .geom's clen+coffset, or a .poni Distance).",
        flag_type="--", rename_param="zdist",
    )
    psana: Optional[Literal["1", "2"]] = Field(
        None,
        description="Only with `exp`. '1' = LCLS-I xtc1, read IN-PROCESS (psana1 coexists with torch "
                    "in the GLINT env). '2' = LCLS-II xtc2, read in conda2 over envbridge, which must "
                    "be installed. The CLI defaults to 1.",
        flag_type="--", rename_param="psana",
    )
    calib_dir: Optional[str] = Field(
        None,
        description="Only with `exp`, psana1: override psana's calib dir. Needed when psana would "
                    "resolve a LATER deployed geometry than the one your refinement was built on -- "
                    "--zdist replaces only Z, so X/Y still come from whatever psana picks.",
        flag_type="--", rename_param="calib-dir",
    )
    geom: str = Field(
        "",
        description="CrystFEL .geom file. With `peaks`/`images` this is the detector geometry. With "
                    "`exp` it is optional to the CLI but USUALLY REQUIRED IN PRACTICE: psana's "
                    "deployed calibration is often the UNREFINED starting geometry, while the "
                    "refinement downstream trusts (btx / BayFAI / CrystFEL) exists only as a .geom "
                    "and never round-trips back into psana. Measured on mfxx49820 r0016 that gap is "
                    "a median 3.16% error in |q|, signed per detector quadrant, which no --zdist can "
                    "absorb -- and it was enough to make blind indexing return a wrong doubled-c cell.",
        flag_type="--", rename_param="geom",
    )
    peakfinder: Literal["v4", "pf9", "stored"] = Field(
        "stored",
        description="Peak finder for `images`: v4 | pf9 | stored. Defaults to `stored`, which reuses "
                    "the peakfinder8 / Cheetah peaks already written into the .cxi -- no re-finding, "
                    "and it avoids v4 over-finding on water rings. This DELIBERATELY overrides the "
                    "GLINT CLI default of v4: on a .cxi that already carries peaks, re-finding them "
                    "is both slower and worse. Set `v4`/`pf9` explicitly to peak-find from scratch. "
                    "(The CLI also accepts pf8, but GLINT has not vendored it: it needs a per-pixel "
                    "q map + radial.py and exits at startup. Use `stored` instead.)",
        flag_type="--", rename_param="peakfinder",
    )
    top_peaks: Optional[PositiveInt] = Field(
        None,
        description="ONLY with `images`: keep the N strongest peaks per frame. USE WITH CARE -- it "
                    "truncates the frame itself, so the smaller list feeds scoring and refinement "
                    "too, not just the candidate search. Measured on 120 real cxidb frames (median "
                    "100 peaks): top_peaks 200 costs ~2 points of correct-lattice, 100 costs ~11, "
                    "and 50 collapses the rate. Leave UNSET unless a finder is over-finding on "
                    "background; it is a guard against that, not a free speedup. REJECTED at config "
                    "time if set alongside `peaks`, because the CLI would not read it -- truncate "
                    "in FindPeaksSFX instead. Unset = keep all.",
        flag_type="--", rename_param="top-peaks",
    )
    wavelength: Optional[PositiveFloat] = Field(
        None, description="Wavelength in A. Unset = read from the .geom / per-event data.",
        flag_type="--", rename_param="wavelength",
    )
    n: Optional[int] = Field(
        None,
        description="Limit to the first N frames (0/None = all) -- the CLI's -N. Mostly for smoke "
                    "tests on a slice of a run before committing a full DAG.",
        flag_type="-", rename_param="N",
    )
    out: str = Field(
        "", description="Output .stream. Orientation-only (placeholder I/sigma) unless `integrate` is "
                        "set or the `tofile` handoff is used -- see the module docstring.",
        flag_type="--", rename_param="out", is_result=True,
    )
    cell: Optional[str] = Field(
        None, description='Known unit cell "a b c al be ga" (else fully-blind cross-frame consensus).',
        flag_type="--", rename_param="cell",
    )
    mode: str = Field(
        "auto", description="Front end: auto | sparse (SFX stills) | dense (rotation clouds).",
        flag_type="--", rename_param="mode",
    )
    nbest: PositiveInt = Field(
        3, description="N-best consensus hypotheses kept per frame.", flag_type="--", rename_param="nbest",
    )
    min_peaks: PositiveInt = Field(
        6, description="Skip frames with fewer peaks.", flag_type="--", rename_param="min-peaks",
    )
    device: str = Field(
        "auto", description="auto (GPU if present) | cpu.", flag_type="--", rename_param="device",
    )
    tofile: Optional[str] = Field(
        None,
        description="WRITE a CrystFEL --indexing=file solution file (the refined-MERGE handoff): "
                    "run `indexamajig --indexing=file --fromfile-input-file=<f> --tolerance=10,10,10,3`. "
                    "Was `fromfile`, which is still accepted -- see the validator below.",
        flag_type="--", rename_param="tofile",
    )
    lattice: str = Field(
        "aP", description="Bravais lattice code for --tofile (e.g. tPc tetragonal).",
        flag_type="--", rename_param="lattice",
    )
    cascade: Optional[str] = Field(
        None, description="Optional external cell-given indexer binary (ffbidx driver) as a fallback.",
        flag_type="--", rename_param="cascade",
    )

    # ---- integration: emit REAL I/sigma so the stream goes straight to partialator ----------------
    # Without these the stream carries placeholder intensities and only the `tofile` -> CrystFEL
    # handoff yields a mergeable dataset. With `integrate` GLINT predicts and box-integrates its own
    # reflections (GPU-fused; the whole-frame float64 upcast that used to dominate is gone), so the DAG
    # can skip indexamajig entirely. TRADE-OFF: CrystFEL's prediction refinement imposes the lattice
    # symmetry and still merges better -- prefer `tofile` when merge quality is what matters, and
    # `integrate` when a CrystFEL-free GPU pipeline is what matters.
    integrate: bool = Field(
        False,
        description="Predict + integrate GLINT's own reflections and write real I/sigma into the "
                    "stream (needs image data: --peaks + `image_dir`, or --images).",
        flag_type="--", rename_param="integrate",
    )
    image_dir: Optional[str] = Field(
        None,
        description="Base directory holding the per-frame images referenced by the peak stream. "
                    "Required for `integrate` when the source is `peaks`; ignored with `images`.",
        flag_type="--", rename_param="image-dir",
    )
    int_dmin: Optional[PositiveFloat] = Field(
        None, description="Resolution limit in A for prediction/integration. Unset = GLINT default (2.0).",
        flag_type="--", rename_param="int-dmin",
    )
    int_tol: PositiveFloat = Field(
        0.002,
        description="Excitation-error tolerance for predicting reflections. Deliberately OVERRIDES the "
                    "GLINT CLI default of 0.006, which measurably over-predicts: on real data 0.006 "
                    "gave 2017 reflections of which only 449 sat on a real peak (78% background) and "
                    "CC1/2 fell to 0.04, while 0.002 gave 641 reflections at <I/sigma> 30 and CC1/2 0.28.",
        flag_type="--", rename_param="int-tol",
    )

    # Validators run in field-definition order and see only EARLIER fields in `values`, so each of
    # these is declared after everything it inspects. They exist because the corresponding failures
    # are otherwise silent: a run that "succeeds" and hands the next task nothing usable.

    @root_validator(pre=True)
    def _accept_legacy_fromfile(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """`fromfile` was renamed to `tofile`: GLINT WRITES that file, and the old name came from
        CrystFEL's reader flag (--fromfile-input-file), so it read backwards from this side.

        Runs pre=True so an existing config keeps working -- silently dropping an unknown `fromfile`
        key would turn "best merge" configs into placeholder-intensity streams with no error, which
        is exactly the failure mode nobody notices until partialator produces nothing."""
        if "fromfile" in values:
            legacy = values.pop("fromfile")
            if values.get("tofile") not in (None, "") and legacy not in (None, ""):
                raise ValueError("set `tofile` OR the deprecated `fromfile`, not both")
            if legacy not in (None, ""):
                values["tofile"] = legacy
        return values

    @validator("exp", always=True)
    def _one_source(cls, exp: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """EXACTLY ONE frame source: a CrystFEL peak stream, raw .cxi images, or raw xtc.

        Hung on `exp` rather than `images` because pydantic runs field validators in DECLARATION
        order, and `exp` is declared last of the three -- so this is the first point at which all
        three values are visible in `values`."""
        peaks: str = values.get("peaks") or ""
        images: str = values.get("images") or ""
        chosen = [n for n, v in (("peaks", peaks), ("images", images), ("exp", exp)) if v]
        if len(chosen) > 1:
            raise ValueError(f"set exactly ONE frame source, got {chosen}: `peaks` (from "
                             "PeakFinderSFX), `images` (raw .cxi) and `exp` (raw xtc) are "
                             "alternatives, not layers")
        if not chosen:
            raise ValueError("one frame source is required: `peaks` (from PeakFinderSFX), `images` "
                             "(raw .cxi), or `exp` (+`run`, raw xtc)")
        return exp

    @validator("run", "zdist", always=True)
    def _xtc_requires(cls, v: Any, values: Dict[str, Any], field: Any) -> Any:
        """`run` and `zdist` are mandatory with `exp`.

        zdist especially: glint_xtc REQUIRES it, because psana's per-pixel Z is nominal. Omitting it
        would fail inside a Slurm job rather than here at config time."""
        if values.get("exp") and v in (None, ""):
            raise ValueError(f"`{field.name}` is required with `exp` (the raw xtc source)")
        return v

    @validator("det", "psana", "calib_dir", always=True)
    def _xtc_only(cls, v: Any, values: Dict[str, Any], field: Any) -> Any:
        """Reject the xtc-only knobs on the other two sources instead of letting them be dropped.

        glint_launch.sh whitelists flags per destination, so one of these set alongside `peaks` would
        be silently discarded -- and the run would look as though it had honoured a setting the
        indexer never saw. NOTE `wavelength` is deliberately NOT in this list: it is meaningful on
        all three sources."""
        if v not in (None, "") and not values.get("exp"):
            raise ValueError(f"`{field.name}` applies only to the `exp` (raw xtc) source")
        return v

    @validator("top_peaks", always=True)
    def _top_peaks_images_only(cls, top_peaks: Optional[int], values: Dict[str, Any]) -> Optional[int]:
        """--top-peaks is only read on the `images` path. Reject it with `peaks` rather than accept
        it: the CLI would ignore the flag, so a silent pass would let a run look like it truncated
        the peak list when it did not."""
        if top_peaks and (values.get("peaks") or ""):
            raise ValueError("`top_peaks` applies only to `images`; the `peaks` path would ignore it "
                             "-- truncate the peak list in FindPeaksSFX instead")
        return top_peaks

    @validator("out", always=True)
    def _out_required(cls, out: str) -> str:
        """Empty `out` is skipped by the LUTE renderer, so GLINT would write glint.stream into the
        Slurm cwd while the Task result records "" -- the concatenator then gets an empty path from a
        run that reported success. Fail at config time instead."""
        if not out:
            raise ValueError("`out` is required: it is the Task result StreamFileConcatenator consumes")
        return out

    @validator("image_dir", always=True)
    def _image_dir_for_integrate(cls, image_dir: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """With `peaks`, integration needs somewhere to find the images. Without it the CLI defaults to
        "." , resolves every frame against the Slurm cwd, finds none, and emits a stream with zero
        integrated reflections rather than an error."""
        if values.get("integrate") and (values.get("peaks") or "") and not image_dir:
            raise ValueError("`integrate` with `peaks` requires `image_dir` (base directory of the "
                             "per-frame images)")
        return image_dir
