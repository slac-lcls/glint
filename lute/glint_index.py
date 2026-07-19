"""LUTE task-parameters model for the GLINT GPU indexer -- a drop-in alternative to CrystFELIndexer
in the SFX DAG:  PeakFinderSFX -> [IndexGLINT] -> StreamFileConcatenator -> PartialatorMerger.
(PeakFinderSFX is optional: set `images` and GLINT peak-finds the raw .cxi on the GPU itself.)

MERGEABILITY: the default GLINT stream is ORIENTATION-ONLY -- every reflection carries placeholder
I=0/sigma=0. To merge you must pick one of `integrate: true` (GLINT predicts and box-integrates its
own reflections, writing real I/sigma) or the `fromfile` -> indexamajig handoff. Feeding the default
stream straight to partialator merges zeros.

Why GLINT in LUTE: none of LUTE's bundled CrystFEL builds (0.10.2 default ... 0.12.0) are compiled
with FFBIDX, so GPU fast-feedback-style indexing is simply unavailable via indexamajig. GLINT fills
that gap -- GPU blind indexing + cross-frame consensus -- and emits the same CrystFEL `.stream` that
ConcatenateStreamFiles / partialator already consume. For the best MERGE, set `fromfile` (GLINT hands
CrystFEL the refined-merge solution file).

INSTALL (see lute/README.md): copy to `lute/io/models/glint_index.py`, add
`from .glint_index import *` to `lute/io/models/__init__.py`, and add
`GLINTIndexer: Executor = Executor("IndexGLINT")` to `lute/managed_tasks.py`.
"""
from typing import Any, Dict, Literal, Optional

from pydantic import Field, PositiveFloat, PositiveInt, validator

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
    geom: str = Field(
        "", description="CrystFEL .geom file.", flag_type="--", rename_param="geom",
    )
    peakfinder: Optional[Literal["v4", "pf9", "stored"]] = Field(
        None,
        description="Peak finder for `images`: v4 | pf9 | stored. `stored` reuses the peakfinder8 / "
                    "Cheetah peaks already written into the .cxi -- the efficient path when the peaks "
                    "exist. (The CLI also accepts pf8, but GLINT has not vendored it: it needs a "
                    "per-pixel q map + radial.py and exits at startup. Use `stored` instead.) "
                    "Unset = GLINT default (v4).",
        flag_type="--", rename_param="peakfinder",
    )
    top_peaks: Optional[PositiveInt] = Field(
        None,
        description="ONLY with `images`: keep the N strongest peaks per frame (~100 is the measured "
                    "sweet spot; weak peaks HURT the indexing rate). The `peaks` path ignores it -- "
                    "truncate in FindPeaksSFX instead. Unset = keep all.",
        flag_type="--", rename_param="top-peaks",
    )
    wavelength: Optional[PositiveFloat] = Field(
        None, description="Wavelength in A. Unset = read from the .geom / per-event data.",
        flag_type="--", rename_param="wavelength",
    )
    out: str = Field(
        "", description="Output .stream. Orientation-only (placeholder I/sigma) unless `integrate` is "
                        "set or the `fromfile` handoff is used -- see the module docstring.",
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
    fromfile: Optional[str] = Field(
        None,
        description="Also emit a CrystFEL --indexing=file solution file (the refined-MERGE handoff): "
                    "run `indexamajig --indexing=file --fromfile-input-file=<f> --tolerance=10,10,10,3`.",
        flag_type="--", rename_param="fromfile",
    )
    lattice: str = Field(
        "aP", description="Bravais lattice code for --fromfile (e.g. tPc tetragonal).",
        flag_type="--", rename_param="lattice",
    )
    cascade: Optional[str] = Field(
        None, description="Optional external cell-given indexer binary (ffbidx driver) as a fallback.",
        flag_type="--", rename_param="cascade",
    )

    # ---- integration: emit REAL I/sigma so the stream goes straight to partialator ----------------
    # Without these the stream carries placeholder intensities and only the `fromfile` -> CrystFEL
    # handoff yields a mergeable dataset. With `integrate` GLINT predicts and box-integrates its own
    # reflections (GPU-fused; the whole-frame float64 upcast that used to dominate is gone), so the DAG
    # can skip indexamajig entirely. TRADE-OFF: CrystFEL's prediction refinement imposes the lattice
    # symmetry and still merges better -- prefer `fromfile` when merge quality is what matters, and
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

    @validator("images", always=True)
    def _one_source(cls, images: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Exactly one frame source: the CrystFEL peak stream, or raw images GLINT finds peaks in."""
        peaks: str = values.get("peaks") or ""
        if images and peaks:
            raise ValueError("set `peaks` OR `images`, not both (they are alternative frame sources)")
        if not images and not peaks:
            raise ValueError("one of `peaks` (from PeakFinderSFX) or `images` (raw .cxi) is required")
        return images

    @validator("top_peaks", always=True)
    def _top_peaks_images_only(cls, top_peaks: Optional[int], values: Dict[str, Any]) -> Optional[int]:
        """--top-peaks is only read on the `images` path; silently ignored with `peaks`."""
        if top_peaks and (values.get("peaks") or ""):
            raise ValueError("`top_peaks` applies only to `images`; on the `peaks` path it is ignored "
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
