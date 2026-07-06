"""LUTE task-parameters model for the GLINT GPU indexer -- a drop-in alternative to CrystFELIndexer
in the SFX DAG:  PeakFinderSFX -> [IndexGLINT] -> ConcatenateStreamFiles -> MergePartialator.

Why GLINT in LUTE: none of LUTE's bundled CrystFEL builds (0.10.2 default ... 0.12.0) are compiled
with FFBIDX, so GPU fast-feedback-style indexing is simply unavailable via indexamajig. GLINT fills
that gap -- GPU blind indexing + cross-frame consensus -- and emits the same CrystFEL `.stream` that
ConcatenateStreamFiles / partialator already consume. For the best MERGE, set `fromfile` (GLINT hands
CrystFEL the refined-merge solution file).

INSTALL (see lute/README.md): copy to `lute/io/models/glint_index.py`, add
`from .glint_index import *` to `lute/io/models/__init__.py`, and add
`GLINTIndexer: Executor = Executor("IndexGLINT")` to `lute/managed_tasks.py`.
"""
from typing import Optional

from pydantic import Field, PositiveInt

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
    geom: str = Field(
        "", description="CrystFEL .geom file.", flag_type="--", rename_param="geom",
    )
    out: str = Field(
        "", description="Output .stream (mergeable).", flag_type="--", rename_param="out", is_result=True,
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
