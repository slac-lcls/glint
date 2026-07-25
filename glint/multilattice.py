"""Multi-lattice / double-hit primitives.

A "double hit" is two (or more) crystals in one shot -- common when the sample is concentrated
(several crystals per drop). GLINT resolves them by DEFLATE-AND-REINDEX: index the first lattice,
remove the peaks it explains, and re-index the residual to find the second. `deflate_peaks` is the
pure (removal) half; the second-lattice search is the GPU indexer, driven from the streaming driver's
double-hit path (StreamDriver(double_hit=True)).

A rising double-hit RATE is a useful live beamline signal (sample too concentrated / jet issues), so
the driver exposes it in stats() as `double_hit_rate`.
"""
import numpy as np


def deflate_peaks(q, M, tol=0.15):
    """Reciprocal vectors of `q` NOT explained by cell `M` -- the residual for a second-lattice search.

    M has reciprocal-basis rows, so hkl = q @ M; a peak is 'assigned' to M when every hkl component is
    within `tol` of an integer. Returns the unassigned peaks (a copy). Pure numpy.
    """
    q = np.asarray(q, float)
    if q.ndim != 2 or len(q) == 0:
        return q.reshape(0, 3)
    hf = q @ np.asarray(M, float)
    assigned = np.abs(hf - np.round(hf)).max(1) < tol
    return q[~assigned]
