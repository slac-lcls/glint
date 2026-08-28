"""A synthetic stand-in for xtc_qreader.run_to_qframes, for the psana-free end-to-end test.

Returns the SAME dict shape (qframes/events/counts) as the real reader, but the q come from a planted
lysozyme cell via the shipped, tested ewald_spots generator instead of from an xtc file. envbridge
calls this in the same env (no conda/psana needed), so end_to_end_test exercises the real envbridge
call + the real driver indexing path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # experiments/, for the generator
from test_geom_bridge import ewald_spots


def run_to_qframes_sim(nframes: int = 30):
    qframes = [np.ascontiguousarray(np.asarray(ewald_spots(i), float)) for i in range(nframes)]
    return {"qframes": qframes, "events": list(range(nframes)),
            "n_events": nframes, "n_sent": nframes, "n_skipped_wl": 0}
