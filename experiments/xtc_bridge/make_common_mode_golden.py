"""Regenerate common_mode_golden.npz from REAL psana. Run where psana is importable.

The case grid, inputs and psana reference driver come from common_mode_cases.py — the same module
the test imports — so the generator cannot drift from what it generates for; the test's
psana-present staleness check re-verifies that whenever psana is available. Refuses to write a
golden set in which any common-mode variant never fires (the defect the 64-column goldens had).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_mode_cases import (GOLDEN_NCOL, GOLDEN_NSEG, assert_modes_fire, make_inputs,
                               psana_reference)

out = {}
for key, arrf, gmask, mode, cormax in make_inputs(GOLDEN_NSEG, GOLDEN_NCOL):
    out[key] = (psana_reference(arrf, gmask, mode, cormax, 10) - arrf).astype(np.float32)

if assert_modes_fire(out):
    print("REFUSING to write goldens: at least one mode never fires at this geometry.")
    sys.exit(1)

# beside this script, so regeneration updates the file the test actually loads no matter
# which directory it is invoked from
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "common_mode_golden.npz")
np.savez_compressed(OUT, psana_release="ana-4.0.59-py3-minipytorch", **out)
print("wrote", OUT, "cases:", len(out), " bytes:", os.path.getsize(OUT))
