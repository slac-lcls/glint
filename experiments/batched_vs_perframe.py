"""Is the batched known-cell path really rate-identical to the per-frame one?

tab:summary's footnote states: "batched = the known-cell engine amortized over B=120, bit-exact and
rate-identical to the unbatched path". The arsenal run contradicts that -- the batched pass rejected
42 frames and a per-frame call recovered 9 of them. This checks the claim directly on all frames:
same cell, same gate, per-frame outcome compared.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT); sys.path.insert(0, WT + "/experiments")
import glint.glint_fast as gf
from glint.glint_fast import index_blind_nbest, matched
from glint.multishot import same_lattice
from glint.replica_gpu import index_known_gpu_cell
from what_are_the_failures import strict_gate, LYSO
import glint.replica_gpu_batch as rgb

frames = [np.asarray(f, float) for f in gf.load("frames_cxidb_clean.txt") if len(f) >= 6]
N = len(frames)
for label, Mc in (("voted cell", None), ("exact LYSO", LYSO)):
    if Mc is None:
        cells = []
        for i in range(5):
            for c, _s in index_blind_nbest(frames[i], 3):
                if c is not None:
                    cells.append(np.asarray(c, float))
                break
        Mc = max(cells, key=lambda a: sum(1 for b in cells if same_lattice(a, b)))
    Mb = rgb.index_fused(frames, Mc, B=N)
    agree = dis = bonly = ponly = 0
    dm = []
    for i, f in enumerate(frames):
        gb = strict_gate(None if Mb[i] is None else np.asarray(Mb[i], float), f, Mc)
        Mp = index_known_gpu_cell(f, Mc)
        gp = strict_gate(None if Mp is None else np.asarray(Mp, float), f, Mc)
        agree += int(gb == gp); dis += int(gb != gp)
        bonly += int(gb and not gp); ponly += int(gp and not gb)
        if Mb[i] is not None and Mp is not None:
            dm.append(abs(matched(np.asarray(Mb[i], float), f) - matched(np.asarray(Mp, float), f)))
    print(f"{label:11s}: gate agrees {agree}/{N}, differs {dis}  "
          f"(batched-only {bonly}, per-frame-only {ponly})")
    if dm:
        dm = np.array(dm)
        print(f"{'':11s}  |matched_b - matched_p|: mean {dm.mean():.2f} max {dm.max()} "
              f"nonzero on {int((dm>0).sum())}/{len(dm)} frames")
