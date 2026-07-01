"""Is the 95%-lattice -> 58%-gate drop ORIENTATION (rescue-specific) or DATA (harder frames lack indexable
spots)? Compare rescue vs blind on the SAME frames (both correct-lattice) and profile the rescue-only
(blind-failed) frames' matched-fraction."""
import os, sys
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import glint.glint_fast as gf
from glint.glint_fast import index_blind_fast
from glint.replica_gpu import index_known_gpu_cell
from glint.multishot import same_lattice
DEV = gf.DEV
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames); LYSO = gf.LYSO


def mfrac(M, q):
    if M is None or not same_lattice(M, LYSO):
        return None
    r = np.asarray(q) @ M; r = r - np.rint(r)
    return float((np.abs(r).max(1) < 0.15).sum()) / len(q)


index_blind_fast(frames[0]); index_known_gpu_cell(frames[0], LYSO)
mr = [mfrac(index_known_gpu_cell(q, LYSO), q) for q in frames]     # rescue matched-frac (None if wrong lattice)
mb = [mfrac(index_blind_fast(q), q) for q in frames]               # blind matched-frac

both = [(a, b) for a, b in zip(mr, mb) if a is not None and b is not None]
ronly = [a for a, b in zip(mr, mb) if a is not None and b is None]  # rescue correct, blind failed
print(f"rescue correct-lattice {sum(a is not None for a in mr)}/{n}; blind {sum(b is not None for b in mb)}/{n}")
print(f"=== overlap (both correct-lattice): {len(both)} frames ===")
ra = np.array([a for a, b in both]); ba = np.array([b for a, b in both])
print(f"  rescue >=25%: {int((ra>=0.25).sum())}/{len(ra)} = {100*int((ra>=0.25).sum())//len(ra)}%   mean mfrac {ra.mean():.2f}")
print(f"  blind  >=25%: {int((ba>=0.25).sum())}/{len(ba)} = {100*int((ba>=0.25).sum())//len(ba)}%   mean mfrac {ba.mean():.2f}")
print(f"  rescue indexes >= blind on {int((ra>=ba-1e-9).sum())}/{len(ra)} of the shared frames")
if ronly:
    ro = np.array(ronly)
    print(f"=== rescue-ONLY (blind failed): {len(ro)} frames ===")
    print(f"  >=25%: {int((ro>=0.25).sum())}/{len(ro)} = {100*int((ro>=0.25).sum())//len(ro)}%   mean mfrac {ro.mean():.2f} (these drag the average)")
