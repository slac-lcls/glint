"""Chase the known-cell completeness gap: 95% get the right LATTICE but only 55% index >=25% of spots.
Is it orientation (fixable) or data (ceiling)? Measure the matched-fraction distribution, and test a
gate-matched final POLISH (re-anneal looser, to the 0.15 gate tolerance not 0.02) + a coverage pick."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import fftindex.glint_fast as gf
from fftindex.glint_fast import index_blind_fast, anneal_batch_t
from fftindex.replica_gpu import index_known_gpu_cell
from fftindex.multishot import same_lattice
DEV = gf.DEV
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames); LYSO = gf.LYSO


def mfrac(M, q):
    if M is None:
        return 0.0
    r = np.asarray(q) @ M; r = r - np.rint(r)
    return float((np.abs(r).max(1) < 0.15).sum()) / len(q)


def polish(M, q, min_thr):
    """re-anneal the placed cell to a gate-matched final tolerance (fit MANY spots at 0.15, not few at 0.02)."""
    Q = torch.as_tensor(np.asarray(q, float), dtype=torch.float64, device=DEV)
    Mt = anneal_batch_t(torch.as_tensor(M[None], dtype=torch.float64, device=DEV), Q,
                        thr0=0.30, contract=0.85, max_iter=15, min_thr=min_thr)
    return Mt[0].cpu().numpy()


index_known_gpu_cell(frames[0], LYSO)                                    # warmup
resc = [index_known_gpu_cell(q, LYSO) for q in frames]
sl = [r is not None and same_lattice(r, LYSO) for r in resc]
print(f"correct-lattice: {sum(sl)}/{n} = {100*sum(sl)//n}%")
# baseline matched-fraction among correct-lattice
mf0 = [mfrac(r, q) for r, q, s in zip(resc, frames, sl) if s]
mf0 = np.array(mf0)
print(f"baseline >=25% (of correct-lattice): {int((mf0>=0.25).sum())}/{len(mf0)} = {100*int((mf0>=0.25).sum())//len(mf0)}%")
print(f"matched-frac among correct-lattice: min/med/mean/p75 = {mf0.min():.2f}/{np.median(mf0):.2f}/{mf0.mean():.2f}/{np.percentile(mf0,75):.2f}")
edges = [0, .10, .15, .20, .25, .30, .50, 1.01]
h, _ = np.histogram(mf0, bins=edges)
print("matched-frac hist [0-.10 .10-.15 .15-.20 .20-.25 .25-.30 .30-.50 .50+]:", h.tolist())

# POLISH: re-anneal to gate-matched tolerances, per-frame keep the best matched-frac
for mt in (0.08, 0.12, 0.15):
    mfp = []
    for r, q, s in zip(resc, frames, sl):
        if not s:
            continue
        rp = polish(r, q, mt)
        mfp.append(max(mfrac(r, q), mfrac(rp, q)))     # keep whichever indexes more (never worse)
    mfp = np.array(mfp)
    print(f"polish min_thr={mt}: >=25% {int((mfp>=0.25).sum())}/{len(mfp)} = {100*int((mfp>=0.25).sum())//len(mfp)}%   (was {100*int((mf0>=0.25).sum())//len(mf0)}%)")

# reference: blind path >=25% on ITS correct-lattice frames (does its coverage scorer index more?)
bl = [index_blind_fast(q) for q in frames]
bsl = [b is not None and same_lattice(b, LYSO) for b in bl]
bmf = np.array([mfrac(b, q) for b, q, s in zip(bl, frames, bsl) if s])
print(f"blind: correct-lattice {sum(bsl)}/{n}; of those >=25%: {int((bmf>=0.25).sum())}/{len(bmf)} = {100*int((bmf>=0.25).sum())//len(bmf)}%")
