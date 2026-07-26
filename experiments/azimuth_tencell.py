"""Dense 10-cell sweep through the SHIPPED azimuth dispatch (no arm override).

azimuth_coverage.py forces each arm with set_grid() to compare half vs full. This runs the
standard 10-cell set through index_known_gpu_cell exactly as production does it -- _azimuth_grid
picks half or full per cell from c01 -- and reports which grid each cell actually received, so the
"100%" claim is about the code that ships rather than about a forced arm.

  python azimuth_tencell.py [ntrial]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glint.replica_gpu as rg
from glint.replica_gpu import index_known_gpu_cell
from experiments.azimuth_coverage import CELLS, cell_to_A, rand_rot, rlps, gate, DENSE_CAP

NTRIAL = int(sys.argv[1]) if len(sys.argv) > 1 else 100

print(f"dense 10-cell sweep, SHIPPED dispatch  ntrial={NTRIAL}/cell  dev={rg.DEV}  NANG={rg.NANG}")
print(f"AZ_TOL={rg.AZ_TOL:g}   half-turn grid = {len(rg.CA)} samples over [0,pi), "
      f"full = {len(rg.CA2)} over [0,2pi)")
print(f"\n{'cell':<11}{'c01':>10}{'grid':>7}{'n_rlp':>7}{'rate':>8}{'ms/fr':>8}")
tot = nfr = 0
for name, cp in CELLS:
    A0 = cell_to_A(*cp)
    _, c01, _, _, _ = rg._axes_from_cell(A0)
    ca, _sa = rg._azimuth_grid(c01)
    grid = "half" if ca is rg.CA else "full"
    rng = np.random.default_rng(abs(hash(name)) % 2**31)
    frames, Mcs = [], []
    for t in range(NTRIAL):
        Ar = rand_rot(rng) @ A0
        q = rlps(Ar, rng)
        if len(q) > DENSE_CAP:
            q = q[rng.choice(len(q), DENSE_CAP, replace=False)]
        if len(q) >= 10:
            frames.append(q); Mcs.append(Ar)
    index_known_gpu_cell(frames[0], Mcs[0])                      # warmup
    if rg.DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = [index_known_gpu_cell(q, Mc) for q, Mc in zip(frames, Mcs)]
    if rg.DEV == "cuda":
        torch.cuda.synchronize()
    ms = 1e3 * (time.perf_counter() - t0) / len(frames)
    ok = sum(gate(M, q, Mc) for M, q, Mc in zip(res, frames, Mcs))
    tot += ok; nfr += len(frames)
    med = int(np.median([len(f) for f in frames]))
    flag = "" if ok == len(frames) else f"   <-- {len(frames)-ok} MISS"
    print(f"{name:<11}{c01:+10.4f}{grid:>7}{med:>7}{100*ok/len(frames):7.1f}%{ms:8.1f}{flag}", flush=True)

print(f"\nTOTAL {tot}/{nfr} = {100*tot/nfr:.1f}%"
      + ("   ALL CELLS 100%" if tot == nfr else "   NOT 100% -- see MISS rows above"))
