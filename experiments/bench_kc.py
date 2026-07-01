"""GLINT bench on LUTE .cxi peaks via working lute_bridge path.
Usage: bench_kc.py <cxi> <geom> [N] [blind|known]
  blind  = GLINT-1 (n-best + consensus + rescue), self-derived cell
  known  = per-frame known-cell rescue against maybe_lyso (= ported ffbidx)"""
import sys, time
sys.path.insert(0, "..")
import numpy as np, h5py
from fftindex.lute_bridge import parse_geom, peaks_to_q, lambda_from_eV
from fftindex.lattice import cell_to_Ar

cxi, geom_path = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
mode = sys.argv[4] if len(sys.argv) > 4 else "known"
REF = (28.0, 62.5, 60.9, 90.0, 90.8, 90.0)

panels, glob = parse_geom(geom_path)
clen_p = glob.get("clen", "").strip(); phot_p = glob.get("photon_energy", "").strip()
frames = []
with h5py.File(cxi, "r") as f:
    nP = np.asarray(f["/entry_1/result_1/nPeaks"])
    X = f["/entry_1/result_1/peakXPosRaw"]; Y = f["/entry_1/result_1/peakYPosRaw"]
    nev = min(N, len(nP))
    def rd(p, e):
        a = np.asarray(f[p]); return float(a[e]) if a.ndim and a.shape[0] == len(nP) else float(a.ravel()[0])
    for e in range(nev):
        k = int(nP[e])
        if k < 6: continue
        fs, ss = np.asarray(X[e])[:k], np.asarray(Y[e])[:k]
        q = peaks_to_q(fs, ss, panels, rd(clen_p, e)/1000.0, lambda_from_eV(rd(phot_p, e)))
        frames.append(q[~np.isnan(q).any(1)])
n = max(len(frames), 1)
print(f"MODE={mode}  frames={len(frames)}  ref cell {REF}")

if mode == "blind":
    from fftindex.hybrid_stream import hybrid_index, _report
    t0 = time.time(); results, stats = hybrid_index(frames, None, Mc_known=None, nbest=3); dt = time.time()-t0
    _report(stats, "(bench)")
    print(f"TIMING: {dt:.1f}s, {1000*dt/n:.1f} ms/frame, {n/max(dt,1e-9):.1f} f/s")
else:
    import torch
    from fftindex.replica_gpu import index_known_gpu_cell
    from fftindex.multishot import same_lattice
    Mc = cell_to_Ar(*REF)
    if frames: index_known_gpu_cell(frames[0], Mc)              # warmup
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time(); solved = matched = tot = 0
    for q in frames:
        M = index_known_gpu_cell(q, Mc)
        if M is None: continue
        solved += 1
        if not same_lattice(M, Mc): continue
        m = int((np.abs(q @ M - np.rint(q @ M)).max(1) < 0.15).sum())
        if m >= 10: matched += 1; tot += m
    if torch.cuda.is_available(): torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"=== KNOWN-CELL per-frame (replica_gpu = ported ffbidx, cell given) ===")
    print(f"  any-solution     : {solved}/{len(frames)}")
    print(f"  indexed (>=10 refl, right cell): {matched}/{len(frames)} ({100*matched//n}%)")
    print(f"  mean inlier-refl/indexed: {tot/max(matched,1):.1f}   total inlier-refl: {tot}")
    print(f"TIMING: {dt:.1f}s, {1000*dt/n:.1f} ms/frame, {n/max(dt,1e-9):.1f} f/s")
