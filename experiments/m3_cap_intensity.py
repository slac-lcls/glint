"""#28: the M3 peak cap ranked by INTENSITY -- the rule --images actually uses.

m3_peak_cap_sweep.py could only sweep proxies (`first` = peakfinder8 order, `inner` = lowest |q|)
because frames_cxidb_clean.txt carries q-vectors only. They disagreed -- 84/80 vs 81/76 at cap 100 --
so the real rule could land anywhere between them, and `--images top_peaks` ranks by intensity
(lute_bridge.frames_from_cxi: `s = pk["intensity"]; keep = argsort(s)[::-1][:top_n]`).

experiments/cf_peaks.cxi is the SOURCE of that frames file -- 120 frames, max 554 peaks, matching
the corpus exactly -- and it carries /entry_1/result_1/peakTotalIntensity. So the real ranking is
testable on the same frames, directly comparable to the proxy numbers.

The correspondence is VERIFIED, not assumed: per-frame peak counts must match nPeaks, or the
intensities cannot be aligned to the q-vectors and the run aborts.

  python m3_cap_intensity.py
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import glint.glint_fast as gf
from glint.glint_index import refine_vec
from glint.multishot import same_lattice

CXI = os.path.join(HERE, "cf_peaks.cxi")
frames = [q for q in gf.load(os.path.join(HERE, "frames_cxidb_clean.txt")) if len(q) >= 6]
n = len(frames)
LYSO = gf.LYSO
_orig_refine = gf._refine

# ---- align the .cxi intensities to the frames file ------------------------------------------
with h5py.File(CXI, "r") as f:
    npk = np.asarray(f["/entry_1/result_1/nPeaks"])
    ints_all = np.asarray(f["/entry_1/result_1/peakTotalIntensity"])
allI = [ints_all[i, :int(npk[i])] for i in range(len(npk))]
raw = [q for q in gf.load(os.path.join(HERE, "frames_cxidb_clean.txt"))]
print(f"alignment check: cxi frames {len(npk)}  frames-file {len(raw)}  (>=6 peaks: {n})")
mism = [i for i in range(min(len(npk), len(raw))) if int(npk[i]) != len(raw[i])]
if len(npk) != len(raw) or mism:
    print(f"*** ABORT: per-frame peak counts differ (first mismatches {mism[:5]}). The intensities")
    print("    cannot be aligned to these q-vectors, so an intensity rank here would be fiction.")
    sys.exit(1)
print("  per-frame counts match on all frames -> intensities align with the q-vectors\n")
INT = [allI[i] for i, q in enumerate(raw) if len(q) >= 6]


def sync():
    if gf.DEV == "cuda":
        torch.cuda.synchronize()


def gate(M, q):
    if M is None or not same_lattice(np.asarray(M, float), LYSO):
        return 0
    r = np.asarray(q) @ M
    m = int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())
    return int(m / len(q) >= 0.25 and m >= 10)


def order_for(i, Q, rule):
    if rule == "intensity":
        return torch.as_tensor(np.argsort(INT[i])[::-1].copy(), device=Q.device)
    if rule == "inner":
        return torch.argsort(Q.norm(dim=1))
    return torch.arange(int(Q.shape[0]), device=Q.device)


def run(cap, rule):
    """Cap ONLY the ascent; M2/M4/M5/M6 keep every peak. qmax stays the full-frame value."""
    state = {"i": 0}

    def _refine(S0, Q, w, qmax):
        i = state["i"]
        if cap and int(Q.shape[0]) > cap:
            idx = order_for(i, Q, rule)[:cap]
            return refine_vec(S0, Q[idx], w[idx], qmax, steps=gf.STEPS, tol=gf.TOL)
        return _orig_refine(S0, Q, w, qmax)

    gf._refine = _refine
    try:
        state["i"] = 0; gf.index_blind_fast(frames[0]); sync()
        t0 = time.perf_counter(); Ms = []
        for i, q in enumerate(frames):
            state["i"] = i
            Ms.append(gf.index_blind_fast(q))
        sync(); ms = 1e3 * (time.perf_counter() - t0) / n
    finally:
        gf._refine = _orig_refine
    lat = sum(int(M is not None and same_lattice(np.asarray(M, float), LYSO)) for M in Ms)
    g = sum(gate(M, q) for M, q in zip(Ms, frames))
    return lat, g, ms


print(f"M3 peak cap by INTENSITY vs proxies, {n} frames, {gf.DEV}, STEPS={gf.STEPS}")
print(f"{'rule':<12}{'cap':>5}{'same_lat':>10}{'gate':>8}{'ms/frame':>11}{'speedup':>9}")
b_lat, b_g, b_ms = run(0, "first")
print(f"{'baseline':<12}{'-':>5}{f'{b_lat}/{n}':>10}{f'{b_g}/{n}':>8}{b_ms:11.2f}{'--':>9}")
best = {}
for rule in ("intensity", "first", "inner"):
    for cap in (200, 150, 100, 75, 50):
        lat, g, ms = run(cap, rule)
        best[(rule, cap)] = (lat, g, ms)
        mark = "  <-- ties/beats baseline" if (lat >= b_lat and g >= b_g) else ""
        print(f"{rule:<12}{cap:>5}{f'{lat}/{n}':>10}{f'{g}/{n}':>8}{ms:11.2f}{b_ms/ms:8.2f}x{mark}")

print("\nverdict")
for cap in (150, 100, 75):
    row = " | ".join(f"{r}: {best[(r,cap)][0]}/{best[(r,cap)][1]}" for r in ("intensity", "first", "inner"))
    print(f"  cap {cap:>3}  (same_lat/gate)   {row}")
print("\nIf intensity does not beat `first`, the peakfinder8 output order already carries the")
print("ranking and `top_peaks` is not buying a better subset -- just a smaller one.")
