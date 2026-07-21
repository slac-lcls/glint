"""#36: validate M3_CAP -- cap the ASCENT, keep every peak for scoring.

The knob LUTE exposes today (`top_peaks` -> frames_from_cxi(top_n=)) truncates the FRAME, so the
short list feeds M2/M4/M5/M6 as well, and costs 11 points of correct-lattice at cap 100 on real
data. M3_CAP caps only the ascent. Both spend the same peak budget on M3; only one throws away
information the scorer needs.

Checks, driving the real env flag via importlib.reload rather than monkeypatching:

  1. M3_CAP vs the ingest truncation at matched cap -- the point of #36
  2. M3_CAP composes with M3_FUSED (they attack the SAME term, so the speedups must NOT be assumed
     to multiply -- measured, not argued)
  3. M3_CAP=0 is an exact no-op (bit-identical cells), so the default path is untouched

  python validate_m3_cap.py
"""
import os, sys, time, importlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from glint.multishot import same_lattice
from glint.lute_bridge import frames_from_cxi

CXI = os.path.join(HERE, "cf_peaks.cxi")
GEOM = os.path.join(HERE, "cf.geom")


def load_gf(cap=0, fused=False):
    os.environ["M3_CAP"] = str(cap)
    os.environ["M3_FUSED"] = "1" if fused else "0"
    import glint.glint_fast as gf
    importlib.reload(gf)
    return gf


gf0 = load_gf(0, False)
LYSO = gf0.LYSO
frames_full = [q for q in frames_from_cxi(CXI, GEOM, peakfinder="stored", top_n=0, min_peaks=6)[0]
               if len(q) >= 6]
n = len(frames_full)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gate(M, q):
    if M is None or not same_lattice(np.asarray(M, float), LYSO):
        return 0
    r = np.asarray(q) @ M
    m = int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())
    return int(m / len(q) >= 0.25 and m >= 10)


def run(frames, cap, fused):
    gf = load_gf(cap, fused)
    gf.index_blind_fast(frames[0]); sync()
    t0 = time.perf_counter()
    Ms = [gf.index_blind_fast(q) for q in frames]
    sync(); ms = 1e3 * (time.perf_counter() - t0) / len(frames)
    lat = sum(int(M is not None and same_lattice(np.asarray(M, float), LYSO)) for M in Ms)
    # gate always against the FULL frame, so truncated arms are not flattered
    g = sum(gate(M, q) for M, q in zip(Ms, frames_full))
    return lat, g, ms, Ms


print(f"M3_CAP validation, {n} real frames from cf_peaks.cxi, "
      f"peaks median {int(np.median([len(f) for f in frames_full]))} "
      f"max {max(len(f) for f in frames_full)}\n")

print("1. M3-only cap vs ingest truncation, matched cap")
print(f"{'arm':<34}{'same_lat':>10}{'gate':>8}{'ms/frame':>11}{'speedup':>9}")
b_lat, b_g, b_ms, b_M = run(frames_full, 0, False)
print(f"{'baseline (no cap)':<34}{f'{b_lat}/{n}':>10}{f'{b_g}/{n}':>8}{b_ms:11.2f}{'--':>9}")
for cap in (200, 100, 75):
    lat, g, ms, _ = run(frames_full, cap, False)
    print(f"{'M3_CAP=' + str(cap) + ' (ascent only)':<34}{f'{lat}/{n}':>10}{f'{g}/{n}':>8}"
          f"{ms:11.2f}{b_ms/ms:8.2f}x")
    tr = [q for q in frames_from_cxi(CXI, GEOM, peakfinder="stored", top_n=cap, min_peaks=6)[0]
          if len(q) >= 6]
    lat2, g2, ms2, _ = run(tr, 0, False)
    print(f"{'  top_peaks=' + str(cap) + ' (frame truncated)':<34}{f'{lat2}/{len(tr)}':>10}"
          f"{f'{g2}/{len(tr)}':>8}{ms2:11.2f}{b_ms/ms2:8.2f}x")

print("\n2. does M3_CAP compose with M3_FUSED? (same term -- do NOT assume they multiply)")
print(f"{'arm':<34}{'same_lat':>10}{'gate':>8}{'ms/frame':>11}{'speedup':>9}")
for cap, fused in ((0, False), (0, True), (100, False), (100, True)):
    lat, g, ms, _ = run(frames_full, cap, fused)
    label = f"M3_CAP={cap or '-'} M3_FUSED={int(fused)}"
    print(f"{label:<34}{f'{lat}/{n}':>10}{f'{g}/{n}':>8}{ms:11.2f}{b_ms/ms:8.2f}x")

print("\n3. M3_CAP=0 must be an exact no-op")
lat0, g0, _, M0 = run(frames_full, 0, False)
ident = sum(int((a is None) == (b is None) and (a is None or np.array_equal(np.asarray(a), np.asarray(b))))
            for a, b in zip(b_M, M0))
print(f"   cells bit-identical to baseline: {ident}/{n}  {'OK' if ident == n else '*** DEFAULT PATH MOVED ***'}")
os.environ["M3_CAP"] = "0"; os.environ["M3_FUSED"] = "0"
