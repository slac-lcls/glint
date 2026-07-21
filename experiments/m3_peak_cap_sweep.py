"""#28 last lever: cap the peaks fed to M3.

M3 is 66% of the blind frame and LINEAR in peak count (64/128/256/512 peaks -> 3.67/8.01/15.06/28.91
ms), and its 8.34 ms/frame mean is dragged well above the median frame by peak-rich ones -- the frame
profiled in profile_blind_frontend.py had 370 peaks and spent 23 ms in M3 alone. Median is 100.
`index_blind_fast` applies no cap, while the `--images` path exposes `top_peaks` precisely because
weak peaks are known to HURT the rate (~100 strongest is the measured sweet spot).

Two arms, because they are NOT the same experiment:

  M3-only   cap the peaks used for the gradient ascent, keep ALL peaks for M2 scoring / M4 / M5 / M6.
            M3 only has to FIND candidate directions; the full set still judges them. This decouples
            cost from information loss and is the variant worth wanting.
  all       subsample the frame outright -- the naive version, as a control. If M3-only beats it,
            the cap is buying time rather than throwing away signal.

CAVEAT on selection. frames_cxidb_clean.txt carries q-vectors only, no intensities, so "keep the N
strongest" is NOT reproducible here. Two proxies are swept instead:
  inner   lowest |q| -- inner resolution shells, typically the strong, well-determined reflections
  first   as stored, i.e. peakfinder8 output order
If the two disagree, selection RULE matters and an intensity-ranked cap needs testing on real
`--images` data before any of this ships.

  python m3_peak_cap_sweep.py
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import glint.glint_fast as gf
from glint.glint_index import refine_vec
from glint.multishot import same_lattice

frames = [q for q in gf.load(os.path.join(HERE, "frames_cxidb_clean.txt")) if len(q) >= 6]
n = len(frames)
LYSO = gf.LYSO
_orig_refine = gf._refine


def sync():
    if gf.DEV == "cuda":
        torch.cuda.synchronize()


def gate(M, q):
    if M is None or not same_lattice(np.asarray(M, float), LYSO):
        return 0
    r = np.asarray(q) @ M
    m = int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())
    return int(m / len(q) >= 0.25 and m >= 10)


def keep_idx(Q, cap, rule):
    P = int(Q.shape[0])
    if cap <= 0 or P <= cap:
        return None
    if rule == "inner":
        return torch.argsort(Q.norm(dim=1))[:cap]
    return torch.arange(cap, device=Q.device)              # "first": as stored


def patch_m3(cap, rule):
    """Cap ONLY the ascent. qmax stays the full-frame value, so the step schedule is unchanged."""
    def _refine(S0, Q, w, qmax):
        idx = keep_idx(Q, cap, rule)
        if idx is None:
            return _orig_refine(S0, Q, w, qmax)
        return refine_vec(S0, Q[idx], w[idx], qmax, steps=gf.STEPS, tol=gf.TOL)
    gf._refine = _refine


def run(label, cap, rule, scope):
    if scope == "all":
        fr = []
        for q in frames:
            Q = torch.as_tensor(np.asarray(q, float), dtype=torch.float32)
            idx = keep_idx(Q, cap, rule)
            fr.append(np.asarray(q, float) if idx is None else np.asarray(q, float)[idx.cpu().numpy()])
        gf._refine = _orig_refine
    else:
        fr = frames
        patch_m3(cap, rule) if cap else setattr(gf, "_refine", _orig_refine)
    try:
        gf.index_blind_fast(fr[0]); sync()
        t0 = time.perf_counter()
        Ms = [gf.index_blind_fast(q) for q in fr]
        sync(); ms = 1e3 * (time.perf_counter() - t0) / len(fr)
    finally:
        gf._refine = _orig_refine
    lat = sum(int(M is not None and same_lattice(np.asarray(M, float), LYSO)) for M in Ms)
    # gate against the FULL frame either way, so the arms are comparable
    g = sum(gate(M, q) for M, q in zip(Ms, frames))
    return lat, g, ms


pk = [len(f) for f in frames]
print(f"M3 peak-cap sweep, {n} cxidb frames, {gf.DEV}, STEPS={gf.STEPS}")
print(f"peaks/frame: median {int(np.median(pk))}  mean {np.mean(pk):.0f}  p90 {int(np.percentile(pk,90))}"
      f"  max {max(pk)}\n")
print(f"{'arm':<22}{'cap':>5}{'same_lat':>10}{'gate':>8}{'ms/frame':>11}{'speedup':>9}")
b_lat, b_g, b_ms = run("baseline", 0, "inner", "m3")
print(f"{'baseline (no cap)':<22}{'-':>5}{f'{b_lat}/{n}':>10}{f'{b_g}/{n}':>8}{b_ms:11.2f}{'--':>9}")

for rule in ("inner", "first"):
    for cap in (200, 150, 100, 75, 50):
        lat, g, ms = run(f"M3-only/{rule}", cap, rule, "m3")
        flag = "" if (lat >= b_lat and g >= b_g) else "   <-- LOSES"
        print(f"{'M3-only cap/' + rule:<22}{cap:>5}{f'{lat}/{n}':>10}{f'{g}/{n}':>8}{ms:11.2f}"
              f"{b_ms/ms:8.2f}x{flag}")

print()
for cap in (150, 100, 75):
    lat, g, ms = run("all/inner", cap, "inner", "all")
    print(f"{'CONTROL all-stages/inner':<22}{cap:>5}{f'{lat}/{n}':>10}{f'{g}/{n}':>8}{ms:11.2f}{b_ms/ms:8.2f}x")

print("\nM3-only should beat all-stages at equal cap: the ascent needs fewer peaks to FIND a")
print("direction than the scorer needs to JUDGE it. If inner and first disagree, the selection rule")
print("matters and an intensity-ranked cap must be tested on real --images data before shipping.")
