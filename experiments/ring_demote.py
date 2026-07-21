"""Radial-excess demotion: are the ring peaks spurious, and does removing them help?

The diagnostic (peak_background_rank.py) found a real ring in cf_peaks.cxi -- 6.14x excess over a
shell-area null at d=3.92 A, water's main diffuse ring being ~3.2 A. Intensity cannot express it
(peakTotalIntensity is a constant 1000.0 for all 16545 peaks), but POSITION can: peaks whose |q|
falls in a radially over-dense shell are more likely to be ring artifacts.

Two design points that decide whether this measures anything:

  NULL. "Excess over a shell-area null" conflates the ring with the resolution-dependent falloff of
  real Bragg peaks. A ring is a NARROW feature, so the null here is a SMOOTHED version of the
  observed radial profile (running median over a wide window). Narrow excesses stand out; broad
  trends are absorbed.

  CONTROL. Dropping any peaks changes the rate, so ring-targeted removal is compared against RANDOM
  removal of the same count per frame. Without that control, "removing 10% of peaks changed the
  rate" says nothing about rings.

The informative outcome is symmetric:
  ring-removal BEATS random  -> the excess really is contamination, and position-based demotion works
  ring-removal TIES random   -> the excess is just where reflections are; nothing to exploit
  ring-removal LOSES         -> those peaks are real signal; the ring shells are INFORMATIVE

  python ring_demote.py
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import glint.glint_fast as gf
from glint.lute_bridge import frames_from_cxi
from glint.multishot import same_lattice

CXI = os.path.join(HERE, "cf_peaks.cxi")
GEOM = os.path.join(HERE, "cf.geom")
LYSO = gf.LYSO
frames = [q for q in frames_from_cxi(CXI, GEOM, peakfinder="stored", top_n=0, min_peaks=6)[0]
          if len(q) >= 6]
n = len(frames)
QN = [np.linalg.norm(q, axis=1) for q in frames]
allq = np.concatenate(QN)


def sync():
    if gf.DEV == "cuda":
        torch.cuda.synchronize()


def gate(M, q):
    if M is None or not same_lattice(np.asarray(M, float), LYSO):
        return 0
    r = np.asarray(q) @ M
    m = int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())
    return int(m / len(q) >= 0.25 and m >= 10)


# ---- ring score: excess over a SMOOTHED radial profile ---------------------------------------
NB, WIN = 80, 15
edges = np.linspace(allq.min(), allq.max() + 1e-9, NB + 1)
cnt, _ = np.histogram(allq, bins=edges)
pad = np.pad(cnt.astype(float), WIN // 2, mode="edge")
smooth = np.array([np.median(pad[i:i + WIN]) for i in range(NB)])       # running median = the null
excess = cnt / np.maximum(smooth, 1.0)
mid = 0.5 * (edges[:-1] + edges[1:])
top = np.argsort(excess)[::-1][:5]
print(f"{n} frames, {len(allq)} peaks. Ring score = count / running-median({WIN} bins).")
print("strongest narrow excesses:")
for b in sorted(top):
    print(f"   |q|={mid[b]:.3f}  d={1/mid[b]:5.2f} A   count {cnt[b]:5d}  smooth {smooth[b]:7.1f}"
          f"   excess {excess[b]:.2f}x")

score = [excess[np.clip(np.digitize(qq, edges) - 1, 0, NB - 1)] for qq in QN]
rng = np.random.default_rng(0)


def subset(frac, mode):
    """Drop frac of each frame's peaks: 'ring' = highest ring score, 'random' = uniform."""
    out = []
    for q, s in zip(frames, score):
        k = len(q); ndrop = int(round(frac * k))
        if ndrop == 0 or k - ndrop < 6:
            out.append(q); continue
        if mode == "ring":
            keep = np.argsort(s)[:k - ndrop]                # lowest ring score survives
        else:
            keep = rng.choice(k, k - ndrop, replace=False)
        out.append(q[np.sort(keep)])
    return out


def run(fr):
    gf.index_blind_fast(fr[0]); sync()
    t0 = time.perf_counter()
    Ms = [gf.index_blind_fast(q) for q in fr]
    sync(); ms = 1e3 * (time.perf_counter() - t0) / len(fr)
    lat = sum(int(M is not None and same_lattice(np.asarray(M, float), LYSO)) for M in Ms)
    g = sum(gate(M, q) for M, q in zip(Ms, frames))      # gate on the FULL frame, always
    return lat, g, ms


print(f"\n{'arm':<28}{'same_lat':>10}{'gate':>8}{'med peaks':>11}{'ms/frame':>11}")
b_lat, b_g, b_ms = run(frames)
print(f"{'baseline (all peaks)':<28}{f'{b_lat}/{n}':>10}{f'{b_g}/{n}':>8}"
      f"{int(np.median([len(q) for q in frames])):>11}{b_ms:11.2f}")
for frac in (0.05, 0.10, 0.20, 0.35):
    for mode in ("ring", "random"):
        fr = subset(frac, mode)
        lat, g, ms = run(fr)
        d = f"{lat - b_lat:+d}/{g - b_g:+d}"
        print(f"{'drop ' + str(int(frac*100)) + '% ' + mode:<28}{f'{lat}/{n}':>10}{f'{g}/{n}':>8}"
              f"{int(np.median([len(q) for q in fr])):>11}{ms:11.2f}   {d}")

print("\nRead the RING vs RANDOM pair at each fraction, not the absolute numbers: both drop the same")
print("count, so any difference between them is attributable to WHICH peaks went.")

# ---- HELD-OUT: the profile above is built from the same 120 frames it is applied to ------------
# Not oracle leakage (no labels are used), but it IS in-sample: the radial profile is tuned to this
# corpus. A deployment would build the profile from a warmup set and apply it to later frames, so
# split half/half and score each half with the OTHER half's profile.
print("\nheld-out check: profile from one half, applied to the other")
half = {"A": list(range(0, n, 2)), "B": list(range(1, n, 2))}


def profile_from(idxs):
    pooled = np.concatenate([QN[i] for i in idxs])
    c, _ = np.histogram(pooled, bins=edges)
    p = np.pad(c.astype(float), WIN // 2, mode="edge")
    sm = np.array([np.median(p[j:j + WIN]) for j in range(NB)])
    return c / np.maximum(sm, 1.0)


def run_idx(idxs, frac, mode, exc):
    fr, full = [], []
    for i in idxs:
        q = frames[i]; k = len(q)
        s = exc[np.clip(np.digitize(QN[i], edges) - 1, 0, NB - 1)]
        ndrop = int(round(frac * k))
        if ndrop == 0 or k - ndrop < 6:
            fr.append(q)
        else:
            keep = np.argsort(s)[:k - ndrop] if mode == "ring" else rng.choice(k, k - ndrop, replace=False)
            fr.append(q[np.sort(keep)])
        full.append(q)
    gf.index_blind_fast(fr[0]); sync()
    Ms = [gf.index_blind_fast(q) for q in fr]
    lat = sum(int(M is not None and same_lattice(np.asarray(M, float), LYSO)) for M in Ms)
    g = sum(gate(M, q) for M, q in zip(Ms, full))
    return lat, g, len(idxs)


print(f"{'held-out half':<18}{'arm':<16}{'same_lat':>10}{'gate':>8}")
for tr, te in (("A", "B"), ("B", "A")):
    exc = profile_from(half[tr])
    for mode in ("none", "ring", "random"):
        lat, g, m = run_idx(half[te], 0.0 if mode == "none" else 0.10, mode, exc)
        lbl = "baseline" if mode == "none" else f"drop 10% {mode}"
        print(f"{'profile ' + tr + ' -> ' + te:<18}{lbl:<16}{f'{lat}/{m}':>10}{f'{g}/{m}':>8}")
print("If ring still beats random here, the score is not an artefact of fitting the test set.")
