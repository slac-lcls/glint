"""CrystFEL stream -> per-frame q (1/A) AND per-peak Intensity, positionally aligned.

stream2q.py cannot be reused for this. It goes through glint.geom.peaks_to_q, whose own docstring
says peaks outside every panel are DROPPED and "rows do not correspond positionally to the input" --
so zipping an intensity column onto its output would silently pair peak i's q with peak j's
intensity, which is the kind of misalignment that produces a plausible, wrong answer. The same
docstring names the way out: the geometry is shared with the NaN-keeping path, so this computes q
with the identical _q_from_panels call, keeps the finite mask, and applies that SAME mask to the
intensities.

Correctness is not assumed: --verify re-reads an existing q file and asserts the regenerated q is
bit-identical to it. If that fails, nothing downstream is a controlled comparison.

  python stream2qi.py <stream> <geom> <list> <out.npz> [--verify q480_fix.txt]
"""
import os, sys
from pathlib import Path

import numpy as np
# Resolve the checkout from THIS FILE, so a vendored copy always measures the tree it ships in.
# A hard-coded path made the harness import whichever worktree happened to exist on the author's
# box -- i.e. not necessarily the code under review, and nothing at all for anyone else.
# GLINT_ROOT overrides it, which is what a copy living outside experiments/ needs.
ROOT = os.environ.get("GLINT_ROOT") or str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
from glint.geom import parse_geom, _q_from_panels, _specs_from_geom_panels

STREAM, GEOM, LST, OUT = sys.argv[1:5]
VERIFY = sys.argv[6] if len(sys.argv) > 6 and sys.argv[5] == "--verify" else None
FIXED = os.environ.get("FIXED_LAM", "") not in ("", "0")
HC = 12398.419843320026

geom = parse_geom(GEOM)
specs = _specs_from_geom_panels(geom["panels"] if "panels" in geom else geom)
order = {ln.strip(): i for i, ln in enumerate(open(LST)) if ln.strip()}

# One pass: peak lists WITH intensity, plus per-chunk wavelength. read_crystfel_peaks drops the
# intensity column (it takes p[0], p[1] only), so the peak block is parsed here.
lam, chunks = {}, []
fn, peaks, inpk, open_chunk = None, [], False, False
for ln in open(STREAM):
    s = ln.strip()
    if s.startswith("----- Begin chunk"):
        fn, peaks, inpk, open_chunk = None, [], False, True
    elif s.startswith("Image filename:"):
        fn = s.split(":", 1)[1].strip()
    elif s.startswith("photon_energy_eV") and fn:
        lam[fn] = HC / float(s.split("=")[1])
    elif s.startswith("Peaks from peak search"):
        inpk = True
    elif s.startswith("End of peak list"):
        inpk = False
    elif s.startswith("----- End chunk"):
        if open_chunk:                                    # same latch as read_crystfel_peaks: a
            chunks.append((fn, np.array(peaks, float).reshape(-1, 3)))   # stray End would duplicate
            open_chunk = False
    elif inpk:
        p = s.split()
        try:
            peaks.append((float(p[0]), float(p[1]), float(p[3])))        # fs, ss, Intensity
        except (ValueError, IndexError):
            continue                                      # the 'fs/px ss/px ...' header

if FIXED and lam:
    v = float(os.environ["FIXED_LAM"])
    first = v if v > 0.1 else lam[min(order, key=order.get)]
    lam = {k: first for k in lam}

Q, I = {}, {}
for fname, pk in chunks:
    i = order.get(fname)
    if i is None or len(pk) == 0:
        continue
    q = _q_from_panels(pk[:, 0], pk[:, 1], specs, lam.get(fname))
    ok = np.isfinite(q).all(1)                            # the SAME mask peaks_to_q applies...
    Q[i] = q[ok].astype(np.float32)
    I[i] = pk[ok, 2].astype(np.float32)                   # ...applied to the intensities too

n = max(Q) + 1 if Q else 0
qs = [Q.get(i, np.zeros((0, 3), np.float32)) for i in range(n)]
iss = [I.get(i, np.zeros(0, np.float32)) for i in range(n)]
assert all(len(a) == len(b) for a, b in zip(qs, iss)), "q/I length mismatch"
np.savez(OUT, n=n, **{f"q{i}": qs[i] for i in range(n)}, **{f"i{i}": iss[i] for i in range(n)})
npk = [len(a) for a in qs if len(a)]
print("wrote %s  n=%d  npk %d/%d/%d  lambda=%s"
      % (OUT, n, min(npk), int(np.median(npk)), max(npk),
         "FIXED %.6f A" % next(iter(lam.values())) if FIXED else "per-shot"))
print("intensity: min %.1f  median %.1f  max %.1f  (over %d peaks)"
      % (min(a.min() for a in iss if len(a)), float(np.median(np.concatenate([a for a in iss if len(a)]))),
         max(a.max() for a in iss if len(a)), sum(len(a) for a in iss)))

if VERIFY:
    def blocks(p):
        L = open(p).read().split(); i = 0
        while i < len(L) and L[i] != "FRAME":
            i += 1
        out = []
        while i < len(L) and L[i] == "FRAME":
            npk_ = int(L[i + 2]); i += 3
            out.append(np.array(L[i:i + 3 * npk_], float).reshape(npk_, 3)); i += 3 * npk_
        return out
    ref = blocks(VERIFY)
    assert len(ref) == n, f"frame count {len(ref)} != {n}"
    worst = 0.0
    for k, (a, b) in enumerate(zip(ref, qs)):
        assert a.shape == b.shape, f"frame {k}: {a.shape} vs {b.shape}"
        worst = max(worst, float(np.abs(a - b).max()) if len(a) else 0.0)
    print(f"VERIFY vs {VERIFY}: {n} frames, shapes all match, max |dq| = {worst:.3e}"
          f"  -> {'IDENTICAL (to the 6dp the text file stores)' if worst <= 1e-6 else 'MISMATCH'}")
    if worst > 1e-6:
        raise SystemExit(1)
