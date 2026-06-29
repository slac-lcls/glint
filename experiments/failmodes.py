"""Backbone: failure-attribution harness -- 'where is indexing failing and why'.

Classifies each frame's blind-indexing outcome and prints the failure histogram, under a
baseline and under synthetic STRESSORS (mosaic broadening, energy rescale, spurious spots,
a second lattice). Every hard-case direction (D1-D6) plugs in here: add a stressor, read
the histogram, THEN build the fix.

Categories per frame:
  too_few     npk < MINPK
  solved      selected cell same_lattice(LYSO) & frac>=GATE & inl>=MININL
  wrong_cell  selected confidently (frac>=GATE,inl>=MININL) but NOT same_lattice  (confident-wrong)
  sel_miss    true cell reachable among annealed candidates but the selector missed it
  gen_miss    true cell not reachable (candidate generation failed)

  python failmodes.py [frames.txt] [N]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex")
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex/experiments")
from glint_fast import index_blind_fast, load, matched, LYSO, DEV
from fftindex.multishot import same_lattice
from oracle_blind import all_annealed
from fat_ewald import sim_fat

GATE = 0.25
MININL = 10
MINPK = 10
CATS = ["solved", "wrong_cell", "sel_miss", "gen_miss", "too_few"]


def reachable(q):
    for (M, m, fr) in all_annealed(q):
        if fr >= GATE and m >= MININL and same_lattice(M, LYSO):
            return True
    return False


def classify(q):
    if len(q) < MINPK:
        return "too_few"
    M = index_blind_fast(q)
    if M is not None:
        m = matched(M, q)
        if m / len(q) >= GATE and m >= MININL:
            return "solved" if same_lattice(M, LYSO) else "wrong_cell"
    return "sel_miss" if reachable(q) else "gen_miss"


# ---- stressors: each returns a perturbed copy of q (one failure isolated at a time) ----
def s_broaden(q, sig, rng):                    # D2 mosaic / localization: jitter peak positions
    return q + rng.normal(0, sig, q.shape)


def s_rescale(q, dlam, rng):                   # D6 energy jitter: global |q| scale offset
    return q * (1.0 + dlam)


def s_spurious(q, frac, rng):                  # D1 spurious load
    ns = int(round(frac * len(q)))
    qm = float(np.linalg.norm(q, axis=1).max())
    d = rng.normal(size=(ns, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.vstack([q, d * (qm * rng.random(ns) ** (1 / 3))[:, None]])


def s_twin(q, frac, rng):                      # D3/D4 second lattice (same LYSO cell, new orientation)
    g2, _ = sim_fat(0.002, n_cap=max(4, int(frac * len(q))), spur=0.0, jitter=0.0, rng=rng)
    return np.vstack([q, g2]) if len(g2) else q


def run(frames, label, fn=None, seed=0):
    rng = np.random.default_rng(seed)
    h = {c: 0 for c in CATS}
    for q in frames:
        qq = fn(np.asarray(q, float), rng) if fn else np.asarray(q, float)
        h[classify(qq)] += 1
    n = len(frames)
    print(f"  {label:26} " + "  ".join(f"{c}={100 * h[c] // n:3d}%" for c in CATS))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6][:N]
    index_blind_fast(frames[0])                # warmup
    print(f"failure-attribution harness  N={len(frames)}  dev={DEV}  gate frac>={GATE} inl>={MININL}")
    run(frames, "baseline")
    for sig in (0.0003, 0.0006, 0.001, 0.0015, 0.002):
        run(frames, f"mosaic broaden sig={sig}", lambda q, r, s=sig: s_broaden(q, s, r))
    for dl in (0.005, 0.01, 0.02):
        run(frames, f"energy rescale dlam={dl}", lambda q, r, d=dl: s_rescale(q, d, r))
    for fr in (0.5, 1.0):
        run(frames, f"+spurious frac={fr}", lambda q, r, f=fr: s_spurious(q, f, r))
    for fr in (0.5, 1.0):
        run(frames, f"+2nd lattice frac={fr}", lambda q, r, f=fr: s_twin(q, f, r))
