"""Binary vs weighted peaks at n=480: does peak INTENSITY carry lattice information the
transform is currently throwing away?

THE CLAIM. Every method in this family binarises before the transform -- xgandalf, TORO, ffbidx
and GLINT all give each peak the same say. GLINT's weight is `invq_weight(Q)`, i.e. |q|^-1 (times
an optional apodization): purely GEOMETRIC. A 40000-count reflection and a 200-count one at the
same |q| are indistinguishable to M2/M3. That is a modelling choice nobody in the field appears to
have tested, and it is cheap to test here because the weight enters at exactly one place.

WHAT IS VARIED. w = invq_weight(Q) * f(I), normalised so sum(w) matches the control arm's -- the
objective is a weighted sum over peaks, so an arm with systematically larger weights would score
differently for a reason that has nothing to do with the SHAPE of the weighting. Only the shape is
under test.

    binary     f = 1                        the shipped behaviour, the control
    sqrt       f = sqrt(I/med)              amplitude-like
    linear     f = I/med                    intensity-like
    quarter    f = (I/med)^0.25             gentle
    log        f = log1p(I/med)             very gentle, compresses the tail
    rank       f = (rank/N) in (0,1]        distribution-free: immune to the absolute scale and
                                            to pf8's arbitrary intensity units
    inverse    f = med/I                    a FALSIFIER, not a candidate: if up-weighting strong
                                            peaks helps, DOWN-weighting them must hurt. An arm
                                            where both "help" is measuring noise, not intensity.

Two endpoints per arm: blind (what the weight directly controls) and GLINT-(1) hybrid (the
deliverable, where consensus and rescue can absorb a worse blind cell). McNemar on the discordant
pairs against the binary control, because a 3-frame move on 480 is not a result.

INPUT. q comes from the PUBLISHED text q-set, not from the npz, and this is not fussiness: the
text file stores 6 decimals while the extraction keeps full float32, so zero of the 480 frames are
bit-identical between them (max |dq| 5e-7). A first run fed the npz's q and the binary control came
back 91 where the guard pins 92 -- the arms were still comparable with each other, but the control
no longer reproduced the published pipeline, which is the one thing that says the harness is
measuring GLINT. Intensities come from the npz, whose per-frame shapes were verified equal.

  python weight_sweep.py <q480_fix.txt> <qi480.npz> <out.npz> [ARMS]
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

# Resolve the checkout from THIS FILE, so a vendored copy always measures the tree it ships in.
# A hard-coded path made the harness import whichever worktree happened to exist on the author's
# box -- i.e. not necessarily the code under review, and nothing at all for anyone else.
# GLINT_ROOT overrides it, which is what a copy living outside experiments/ needs.
ROOT = os.environ.get("GLINT_ROOT") or str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import torch
import glint.glint_fast as gf
import glint.hybrid_stream as hs
from glint.glint_fast import matched
from glint.multishot import same_lattice
LYSO = gf.LYSO

QTXT, QI, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.load(QI)
qs_txt = gf.load(QTXT)                                  # the exact published input
n_all = int(z["n"])
assert len(qs_txt) == n_all, f"{len(qs_txt)} text frames vs {n_all} in the npz"
frames, inten = [], []
for i in range(n_all):
    q_txt, ii = np.asarray(qs_txt[i], float), np.asarray(z[f"i{i}"], float)
    assert len(q_txt) == len(ii), f"frame {i}: {len(q_txt)} q vs {len(ii)} intensities"
    if len(q_txt) >= 6:
        frames.append(q_txt); inten.append(ii)
n = len(frames)
print(f"# {n} frames (>=6 peaks); q from {QTXT}, intensity from {QI}", flush=True)


def f_binary(I):  return np.ones_like(I)
def f_sqrt(I):    return np.sqrt(np.maximum(I, 0) / np.median(I))
def f_linear(I):  return np.maximum(I, 0) / np.median(I)
def f_quarter(I): return (np.maximum(I, 0) / np.median(I)) ** 0.25
def f_log(I):     return np.log1p(np.maximum(I, 0) / np.median(I))
def f_rank(I):    return (np.argsort(np.argsort(I)) + 1.0) / len(I)
def f_inverse(I): return np.median(I) / np.maximum(I, np.median(I) * 1e-3)

ARMS = [("binary", f_binary), ("sqrt", f_sqrt), ("linear", f_linear), ("quarter", f_quarter),
        ("log", f_log), ("rank", f_rank), ("inverse", f_inverse)]
if len(sys.argv) > 4:
    keep = set(sys.argv[4].split(","))
    ARMS = [a for a in ARMS if a[0] in keep]

_BASE_INVQ = gf.invq_weight
_IW = {"v": None}


def _patched_invq(Q):
    """invq_weight is IMPORTED into glint_fast's namespace (line 16), so rebinding gf.invq_weight
    is what lines 175/417 actually look up. Q there is the full frame in input order, straight from
    the caller's q, so a per-peak vector aligned to q aligns to w."""
    w = _BASE_INVQ(Q)
    v = _IW["v"]
    if v is None:
        return w
    if len(v) != int(Q.shape[0]):                 # never silently weight the wrong peaks
        raise RuntimeError(f"intensity weight length {len(v)} != {int(Q.shape[0])} peaks")
    return w * torch.as_tensor(v, dtype=w.dtype, device=w.device)


gf.invq_weight = _patched_invq
_BLIND_FAST, _BLIND_NBEST = gf.index_blind_fast, gf.index_blind_nbest


def _mk(fn, iw_by_id):
    def wrapped(q, *a, **k):
        _IW["v"] = iw_by_id.get(id(q))
        try:
            return fn(q, *a, **k)
        finally:
            _IW["v"] = None
    return wrapped


def strict(M, q):
    if M is None or not same_lattice(M, LYSO):
        return False
    m = matched(M, q)
    return m / len(q) >= 0.25 and m >= 10


def mcnemar(a, b):
    """Exact two-sided binomial on the discordant pairs."""
    from math import comb
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum()); m = n01 + n10
    if m == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2.0 * sum(comb(m, j) for j in range(k + 1)) / 2.0 ** m)
    return n01, n10, p


def kish(v):
    """Kish effective sample size / N: 1.0 for uniform weights, lower as the variance grows.
    If reweighting hurts through variance rather than through intensity being anti-informative,
    the damage should track THIS and not the direction -- which is what the `inverse` arm is for."""
    v = np.asarray(v, float); s = v.sum()
    return float(s * s / (len(v) * (v * v).sum())) if s > 0 else 1.0


rows, gates, neff = [], {}, {}
for name, f in ARMS:
    # Per-frame weights, normalised to the control's total so only the SHAPE differs.
    iw = {}
    for q, I in zip(frames, inten):
        v = f(I) if len(I) else np.ones(len(q))
        v = np.where(np.isfinite(v), v, 0.0)
        s = v.sum()
        iw[id(q)] = (v * (len(v) / s)) if s > 0 else np.ones(len(q))
    neff[name] = float(np.mean([kish(iw[id(q)]) for q in frames]))
    gf.index_blind_fast = _mk(_BLIND_FAST, iw)
    gf.index_blind_nbest = _mk(_BLIND_NBEST, iw)
    hs.index_blind_nbest = gf.index_blind_nbest         # hybrid_stream imported it by name

    t0 = time.time()
    _IW["v"] = iw[id(frames[0])]; _BLIND_FAST(frames[0]); _IW["v"] = None      # warmup
    gb = np.array([strict(gf.index_blind_fast(q), q) for q in frames])
    res, st = hs.hybrid_index(frames, Mc_known=None, warmup=True)
    gh = np.array([strict(r["M"], q) for r, q in zip(res, frames)])
    gates[f"blind_{name}"] = gb; gates[f"hybrid_{name}"] = gh
    rows.append((name, int(gb.sum()), int(gh.sum()), int(gb[:120].sum()), int(gh[:120].sum())))
    print(f"{name:8s} blind {gb.sum():3d}/{n} ({100*gb.mean():4.1f}%)   "
          f"hybrid {gh.sum():3d}/{n} ({100*gh.mean():4.1f}%)   "
          f"| first-120 blind {gb[:120].sum():3d} hybrid {gh[:120].sum():3d}   "
          f"| n_eff/N {neff[name]:.3f}   | {time.time()-t0:5.0f}s", flush=True)

# The control is checked BEFORE anything is written. A harness that says "do not report these
# numbers" and then exits 0 having saved them is offering an unattended runner a result it has
# itself disowned -- and the first run of this sweep DID fail this control (the npz's q is full
# float32, the published text q-set is 6dp, and not one of the 480 frames is bit-identical).
_b = dict((r[0], r) for r in rows).get("binary")
if _b is not None:
    _ok = (_b[4] == 92)
    print(f"\nCONTROL binary first-120: blind {_b[3]} (expect 78, the STRICT bar -- the 84 in"
          f" glint_fast's comment is same_lattice only), hybrid {_b[4]} (expect 92)"
          f"  -> {'OK' if _ok else 'MISMATCH'}", flush=True)
    if not _ok:
        raise SystemExit(f"control failed: binary first-120 hybrid {_b[4]} != 92; this is not the "
                         f"published pipeline, so {OUT} was NOT written")
else:
    print("\n!! no binary arm -- there is no control and no baseline to compare against", flush=True)

np.savez(OUT, arms=np.array([r[0] for r in rows]), n=n,
         neff=np.array([neff[r[0]] for r in rows]), **gates)
print(f"\nwrote {OUT}")

if "binary" in dict((r[0], r) for r in rows):
    print("\n# McNemar vs the binary control (exact two-sided, discordant pairs)")
    for ep in ("blind", "hybrid"):
        base = gates[f"{ep}_binary"]
        for name, _ in ARMS:
            if name == "binary":
                continue
            n01, n10, p = mcnemar(base, gates[f"{ep}_{name}"])
            d = int(gates[f"{ep}_{name}"].sum()) - int(base.sum())
            print(f"  {ep:6s} {name:8s} {d:+4d} frames   gained {n01:3d} lost {n10:3d}   p = {p:.3f}"
                  f"{'' if p >= 0.05 else '   <- significant'}", flush=True)
    print("\n# Is the damage the DIRECTION of the weighting, or just its variance?")
    print("#   n_eff/N is Kish effective sample size: 1.0 = uniform. `inverse` weights DOWN what")
    print("#   the others weight UP, so if both hurt in proportion to n_eff, intensity is not")
    print("#   anti-informative -- spreading the weights is simply throwing peaks away.")
    for name, _ in ARMS:
        d = int(gates[f"blind_{name}"].sum()) - int(gates["blind_binary"].sum())
        print(f"  {name:8s} n_eff/N {neff[name]:.3f}   blind {d:+5d}")
    xs = np.array([neff[nm] for nm, _ in ARMS]); ys = np.array(
        [int(gates[f"blind_{nm}"].sum()) - int(gates["blind_binary"].sum()) for nm, _ in ARMS])
    if len(xs) > 2:
        print(f"  Pearson r(n_eff, blind delta) = {np.corrcoef(xs, ys)[0,1]:+.3f}")

