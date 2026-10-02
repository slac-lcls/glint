"""Self-contained SFX merge + half-set statistics from a GLINT .stream (no CrystFEL needed).

Parse (frame, h,k,l, I, sigma) rows, reduce hkl to the 4/mmm asymmetric unit, Monte-Carlo merge
(per-frame linear scale to the common mean, frames whose mean(I) is not measured dropped, then a
1/sigma^2-weighted mean per unique reflection), and
report CC1/2, CC*, Rsplit, redundancy, <I/sigma>, #unique -- the standard real-data figures of merit.
Same math as process_hkl (no partiality model). 4/mmm = ProK / lysozyme Laue group.

  python merge_stats.py [stream] [SYM=4/mmm]
"""
import os, sys, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glint.merge_scale import frame_scale      # noqa: E402  the live merge's per-frame scale + gate

stream = sys.argv[1] if len(sys.argv) > 1 else "/pscratch/sd/s/smarches/glint_real/prok_glint.stream"


def laue_ops_4mmm():
    gens = [np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),   # 4-fold c
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),  # 2-fold a
            -np.eye(3, dtype=int)]                          # inversion (Friedel)
    G = [np.eye(3, dtype=int)]
    ch = True
    while ch:
        ch = False
        for g in list(G):
            for s in gens:
                h = (s @ g).astype(int)
                if not any(np.array_equal(h, x) for x in G):
                    G.append(h); ch = True
    return G


OPS = laue_ops_4mmm()


def canon(hkl):
    eqs = np.stack([hkl @ op.T for op in OPS])               # (nops,n,3)
    key = eqs[:, :, 0] * 10**8 + eqs[:, :, 1] * 10**4 + eqs[:, :, 2]
    best = key.argmax(0)
    return eqs[best, np.arange(eqs.shape[1])]


# ---- parse stream ----
frames, H, I, S = [], [], [], []
fi = -1
inrefl = False
for line in open(stream):
    if line.startswith("----- Begin chunk"):
        fi += 1; inrefl = False
    elif line.startswith("Reflections measured"):
        inrefl = True
    elif line.startswith("End of reflections"):
        inrefl = False
    elif inrefl:
        p = line.split()
        if len(p) >= 5 and re.match(r"^-?\d+$", p[0]) and re.match(r"^-?\d+$", p[1]) and re.match(r"^-?\d+$", p[2]):
            try:
                h, k, l = int(p[0]), int(p[1]), int(p[2]); ii = float(p[3]); ss = float(p[4])
            except ValueError:
                continue
            frames.append(fi); H.append((h, k, l)); I.append(ii); S.append(ss)
frames = np.array(frames); H = np.array(H); I = np.array(I); S = np.maximum(np.array(S), 1e-3)
nf = frames.max() + 1
print(f"{len(I)} measurements over {nf} indexed frames ({len(I)//max(nf,1)}/frame); "
      f"<I/sig>={np.mean(I/S):.2f}")

# ---- per-frame linear scale to common mean (1 pass) ----
gmean = I[I > 0].mean()
# frames whose mean intensity is not measured are dropped, the live merge's rule (frame_scale,
# review r2 s1-01/s1-04); they used to stay in raw units with scale 1
scale = np.full(nf, np.nan)
for fr in range(nf):
    m = frames == fr
    fs = frame_scale(I[m])
    if fs is not None:
        scale[fr] = gmean * fs
print(f"{int(np.isnan(scale).sum())}/{nf} frames not merged (mean(I) not measured)")
_ok = np.isfinite(scale[frames])
frames, H, I, S = frames[_ok], H[_ok], I[_ok], S[_ok]
Is = I * scale[frames]

# ---- asu key ----
ch = canon(H)
key = ch[:, 0].astype(np.int64) * 10**8 + ch[:, 1] * 10**4 + ch[:, 2]


def merge(mask):
    k = key[mask]; v = Is[mask]; w = 1.0 / S[mask] ** 2
    order = np.argsort(k); k, v, w = k[order], v[order], w[order]
    uk, idx = np.unique(k, return_index=True)
    sw = np.add.reduceat(w, idx); swv = np.add.reduceat(w * v, idx)
    cnt = np.add.reduceat(np.ones_like(w), idx)
    return uk, swv / sw, cnt


uk, mI, cnt = merge(np.ones(len(I), bool))
print(f"unique reflections (4/mmm) = {len(uk)};  redundancy = {cnt.mean():.1f}")

# ---- CC1/2 via odd/even FRAME half-sets, swept over a per-measurement I/sigma floor ----
snr = I / S
odd = (frames % 2) == 1
print(f"\n{'I/sig floor':>12}{'#meas':>10}{'#common':>9}{'CC1/2':>8}{'CC*':>8}{'Rsplit%':>9}")
for thr in (-np.inf, 0.0, 1.0, 2.0, 3.0, 5.0):          # -inf: no floor, I <= 0 kept (review r2 s1-05)
    sel = snr > thr
    k1, v1, _ = merge(sel & odd); k2, v2, _ = merge(sel & ~odd)
    common, i1, i2 = np.intersect1d(k1, k2, return_indices=True)
    if len(common) < 10:
        print(f"  {thr:>10}{sel.sum():>10}{len(common):>9}   (too few)"); continue
    a, b = v1[i1], v2[i2]
    cc = np.corrcoef(a, b)[0, 1]
    ccs = np.sqrt(2 * cc / (1 + cc)) if cc > 0 else float("nan")
    rs = (1 / np.sqrt(2)) * np.sum(np.abs(a - b)) / (0.5 * np.sum(a + b))
    print(f"  {thr:>10}{sel.sum():>10}{len(common):>9}{cc:>8.3f}{ccs:>8.3f}{100*rs:>9.1f}")
