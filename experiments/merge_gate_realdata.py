"""Live-merge CC1/2 on real integrated crystals under the per-frame rules compared for review r2 s1-01/s1-04/s1-05.

Reads a CrystFEL-format .stream (plain or .gz) of integrated crystals, puts each crystal's unique axis last (4/mmm,
from its "Cell parameters" line), and folds the crystals into MergeAccumulator in 10 random orders (half-set =
frame parity, so the order is the half-split):

  old           v = I/mean(I) (scale 1 if mean <= 0 or n <= 5), w = 1/sigma^2, I > 0 only   (main before the fix)
  gate K        frames with mean(I) <= K sem(I) or n <= 5 not merged, w = 1/sigma^2
  gate K + ivw  same gate, w = mean(I)^2/sigma^2 = 1/(sigma*scale)^2 (inverse variance of v)
  ... keep I<=0 the gate-3 rule with the -inf bin (the shipped default)

Measured 1 Oct 2026 (10 half-splits, mean +- sd):
  cxidb-17 lysozyme, 793 crystals (glint_full_int.stream):  old 0.015 +- 0.007; gate 3: 0.332 +- 0.033;
      gate 3 + ivw 0.305 +- 0.025; gate 3 keep I<=0: 0.346 +- 0.055 (212 frames refused)
  mfx100848724 r51, 303 crystals (LUTE round 7):  old 0.044 +- 0.013; gate 3: 0.071 +- 0.020; gate 3 + ivw 0.059

  PYTHONPATH=. python experiments/merge_gate_realdata.py STREAM[.gz]
"""
import gzip
import sys

import numpy as np

sys.path.insert(0, ".")
from glint.stream_driver import MergeAccumulator, laue_ops_4mmm  # noqa: E402


def read_stream(path):
    op = gzip.open if path.endswith(".gz") else open
    crys, cell, rows = [], None, None
    with op(path, "rt") as fh:
        for ln in fh:
            if ln.startswith("Cell parameters"):
                p = ln.split(); cell = np.array([float(p[2]), float(p[3]), float(p[4])])
            elif ln.startswith("Reflections measured after indexing"):
                rows = []; next(fh)
            elif ln.startswith("End of reflections"):
                a = np.array(rows, float).reshape(-1, 5)
                if len(a) and cell is not None:
                    perm = np.argsort(-cell)              # long, long, short: the 4-fold axis last
                    crys.append((a[:, :3][:, perm].astype(int), a[:, 3], np.maximum(a[:, 4], 1e-3)))
                rows = None
            elif rows is not None:
                try:
                    rows.append([float(x) for x in ln.split()[:5]])
                except ValueError:
                    pass
    return crys


def merge(frames, rule, K=3.0, keep_neg=False):
    bins = (-np.inf, 0.0, 1.0, 2.0, 3.0, 5.0) if keep_neg else (0.0, 1.0, 2.0, 3.0, 5.0)
    acc = MergeAccumulator(snr_bins=bins, ops=laue_ops_4mmm()); refused = 0
    for fi, (hkl, I, s) in enumerate(frames):
        n = I.size; m = I.mean()
        if rule == "old":
            scale = 1.0 / m if (n > 5 and m > 0) else 1.0
            acc.add_frame(hkl, I, s, fi, values=I * scale, weights=1 / s ** 2); continue
        if not (n > 5 and m > K * I.std() / np.sqrt(n)):
            refused += 1; continue
        acc.add_frame(hkl, I, s, fi, values=I / m, weights=(m / s) ** 2 if rule == "ivw" else 1 / s ** 2)
    return acc.stats(-np.inf if keep_neg else 0.0), refused


if __name__ == "__main__":
    crys = read_stream(sys.argv[1])
    print(f"{sys.argv[1]}: {len(crys)} crystals, {sum(len(c[1]) for c in crys)} measurements")
    orders = [np.random.default_rng(s).permutation(len(crys)) for s in range(10)]
    arms = [("old", "old", 0, False)] + [(f"gate {K:g}", "gate", K, False) for K in (0, 1, 2, 3)] + \
           [(f"gate {K:g} + ivw", "ivw", K, False) for K in (0, 1, 2, 3)] + [("gate 3, keep I<=0", "gate", 3, True)]
    print(f"{'rule':20s}  CC1/2 mean +- sd    Rsplit  redund  refused")
    for name, rule, K, kn in arms:
        r = [merge([crys[i] for i in o], rule, K, kn) for o in orders]
        cc = np.array([x[0]["cc_half"] for x in r])
        print(f"{name:20s}  {cc.mean():.3f} +- {cc.std():.3f}   {np.mean([x[0]['rsplit'] for x in r]):.3f}"
              f"   {r[0][0]['redundancy']:5.1f}   {r[0][1]}")
