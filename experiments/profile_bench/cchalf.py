"""Merge-free CC1/2 against resolution for each arm, from the integrated streams.

Frames go to two random halves (seeded, the same split for every arm). Each frame is scaled by
1/mean(I) with GLINT's rule (glint.merge_scale.frame_scale: frames whose mean intensity is not
measured are dropped), then each half is merged per unique ASU reflection (Friedel mates together,
or apart with --anomalous) as the unweighted mean of the scaled intensities. No partiality model, so
the absolute values are pessimistic next to a partialator or abismal merge; the arms go through the
same steps, so the difference between them is the integrators'.

    python cchalf.py --work WORK --integrators box,profile --runs r1 r2 ... -o cchalf.json [--shells 20]
"""
import argparse
import json

import gemmi
import numpy as np
import reciprocalspaceship as rs

from glint.merge_scale import frame_scale
from report import read_stream, _hkl_key


def half_means(I, key, half):
    """Per (key, half): mean of I and count. Returns keys, mean[2, n], count[2, n]."""
    k = key * 2 + half
    u, inv = np.unique(k, return_inverse=True)
    s = np.bincount(inv, weights=I)
    c = np.bincount(inv)
    keys = np.unique(u // 2)
    j = np.searchsorted(keys, u // 2)
    m = np.full((2, len(keys)), np.nan)
    n = np.zeros((2, len(keys)), int)
    m[u % 2, j] = s / c
    n[u % 2, j] = c
    return keys, m, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--integrators", default="box,profile")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cell", nargs=6, type=float, default=[79.2, 79.2, 37.9, 90, 90, 90])
    ap.add_argument("--sg", default="P 43 21 2")
    ap.add_argument("--dmin", type=float, default=1.9)
    ap.add_argument("--shells", type=int, default=20)
    ap.add_argument("--anomalous", action="store_true", help="keep Friedel mates apart")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    cell, sg = gemmi.UnitCell(*a.cell), gemmi.SpaceGroup(a.sg)
    out = {"anomalous": a.anomalous, "integrators": {}}
    edges = None
    for integ in a.integrators.split(","):
        keys, Is, halves = [], [], []
        n_frames = n_dropped = 0
        for ri, r in enumerate(a.runs):
            x = read_stream(f"{a.work}/streams/{integ}/{r}.stream")
            if not len(x):
                continue
            chunk = x[:, 0].astype(np.int64)
            I = x[:, 4].astype(np.float64)
            # the same split for every arm: a hash of (run, chunk), not of row order
            fid = ri * 1_000_000 + chunk
            half = ((fid * 2654435761 + a.seed) >> 7) & 1
            scale = np.zeros(len(x))
            for c in np.unique(chunk):
                m = chunk == c
                s = frame_scale(I[m])
                n_frames += 1
                if s is None:
                    n_dropped += 1
                else:
                    scale[m] = s
            ok = scale > 0
            H, isym = rs.utils.hkl_to_asu(x[ok, 1:4].astype(int), sg)
            if a.anomalous:                              # odd isym is the Friedel minus
                H = np.where((isym % 2 == 0)[:, None], H, -H)
            keys.append(_hkl_key(H))
            Is.append(I[ok] * scale[ok])
            halves.append(half[ok])
        key, I, half = np.concatenate(keys), np.concatenate(Is), np.concatenate(halves)
        uk, m, n = half_means(I, key, half)
        both = np.isfinite(m).all(0)
        uk, m, n = uk[both], m[:, both], n[:, both]
        H = np.stack([uk // 1048576 - 512, uk // 1024 % 1024 - 512, uk % 1024 - 512], 1)
        d = cell.calculate_d_array(np.abs(H) if a.anomalous else H)
        if edges is None:                                # equal-count shells, shared by the arms
            dd = d[d >= a.dmin]
            edges = np.quantile(1 / dd ** 2, np.linspace(0, 1, a.shells + 1))
        b = np.clip(np.searchsorted(edges, 1 / d ** 2, side="right") - 1, 0, a.shells - 1)
        keep = d >= a.dmin
        cc = [float(np.corrcoef(m[0, keep & (b == s)], m[1, keep & (b == s)])[0, 1]) for s in range(a.shells)]
        mult = [float(np.median(n[:, keep & (b == s)].sum(0))) for s in range(a.shells)]
        out["integrators"][integ] = {
            "cchalf": cc, "median_multiplicity": mult, "n_frames": n_frames,
            "n_frames_dropped": n_dropped, "n_unique": int(keep.sum()),
            "cchalf_overall": float(np.corrcoef(m[0, keep], m[1, keep])[0, 1]),
        }
        print(integ, " ".join(f"{c:.3f}" for c in cc), flush=True)
    out["d_mid"] = [float(1 / np.sqrt((edges[s] + edges[s + 1]) / 2)) for s in range(a.shells)]
    out["d_edges"] = [float(1 / np.sqrt(e)) for e in edges]
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
