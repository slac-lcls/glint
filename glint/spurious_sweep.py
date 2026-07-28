"""Sweep the spurious-peak ceiling meter (glint.spurious_meter.null_margin) across a whole run.

Turns a single "N% indexed" number into a per-frame map of WHY the rest fail: a frame that misses can be
*blank* (no crystal -- nothing to index) or *wall-limited* (a real crystal whose true orientation the
spurious peaks bury below the random-orientation floor). Only the second kind is a ceiling you can fight
(consensus, better peak selection); the first is just empty exposures. The sweep classifies every frame,
histograms the separation z, and reports the wall-rate.

Core `sweep(frames_q, M, ...)` is data-format-agnostic (list of (N,3) reciprocal-peak arrays + a cell).
The CLI loads frames from an .npz (key `frames`, object array) or straight from a .cxi via
glint.lute_bridge.frames_from_cxi, and the cell from a 3x3 .npy.

    python -m glint.spurious_sweep --cxi run.cxi --geom det.geom --cell cellA.npy --search-size 5e6
    python -m glint.spurious_sweep --npz frames.npz --cell cellA.npy
"""
import argparse
import numpy as np

from .spurious_meter import null_margin, WALL_Z


def classify(rec, min_peaks):
    """blank | wall | clear from a null_margin record."""
    if rec["n"] < min_peaks or rec["s_true"] <= rec["null_mean"] + rec["null_std"]:
        return "blank"                                  # no signal above the random-orientation noise
    walled = rec["wall"] if rec["wall"] is not None else rec["wall_z"]
    return "wall" if walled else "clear"


def sweep(frames_q, M, K=None, n_null=128, tol=0.15, min_peaks=6, seed=0):
    """Run the ceiling meter over every frame. Returns dict(per_frame=[...], z=[...], counts={...},
    wall_rate, median_z). wall_rate is over frames that are NOT blank (the frames that could index)."""
    rng = np.random.default_rng(seed)
    M = np.asarray(M, float)
    per, zs = [], []
    counts = {"blank": 0, "wall": 0, "clear": 0}
    for i, q in enumerate(frames_q):
        q = np.asarray(q, float)
        r = null_margin(q, M, tol=tol, n_null=n_null, K=K, rng=rng)
        cls = classify(r, min_peaks)
        counts[cls] += 1
        per.append(dict(frame=i, n=r["n"], s_true=r["s_true"], z=r["z"],
                        null_mean=r["null_mean"], evt_floor=r["evt_floor"], cls=cls))
        if cls != "blank":
            zs.append(r["z"])
    indexable = counts["wall"] + counts["clear"]
    return dict(per_frame=per, z=zs, counts=counts,
                wall_rate=(counts["wall"] / indexable if indexable else 0.0),
                median_z=(float(np.median(zs)) if zs else 0.0),
                n_frames=len(per))


def _histogram(zs, bins=(0, 1, 2, 3, 4, 6, 8, 12, 1e9), width=44):
    """ASCII histogram of z over the given edges."""
    zs = np.asarray(zs, float)
    if zs.size == 0:
        return "  (no non-blank frames)"
    edges = np.array(bins, float)
    idx = np.clip(np.digitize(zs, edges) - 1, 0, len(edges) - 2)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    top = max(counts.max(), 1)
    lines = []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        lab = f"z {lo:>4.0f}-{'inf' if hi > 1e8 else f'{hi:<4.0f}'}"
        bar = "#" * int(round(width * counts[b] / top))
        lines.append(f"  {lab} | {bar} {counts[b]}")
    return "\n".join(lines)


def format_sweep(res, wall_z=WALL_Z):
    c = res["counts"]
    n = res["n_frames"]
    pct = lambda k: f"{100 * c[k] / n:.0f}%" if n else "0%"
    out = [f"frames: {n}",
           f"  clear (indexable) : {c['clear']:>5}  ({pct('clear')})",
           f"  wall  (spurious)  : {c['wall']:>5}  ({pct('wall')})",
           f"  blank (no crystal): {c['blank']:>5}  ({pct('blank')})",
           f"  wall-rate (of non-blank): {100 * res['wall_rate']:.0f}%    median z (non-blank): {res['median_z']:.1f}",
           "z separation of true fit above random-orientation floor:",
           _histogram(res["z"])]
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------- CLI
def _load_frames(a):
    if a.npz:
        d = np.load(a.npz, allow_pickle=True)
        return list(d["frames"])
    if a.cxi:
        from .lute_bridge import frames_from_cxi
        frames, _ = frames_from_cxi(a.cxi, a.geom, wavelength_A=a.wavelength, n=a.n,
                                    peakfinder=a.peakfinder, top_n=a.top_n)
        return frames
    raise SystemExit("provide --npz or (--cxi and --geom)")


def main(argv=None):
    p = argparse.ArgumentParser(description="Sweep the blind-indexing spurious ceiling across a run.")
    src = p.add_argument_group("frame source")
    src.add_argument("--npz", help=".npz with object array `frames` = list of (N,3) q-vectors")
    src.add_argument("--cxi", help=".cxi or CrystFEL .list to peak-find/bridge to q")
    src.add_argument("--geom", help="CrystFEL .geom (with --cxi)")
    src.add_argument("--wavelength", type=float, default=None, help="angstrom (else from geom photon_energy)")
    src.add_argument("--peakfinder", default="stored", help="v4 | pf9 | stored (default)")
    src.add_argument("--top-n", type=int, default=0, dest="top_n")
    src.add_argument("--n", type=int, default=0, help="cap number of frames")
    p.add_argument("--cell", required=True, help="3x3 .npy real cell (columns a,b,c); hkl = q @ M")
    p.add_argument("--search-size", type=float, default=None, dest="K",
                   help="blind search size K -> enables the extreme-value wall flag (e.g. 5e6)")
    p.add_argument("--n-null", type=int, default=128, dest="n_null")
    p.add_argument("--tol", type=float, default=0.15)
    p.add_argument("--min-peaks", type=int, default=6, dest="min_peaks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", help="optional .npz to write the per-frame table to")
    a = p.parse_args(argv)

    frames = _load_frames(a)
    M = np.load(a.cell)
    res = sweep(frames, M, K=(int(a.K) if a.K else None), n_null=a.n_null, tol=a.tol,
                min_peaks=a.min_peaks, seed=a.seed)
    print(format_sweep(res))
    if a.save:
        pf = res["per_frame"]
        np.savez(a.save, frame=[r["frame"] for r in pf], n=[r["n"] for r in pf],
                 s_true=[r["s_true"] for r in pf], z=[r["z"] for r in pf],
                 cls=[r["cls"] for r in pf])
        print(f"\nwrote {a.save}")
    return res


if __name__ == "__main__":
    main()
