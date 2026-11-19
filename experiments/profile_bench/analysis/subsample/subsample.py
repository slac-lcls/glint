"""CC1/2 and CCanom against the number of merged images, from random subsamples merged with partialator.

For one dataset: index every arm's per-run streams (byte ranges of the chunks that hold a crystal), draw
REPS random subsets of SIZE lattices (the same lattices for every arm), write each subset as a stream,
merge it with the pipeline's partialator settings, and compute from the half-datasets:
  CC1/2 (Pearson) and CCanom (inverse-variance-weighted Pearson, and Spearman) per fixed shell, to the
  common cut, and in the outer shell. The full set is the existing workflow merge (one draw).
Lattice k of a run is the k-th chunk with a crystal; the arms pair on (run, k), as report.py does.

    python subsample.py --work W --cell a,b,c,al,be,ga --dmin 1.06 --nshell 14 --outer 1.4014,1.353 \
        --sizes 500,1000,... --reps 5 --workers 12 --threads 8 --out OUTDIR
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

import numpy as np

ARMS = ("box", "profile", "crystfel")
CRYSTFEL = "/pscratch/sd/k/kmdalton/glint/.mamba/envs/crystfel/bin/partialator"


def index_stream(path):
    """(header bytes, [(start, end) of each chunk holding a crystal])."""
    spans, hend = [], None
    off = start = 0
    has = False
    with open(path, "rb") as f:
        for ln in f:
            if ln.startswith(b"----- Begin chunk -----"):
                if hend is None:
                    hend = off
                start, has = off, False
            elif ln.startswith(b"--- Begin crystal"):
                has = True
            elif ln.startswith(b"----- End chunk -----"):
                if has:
                    spans.append((start, off + len(ln)))
            off += len(ln)
    with open(path, "rb") as f:
        header = f.read(hend or 0)
    return header, spans


def write_subset(arm_files, header, spans, ids, out):
    """ids: sorted (run_index, k). Copies those chunks' bytes into a new stream."""
    with open(out, "wb") as o:
        o.write(header)
        cur, fh = None, None
        for r, k in ids:
            if r != cur:
                if fh:
                    fh.close()
                fh, cur = open(arm_files[r], "rb"), r
            s, e = spans[r][k]
            fh.seek(s)
            o.write(fh.read(e - s))
        if fh:
            fh.close()


def stats(hkldir, cell, dmin, nshell, outer, cut):
    import warnings
    warnings.filterwarnings("ignore")
    import gemmi
    import reciprocalspaceship as rs
    from scipy.stats import spearmanr
    sys.path.insert(0, "/pscratch/sd/k/kmdalton/glint/experiments/profile_bench")
    from hkl_to_mtz import read_hkl as _read_hkl, reindex as _reindex
    import os
    def read_hkl(p):
        """The merge in the model's setting: REINDEX (env, default h,k,l) maps GLINT's indices to SG's."""
        x = _read_hkl(p).astype(float)
        x[:, :3] = _reindex(x[:, :3], os.environ.get("REINDEX", "h,k,l"))
        return x
    uc, sg = gemmi.UnitCell(*cell), gemmi.SpaceGroup(os.environ.get("SG", "P 43 21 2"))

    def load(p):
        x = read_hkl(p)
        d = rs.DataSet({"H": x[:, 0].astype(int), "K": x[:, 1].astype(int), "L": x[:, 2].astype(int),
                        "I": x[:, 3], "S": x[:, 4]}, cell=uc, spacegroup=sg).infer_mtz_dtypes().set_index(["H", "K", "L"])
        d = d[d.S > 0].hkl_to_asu(anomalous=True).remove_absences()
        d.merged = True
        return d.drop(columns=["M/ISYM"], errors="ignore")

    def anom(d):
        u = d.unstack_anomalous(["I", "S"]).dropna()
        return rs.DataSet({"D": u["I(+)"] - u["I(-)"], "V": u["S(+)"] ** 2 + u["S(-)"] ** 2}, cell=uc, spacegroup=sg)

    a, b = load(f"{hkldir}/merged.hkl1"), load(f"{hkldir}/merged.hkl2")
    j = a[["I"]].join(b[["I"]], rsuffix="2", how="inner").dropna().compute_dHKL()
    jj = anom(a).join(anom(b), rsuffix="2", how="inner").dropna()
    jj.cell, jj.spacegroup = uc, sg
    jj = jj.compute_dHKL()
    hi_ = np.linspace(30.0 ** -3, dmin ** -3, nshell + 1) ** (-1 / 3)
    sel = [(h, l) for h, l in zip(hi_[:-1], hi_[1:])] + [(30.0, cut), tuple(outer)]

    def cc(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else None

    def wcc(x, y, w):
        if len(x) < 3:
            return None
        w = w / w.sum()
        mx, my = (w * x).sum(), (w * y).sum()
        return float((w * (x - mx) * (y - my)).sum() / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum()))

    def rcc(x, y):
        return float(spearmanr(x, y)[0]) if len(x) > 2 else None

    out = {"cchalf": [], "ccano_w": [], "ccano_rank": [], "n": [], "n_ano": []}
    d1, I1, I2 = j.dHKL.to_numpy(float), j.I.to_numpy(float), j.I2.to_numpy(float)
    d2, D1, D2 = jj.dHKL.to_numpy(float), jj.D.to_numpy(float), jj.D2.to_numpy(float)
    W = 1 / (jj.V.to_numpy(float) + jj.V2.to_numpy(float))
    for h, l in sel:
        s, t = (d1 < h) & (d1 >= l), (d2 < h) & (d2 >= l)
        out["cchalf"].append(cc(I1[s], I2[s]))
        out["ccano_w"].append(wcc(D1[t], D2[t], W[t]))
        out["ccano_rank"].append(rcc(D1[t], D2[t]))
        out["n"].append(int(s.sum()))
        out["n_ano"].append(int(t.sum()))
    out["edges"] = [round(float(e), 4) for e in hi_]
    return out  # entries 0..nshell-1: fixed shells; nshell: overall to cut; nshell+1: outer shell


def task(arm, size, rep, ids, arm_files, header, spans, a):
    tag = f"{arm}_n{size}_r{rep}"
    od = f"{a.out}/merges/{tag}"
    res = f"{a.out}/results/{tag}.json"
    if os.path.exists(res):
        return tag, "cached"
    os.makedirs(od, exist_ok=True)
    t0 = time.time()
    if not os.path.exists(f"{od}/merged.hkl2"):
        st = f"{a.tmp}/{tag}.stream"
        write_subset(arm_files, header, spans, ids, st)
        with open(f"{od}/partialator.log", "w") as lg:
            rc = subprocess.run([CRYSTFEL, "-i", st, "-o", f"{od}/merged.hkl", "-y", os.environ.get("PG", "422"), "--model=unity",
                                 "--iterations=1", "--no-logs", "-j", str(a.threads)], cwd=od, stdout=lg,
                                stderr=subprocess.STDOUT).returncode
        os.remove(st)
        if rc:
            return tag, f"partialator rc {rc}"
    s = stats(od, a.cell, a.dmin, a.nshell, a.outer, a.cut)
    s.update({"arm": arm, "size": size, "rep": rep, "seconds": round(time.time() - t0, 1)})
    json.dump(s, open(res, "w"))
    return tag, f"{s['seconds']} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--dmin", type=float, required=True)
    ap.add_argument("--nshell", type=int, required=True)
    ap.add_argument("--outer", required=True)
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    a = ap.parse_args()
    a.cell = [float(x) for x in a.cell.split(",")]
    a.outer = [float(x) for x in a.outer.split(",")]
    a.cut = json.load(open(f"{a.work}/partialator/merge/cut.json"))["common"]
    os.makedirs(f"{a.out}/results", exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)

    runs = sorted(os.path.basename(p)[:-7] for p in os.listdir(f"{a.work}/streams/box") if p.endswith(".stream"))
    files = {arm: [f"{a.work}/streams/{arm}/{r}.stream" for r in runs] for arm in ARMS}
    t0 = time.time()
    with cf.ProcessPoolExecutor(min(len(runs) * 3, 48)) as ex:
        idx = {arm: list(ex.map(index_stream, files[arm])) for arm in ARMS}
    headers = {arm: idx[arm][0][0] for arm in ARMS}
    spans = {arm: [s for _, s in idx[arm]] for arm in ARMS}
    counts = [len(s) for s in spans["box"]]
    for arm in ARMS:
        bad = [r for r, s, c in zip(runs, spans[arm], counts) if len(s) != c]
        if bad:
            sys.exit(f"{arm}: lattice counts differ from box in {len(bad)} runs, e.g. {bad[0]}")
    lattices = [(r, k) for r, c in enumerate(counts) for k in range(c)]
    N = len(lattices)
    print(f"{len(runs)} runs, {N} lattices, indexed in {time.time() - t0:.0f} s", flush=True)

    jobs = []
    for size in [int(s) for s in a.sizes.split(",")]:
        for rep in range(a.reps):
            rng = np.random.default_rng(1000 * size + rep)
            pick = sorted(lattices[i] for i in rng.choice(N, size, replace=False))
            for arm in ARMS:
                jobs.append((arm, size, rep, pick))
    jobs.sort(key=lambda j: -j[1])            # largest first, so the long merges start early
    with cf.ProcessPoolExecutor(a.workers) as ex:
        fut = {ex.submit(task, arm, size, rep, pick, files[arm], headers[arm], spans[arm], a): (arm, size, rep)
               for arm, size, rep, pick in jobs}
        # the full set: the workflow's own merges, statistics only
        for arm in ARMS:
            res = f"{a.out}/results/{arm}_n{N}_r0.json"
            if not os.path.exists(res):
                s = stats(f"{a.work}/partialator/merge/{arm}", a.cell, a.dmin, a.nshell, a.outer, a.cut)
                s.update({"arm": arm, "size": N, "rep": 0, "seconds": 0})
                json.dump(s, open(res, "w"))
        for f in cf.as_completed(fut):
            tag, msg = f.result()
            print(f"{time.strftime('%H:%M:%S')} {tag}: {msg}", flush=True)

    rows = [json.load(open(f"{a.out}/results/{p}")) for p in sorted(os.listdir(f"{a.out}/results"))]
    json.dump({"n_lattices": N, "cut": a.cut, "outer": a.outer, "rows": rows}, open(f"{a.out}/all.json", "w"))
    k = a.nshell
    print("arm size  CC1/2(cut) outer | CCano_w(cut) outer | CCano_rank(cut)  (mean over reps)")
    for arm in ARMS:
        for size in sorted({r["size"] for r in rows}):
            rr = [r for r in rows if r["arm"] == arm and r["size"] == size]
            m = lambda key, i: np.mean([r[key][i] for r in rr if r[key][i] is not None])
            print(f"{arm:8s} {size:6d} {m('cchalf', k):.3f} {m('cchalf', k + 1):.3f} | "
                  f"{m('ccano_w', k):.3f} {m('ccano_w', k + 1):.3f} | {m('ccano_rank', k):.3f}  n={len(rr)}")


if __name__ == "__main__":
    main()
