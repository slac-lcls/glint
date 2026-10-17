#!/usr/bin/env python
"""Helpers (numpy only). Nothing here indexes, integrates or merges: CrystFEL 0.10.2 does that in j13_merge_one.sh.

  j13_tools.py frames  STREAM                              n chunks, frames with a crystal, crystals, multi-crystal frames
  j13_tools.py subset  STREAM FRAC SEED PERM DST           DST (a stream) + DST.json
        SEED 0      every frame with a crystal, in stream order (FRAC must be 1.00)
        SEED >= 1   the first round(FRAC*N) frames of default_rng(SEED).permutation(N), in that order. Nested: within a
                    seed the 10% subset is inside the 25% subset, and so on, as if one run were stopped early.
                    The order also sets the half-split (partialator and process_hkl split alternate crystals).
        PERM none   the frames as they are
             all    null: in every crystal, the (I, sigma(I)) pairs permuted over all its reflections
             shell  null: the same, within blocks of 20 reflections that are neighbours in 1/d (keeps the Wilson falloff)
                    permutation rng = default_rng(1_000_003 + SEED); hkl, positions, peak, background are not touched
  j13_tools.py range   MDIR                                "dlow dhigh" (A) from MDIR/auto/compare_CC.log
  j13_tools.py record  MDIR TAG FRAC SEED PERM MERGER      MDIR/record.json from the logs in MDIR
  j13_tools.py collect ROOT                                ROOT/results.json, ROOT/sweep.tsv, ROOT/summary.md

read_stream() follows ../merges_tofile_vs_crystfel/j2_tools.py; it is copied so that this check does not depend on that folder.
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np

BEGIN_CHUNK, END_CHUNK = "----- Begin chunk", "----- End chunk"
CC_FOR_CCSTAR_09 = 0.81 / 1.19          # CC* = sqrt(2 CC / (1 + CC)) = 0.9  <=>  CC1/2 = 0.6807
CC_FOR_CCSTAR_05 = 1.0 / 7.0            # CC* = 0.5  <=>  CC1/2 = 0.143 (the usual resolution criterion)


# ----------------------------------------------------------------------------------------------------------------
# streams
def read_stream(path):
    """(header_lines, [(key, chunk_lines, n_crystals)]); key = '<image> <event>' as CrystFEL writes them."""
    header, chunks, cur = [], [], None
    with open(path, errors="replace") as fh:
        for ln in fh:
            if cur is None:
                if ln.startswith(BEGIN_CHUNK):
                    cur = [ln]
                else:
                    header.append(ln)
                continue
            cur.append(ln)
            if ln.startswith(END_CHUNK):
                img = ev = ""
                ncr = 0
                for c in cur:
                    if c.startswith("Image filename:"):
                        img = c.split(":", 1)[1].strip()
                    elif c.startswith("Event:"):
                        ev = c.split(":", 1)[1].strip()
                    elif c.startswith("--- Begin crystal"):
                        ncr += 1
                chunks.append((f"{img} {ev}".strip(), cur, ncr))
                cur = None
    return header, chunks


def indexed_frames(chunks):
    """Frames with at least one crystal, first occurrence of each key, in stream order."""
    seen, out = set(), []
    for k, lines, n in chunks:
        if n > 0 and k not in seen:
            seen.add(k)
            out.append((k, lines, n))
    return out


def cmd_frames(stream):
    _, chunks = read_stream(stream)
    fr = indexed_frames(chunks)
    rec = {"stream": stream, "chunks": len(chunks), "frames_with_crystal": len(fr),
           "crystals": int(sum(n for _, _, n in fr)), "frames_multi": int(sum(1 for _, _, n in fr if n > 1)),
           "duplicate_keys": len([1 for k, _, n in chunks if n > 0]) - len(fr)}
    print(json.dumps(rec))


def _vec(ln):
    return np.array([float(x) for x in ln.split("=", 1)[1].split()[:3]])


def permute_crystal_rows(rows, mode, rng, astar, bstar, cstar):
    """rows: reflection lines of one crystal. Returns the lines with the (I, sigma) strings permuted."""
    parts = [r.split(None, 5) for r in rows]
    ok = [i for i, p in enumerate(parts) if len(p) == 6]
    if len(ok) < 2:
        return rows
    m = len(ok)
    if mode == "all":
        perm = rng.permutation(m)
    elif mode == "shell":
        hkl = np.array([[int(parts[i][0]), int(parts[i][1]), int(parts[i][2])] for i in ok], dtype=float)
        q = np.linalg.norm(hkl @ np.vstack([astar, bstar, cstar]), axis=1)
        order = np.argsort(q, kind="stable")
        perm = np.arange(m)
        for b in range(0, m, 20):
            blk = order[b:b + 20]
            perm[blk] = blk[rng.permutation(len(blk))]
    else:
        raise ValueError(mode)
    out = list(rows)
    for j, i in enumerate(ok):
        src = parts[ok[perm[j]]]
        p = parts[i]
        out[i] = f"{p[0]:>4} {p[1]:>4} {p[2]:>4} {src[3]:>10} {src[4]:>10} {p[5]}"
    return out


def permute_chunk(lines, mode, rng):
    out, i, n = [], 0, len(lines)
    astar = bstar = cstar = None
    while i < n:
        ln = lines[i]
        if ln.startswith("astar ="):
            astar = _vec(ln)
        elif ln.startswith("bstar ="):
            bstar = _vec(ln)
        elif ln.startswith("cstar ="):
            cstar = _vec(ln)
        if ln.startswith("Reflections measured after indexing"):
            out.append(ln)
            out.append(lines[i + 1])                       # column header
            j = i + 2
            while j < n and not lines[j].startswith("End of reflections"):
                j += 1
            out.extend(permute_crystal_rows(lines[i + 2:j], mode, rng, astar, bstar, cstar))
            i = j
            continue
        out.append(ln)
        i += 1
    return out


CELL_RE = re.compile(r"^Cell parameters\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+nm,\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+deg")


def cell_stats(frames):
    cells = []
    for _, lines, _ in frames:
        for ln in lines:
            m = CELL_RE.match(ln)
            if m:
                v = [float(x) for x in m.groups()]
                cells.append([10 * v[0], 10 * v[1], 10 * v[2], v[3], v[4], v[5]])
    if not cells:
        return {}
    c = np.array(cells)
    med = np.median(c, axis=0)
    q1, q3 = np.percentile(c, 25, axis=0), np.percentile(c, 75, axis=0)
    names = ["a", "b", "c", "al", "be", "ga"]
    return {"n": len(cells), "median": dict(zip(names, np.round(med, 4).tolist())),
            "iqr": dict(zip(names, np.round(q3 - q1, 4).tolist()))}


def cmd_subset(stream, frac, seed, perm, dst):
    frac, seed = float(frac), int(seed)
    header, chunks = read_stream(stream)
    fr = indexed_frames(chunks)
    N = len(fr)
    if seed == 0:
        if abs(frac - 1.0) > 1e-9:
            sys.exit("seed 0 is the stream order and is used at FRAC 1.00 only")
        sel = list(range(N))
    else:
        k = max(2, int(round(frac * N)))
        sel = np.random.default_rng(seed).permutation(N)[:k].tolist()
    prng = np.random.default_rng(1_000_003 + seed)
    picked = [fr[i] for i in sel]
    with open(dst, "w") as f:
        f.writelines(header)
        for _, lines, _ in picked:
            f.writelines(lines if perm == "none" else permute_chunk(lines, perm, prng))
    rec = {"stream": stream, "frac": frac, "seed": seed, "perm": perm, "frames_available": N,
           "frames": len(picked), "crystals": int(sum(n for _, _, n in picked)), "cell": cell_stats(picked)}
    json.dump(rec, open(dst + ".json", "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("frames_available", "frames", "crystals", "perm", "seed")}))


# ----------------------------------------------------------------------------------------------------------------
# log parsing
def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def parse_compare(path):
    if not os.path.exists(path):
        return {}
    t = open(path, errors="replace").read()
    r = {}
    m = re.search(r"Overall (CC\*|CC|Rsplit) = ([-\d.eE+naninf]+)", t)
    if m:
        r["value"] = _f(m.group(2))
    m = re.search(r"(\d+) reflection pairs accepted", t)
    if m:
        r["pairs"] = int(m.group(1))
    m = re.search(r"Accepted resolution range: .*?\(([\d.]+) to ([\d.]+) Angstroms\)", t)
    if m:
        r["d_range_A"] = [float(m.group(1)), float(m.group(2))]
    return r


def parse_check(path):
    if not os.path.exists(path):
        return {}
    t = open(path, errors="replace").read()
    r = {}
    for pat, key, cast in ((r"Overall <snr> = ([-\d.eE+naninf]+)", "mean_I_over_sigma", _f),
                           (r"Overall redundancy = ([-\d.eE+naninf]+)", "multiplicity", _f),
                           (r"Overall completeness = ([-\d.eE+naninf]+)", "completeness_pct", _f),
                           (r"(\d+) measurements in total", "measurements", int),
                           (r"(\d+) reflections in total", "unique", int),
                           (r"(\d+) reflections possible", "possible", int)):
        m = re.search(pat, t)
        if m:
            r[key] = cast(m.group(1))
    return r


def parse_shells(path):
    """compare_hkl shell file -> [(d_hi_A, value, nref)], low -> high resolution."""
    rows = []
    if not os.path.exists(path):
        return rows
    for ln in open(path):
        p = ln.split()
        if len(p) < 6 or not re.match(r"^[\d.]+$", p[0]):
            continue
        rows.append((round(10.0 / float(p[5]), 3), _f(p[1]), int(float(p[2]))))
    return rows


def parse_check_shells(path):
    """check_hkl shell file -> [(d_hi_A, completeness_pct, multiplicity, snr)]."""
    rows = []
    if not os.path.exists(path):
        return rows
    for ln in open(path):
        p = ln.split()
        if len(p) < 11 or not re.match(r"^[\d.]+$", p[0]):
            continue
        rows.append((round(10.0 / float(p[10]), 3), _f(p[3]), _f(p[5]), _f(p[6])))
    return rows


def d_at_cc(shells, thresh=CC_FOR_CCSTAR_05):
    """High-resolution edge of the last shell, from low resolution, before CC1/2 first drops below `thresh`."""
    last = None
    for d_hi, v, n in shells:
        if v is None or v < thresh:
            break
        last = d_hi
    return last


def parse_partialator(path):
    if not os.path.exists(path):
        return {}
    t = open(path, errors="replace").read()
    r = {}
    ok = re.findall(r"^\s*(\d+) OK\s*$", t, flags=re.M)
    if ok:
        r["n_ok_final"] = int(ok[-1])
    res = re.findall(r"Log residual went from .*?, (\d+) crystals", t)
    if res:
        r["n_residual_final"] = int(res[-1])
    m = re.search(r"(\d+) crystals? (?:loaded|ready)", t)
    if m:
        r["n_loaded"] = int(m.group(1))
    flags = dict((k, int(v)) for v, k in re.findall(r"^\s*(\d+) (bad [\w ]+|[A-Za-z][\w /]*?)\s*$", t, flags=re.M)
                 if k not in ("OK",))
    if flags:
        r["flags"] = flags
    return r


def stats_block(d):
    r = {}
    for fom in ("CC", "CCstar", "Rsplit"):
        c = parse_compare(os.path.join(d, f"compare_{fom}.log"))
        r[fom] = c.get("value")
        if fom == "CC":
            r["pairs"] = c.get("pairs")
            r["d_range_A"] = c.get("d_range_A")
    r.update(parse_check(os.path.join(d, "check.log")))
    sh = parse_shells(os.path.join(d, "shells_CC.dat"))
    r["shells_CC"] = sh
    r["shells_Rsplit"] = parse_shells(os.path.join(d, "shells_Rsplit.dat"))
    r["shells_check"] = parse_check_shells(os.path.join(d, "shells_check.dat"))
    r["d_ccstar_0p5_A"] = d_at_cc(sh)
    return r


def cmd_range(mdir):
    c = parse_compare(os.path.join(mdir, "auto", "compare_CC.log"))
    if "d_range_A" not in c:
        sys.exit(f"no resolution range in {mdir}/auto/compare_CC.log")
    lo, hi = c["d_range_A"]
    print(f"{lo:.3f} {hi:.3f}")


def cmd_record(mdir, tag, frac, seed, perm, merger):
    rec = {"input": tag, "frac": float(frac), "seed": int(seed), "perm": perm, "merger": merger}
    sj = os.path.join(mdir, "in.stream.json")
    if os.path.exists(sj):
        s = json.load(open(sj))
        rec.update({k: s.get(k) for k in ("frames_available", "frames", "crystals", "cell")})
    if merger == "partialator":
        rec["partialator"] = parse_partialator(os.path.join(mdir, "partialator.log"))
    for w in ("auto", "fixed"):
        if os.path.isdir(os.path.join(mdir, w)):
            rec[w] = stats_block(os.path.join(mdir, w))
    json.dump(rec, open(os.path.join(mdir, "record.json"), "w"), indent=1)
    f = rec.get("fixed") or rec.get("auto") or {}
    print(f"{tag} {merger} perm={perm} frac={frac} seed={seed}: frames {rec.get('frames')} crystals {rec.get('crystals')}"
          f" CC1/2 {f.get('CC')} CC* {f.get('CCstar')} Rsplit {f.get('Rsplit')} compl {f.get('completeness_pct')}")


# ----------------------------------------------------------------------------------------------------------------
# collection and analysis
SCALARS = [("CC", "CC1/2"), ("CCstar", "CC*"), ("Rsplit", "Rsplit %"), ("completeness_pct", "completeness %"),
           ("multiplicity", "multiplicity"), ("mean_I_over_sigma", "<I/sigma>"), ("unique", "unique refl."),
           ("d_ccstar_0p5_A", "d(CC*>=0.5) A")]


def _ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"median": float(np.median(xs)), "min": float(min(xs)), "max": float(max(xs)), "n": len(xs)}


def _fmt(s, nd=3):
    if not s:
        return "-"
    if s["n"] == 1:
        return f"{s['median']:.{nd}f}"
    return f"{s['median']:.{nd}f} [{s['min']:.{nd}f}, {s['max']:.{nd}f}]"


def fit_spearman_brown(ns, ccs):
    """CC1/2(N) = N / (N + N0) (constant per-frame reliability, halves of N/2 frames). Least squares on a log grid."""
    ns, ccs = np.asarray(ns, float), np.asarray(ccs, float)
    if len(ns) < 3 or np.all(ccs <= 0):
        return None
    grid = np.logspace(0, 7, 7001)
    sse = [float(np.sum((ccs - ns / (ns + g)) ** 2)) for g in grid]
    i = int(np.argmin(sse))
    n0 = float(grid[i])
    nmax = float(ns.max())
    n09 = n0 * CC_FOR_CCSTAR_09 / (1 - CC_FOR_CCSTAR_09)
    return {"N0_frames": round(n0, 1), "rms_resid": round(math.sqrt(sse[i] / len(ns)), 4),
            "N_for_CC12_0p5": round(n0, 0), "N_for_CCstar_0p9": round(n09, 0),
            "extrapolation_factor_CCstar_0p9": round(n09 / nmax, 2),
            "note": "extrapolation, not a measurement, wherever the factor exceeds 1"}


def loglog_slope(ns, ys):
    pts = [(n, y) for n, y in zip(ns, ys) if n and y and n > 0 and y > 0]
    if len(pts) < 3:
        return None
    x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    return round(float(np.polyfit(x, y, 1)[0]), 3)


def _defined(table, key):
    """Fractions at which `key` is defined in at least 3/4 of the merges (CC1/2 is undefined when the half-sets share
    too few reflections: compare_hkl stops with 'Not enough reflections for scaling')."""
    out = []
    for fr in sorted(table):
        st, nm = table[fr].get(key), (table[fr].get("frames") or {}).get("n", 0)
        if st and nm and st["n"] >= 0.75 * nm:
            out.append(fr)
    return out


def classify(real, null, key):
    """Pre-registered rule (README): between the smallest fraction at which the statistic is defined and 100%,
    it depends on N if the seed ranges [min, max] at the two ends do not overlap. It is 'counting' if the null
    permuted over all reflections changes the same way with ranges that do not overlap either, and 'signal' if not."""
    fr = _defined(real, key)
    if len(fr) < 2:
        return "undefined at every fraction but one" if fr else "undefined"
    lo, hi = real[fr[0]][key], real[fr[-1]][key]
    first = f"defined from fraction {fr[0]:.2f}; "
    if not (lo["max"] < hi["min"] or hi["max"] < lo["min"]):
        return first + "no N-dependence resolved"
    dep = first + f"depends on N ({lo['median']:.3g} -> {hi['median']:.3g})"
    if null and fr[0] in null and fr[-1] in null:
        nlo, nhi = null[fr[0]].get(key), null[fr[-1]].get(key)
        if nlo and nhi and (nlo["max"] < nhi["min"] or nhi["max"] < nlo["min"]) and \
                (nhi["median"] - nlo["median"]) * (hi["median"] - lo["median"]) > 0:
            return dep + f"; the null changes the same way ({nlo['median']:.3g} -> {nhi['median']:.3g}): counting"
        return dep + "; the null does not: signal"
    return dep


def cmd_collect(root):
    recs = []
    for p in sorted(glob.glob(os.path.join(root, "*", "merges", "*", "record.json"))):
        try:
            recs.append(json.load(open(p)))
        except (OSError, ValueError):
            pass
    json.dump(recs, open(os.path.join(root, "results.json"), "w"), indent=1)
    w = "fixed"
    cols = ["input", "merger", "perm", "frac", "seed", "frames", "crystals", "n_ok_final"] + [k for k, _ in SCALARS] + \
           ["cell_a", "cell_c"]
    with open(os.path.join(root, "sweep.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in recs:
            s = r.get(w) or {}
            cm = (r.get("cell") or {}).get("median", {})
            row = [r["input"], r["merger"], r["perm"], r["frac"], r["seed"], r.get("frames"), r.get("crystals"),
                   (r.get("partialator") or {}).get("n_ok_final")] + [s.get(k) for k, _ in SCALARS] + \
                  [cm.get("a"), cm.get("c")]
            f.write("\t".join("" if v is None else str(v) for v in row) + "\n")

    groups = {}
    for r in recs:
        groups.setdefault((r["input"], r["merger"], r["perm"]), []).append(r)
    L = ["# Merge-redundancy sweep: summary", "",
         "Statistics in the FIXED resolution range of each input (the range compare_hkl accepts for the 100% merge in",
         "stream order). Cells: median [min, max] over the seeds at that fraction; seed 0 (stream order) is in the 100% row.",
         "CC1/2, CC*, Rsplit: compare_hkl --fom=CC|CCstar|Rsplit on the half-sets; completeness, multiplicity, <I/sigma>,",
         "unique: check_hkl on the full merge. d(CC*>=0.5): high-resolution edge of the last of the 10 shells, from low",
         "resolution, before shell CC1/2 drops below 0.143.", ""]
    analysis = {}
    for inp in sorted({g[0] for g in groups}):
        for merger in ("partialator", "process_hkl"):
            for perm in ("none", "all", "shell"):
                g = groups.get((inp, merger, perm))
                if not g:
                    continue
                byf = {}
                for r in g:
                    byf.setdefault(r["frac"], []).append(r)
                table = {}
                L.append(f"## {inp}, {merger}, {'real intensities' if perm == 'none' else 'null: permuted ' + perm}")
                L.append("")
                L.append("| fraction | frames | crystals | kept by partialator | " + " | ".join(n for _, n in SCALARS) +
                         " | median a, c (A) |")
                L.append("|" + "---|" * (5 + len(SCALARS)))
                for fr in sorted(byf):
                    rs = byf[fr]
                    st = {k: _ms([(r.get(w) or {}).get(k) for r in rs]) for k, _ in SCALARS}
                    st["frames"] = _ms([r.get("frames") for r in rs])
                    st["crystals"] = _ms([r.get("crystals") for r in rs])
                    st["kept"] = _ms([(r.get("partialator") or {}).get("n_ok_final") for r in rs])
                    st["a"] = _ms([((r.get("cell") or {}).get("median") or {}).get("a") for r in rs])
                    st["c"] = _ms([((r.get("cell") or {}).get("median") or {}).get("c") for r in rs])
                    table[fr] = st
                    L.append(f"| {fr:.2f} | {_fmt(st['frames'], 0)} | {_fmt(st['crystals'], 0)} | {_fmt(st['kept'], 0)} | " +
                             " | ".join(_fmt(st[k], 1 if k in ('Rsplit', 'completeness_pct', 'unique') else 3)
                                        for k, _ in SCALARS) +
                             f" | {_fmt(st['a'], 2)}, {_fmt(st['c'], 2)} |")
                L.append("")
                analysis[(inp, merger, perm)] = table
    L.append("## Fits and classification (real intensities against the null permuted over all reflections)")
    L.append("")
    out = {}
    for (inp, merger, perm), table in analysis.items():
        if perm != "none":
            continue
        null = analysis.get((inp, merger, "all"))
        fr = sorted(table)
        ns = [table[f]["frames"]["median"] for f in fr]
        cc = [table[f]["CC"]["median"] if table[f]["CC"] else None for f in fr]
        sb = fit_spearman_brown([n for n, c in zip(ns, cc) if c is not None], [c for c in cc if c is not None])
        rs = loglog_slope(ns, [table[f]["Rsplit"]["median"] if table[f]["Rsplit"] else None for f in fr])
        mu = loglog_slope(ns, [table[f]["multiplicity"]["median"] if table[f]["multiplicity"] else None for f in fr])
        cls = {name: classify(table, null, k) for k, name in SCALARS}
        out[f"{inp}|{merger}"] = {"spearman_brown_CC12": sb, "loglog_slope_Rsplit_vs_frames": rs,
                                  "loglog_slope_multiplicity_vs_frames": mu, "classification": cls}
        L.append(f"### {inp}, {merger}")
        L.append(f"- CC1/2(N) = N/(N+N0) fit: {json.dumps(sb)}")
        L.append(f"- log-log slope vs frames: Rsplit {rs} (1/sqrt(multiplicity) predicts -0.5), multiplicity {mu} (1.0)")
        for name, c in cls.items():
            L.append(f"- {name}: {c}")
        L.append("")
    json.dump(out, open(os.path.join(root, "analysis.json"), "w"), indent=1)
    rp = os.path.join(root, "repro.txt")
    if os.path.exists(rp):
        L += ["## Reproduction check", "", "```", open(rp).read().rstrip(), "```", ""]
    pp = os.path.join(root, "provenance.json")
    if os.path.exists(pp):
        L += ["## Provenance", "", "```", open(pp).read().rstrip(), "```", ""]
    open(os.path.join(root, "summary.md"), "w").write("\n".join(L) + "\n")
    print(f"collect: {len(recs)} merges -> {root}/results.json, sweep.tsv, analysis.json, summary.md")


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    cmd, a = argv[1], argv[2:]
    if cmd == "frames":
        cmd_frames(*a)
    elif cmd == "subset":
        cmd_subset(*a)
    elif cmd == "range":
        cmd_range(*a)
    elif cmd == "record":
        cmd_record(*a)
    elif cmd == "collect":
        cmd_collect(*a)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
