#!/usr/bin/env python3
"""Helpers: solution-file shuffles, CrystFEL stream subsets, log parsing, and the results table.

Thin glue around the shipped code paths (GLINT 95d51ce `glint.glint_cli --tofile`, CrystFEL 0.10.2). Nothing here
indexes, integrates or merges; those are GLINT's and CrystFEL's own programs, called from the sbatch files.
numpy only, plus glint.multishot.same_lattice / glint.lattice.cell_to_Ar (PYTHONPATH = the pinned tree) for the cell check.

  merge_tools.py shuffle   IN.sol OUT.sol SEED          orientations permuted among frames (a derangement; cell kept)
  merge_tools.py splitfile KEYS SEED OUT.txt           partialator --custom-split: halfA/halfB keyed BY KEY, not chunk order
  merge_tools.py sollist   IN.sol OUT.lst               the frames of a solution file, as an indexamajig input list
  merge_tools.py keys      STREAM OUT.keys              frames with >= 1 crystal ("<image> <event>", sorted)
  merge_tools.py common    A.keys B.keys OUT.keys       intersection
  merge_tools.py subset    STREAM KEYS|- SEED FIRST OUT selected chunks, order permuted by SEED (0 = as is),
                                                     FIRST=1 keeps only the first crystal of each chunk
  merge_tools.py cellcheck IN.sol "a b c al be ga"      fraction of solutions that are the reference lattice
  merge_tools.py collect   OUTDIR                       RESULTS.json + RESULTS.md from everything under OUTDIR
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np

BEGIN_CHUNK = "----- Begin chunk -----"
END_CHUNK = "----- End chunk -----"


# ----------------------------------------------------------------------------------------------------------------
# solution files (glint/predict.py::write_solution_file format)
#   <image> //<event> a*x a*y a*z b*x b*y b*z c*x c*y c*z shift_x shift_y <lattice_code>     (nm^-1)
def read_sol(path):
    rows = []
    for ln in open(path):
        p = ln.split()
        if len(p) < 14:
            continue
        rows.append((p[0], p[1], [float(x) for x in p[2:11]], p[11], p[12], p[13]))
    return rows


def write_sol(path, rows):
    with open(path, "w") as f:
        for img, ev, v, sx, sy, code in rows:
            f.write("%s %s %s %s %s %s\n" % (img, ev, " ".join("%.7f" % x for x in v), sx, sy, code))


def derangement(n, seed):
    """A permutation with no fixed point (n >= 2), from a seeded generator; deterministic per (n, seed)."""
    if n < 2:
        raise SystemExit("shuffle needs at least 2 solutions")
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
    raise SystemExit("no derangement found")


def cmd_shuffle(src, dst, seed):
    rows = read_sol(src)
    p = derangement(len(rows), int(seed))
    out = [(rows[i][0], rows[i][1], rows[p[i]][2], rows[i][3], rows[i][4], rows[i][5]) for i in range(len(rows))]
    write_sol(dst, out)
    print(f"shuffle: {len(rows)} solutions, seed {seed}, no frame keeps its own orientation -> {dst}")


def cmd_sollist(src, dst):
    seen, out = set(), []
    for img, ev, *_ in read_sol(src):
        if (img, ev) not in seen:
            seen.add((img, ev))
            out.append(f"{img} {ev}")
    open(dst, "w").write("\n".join(out) + "\n")
    print(f"sollist: {len(out)} frames -> {dst}")


# ----------------------------------------------------------------------------------------------------------------
# CrystFEL streams
def read_stream(path):
    """(header_lines, [(key, chunk_lines, n_crystals)]). key = '<image> <event>' as CrystFEL writes them."""
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


def first_crystal_only(lines):
    out, seen_end = [], False
    skipping = False
    for ln in lines:
        if ln.startswith("--- Begin crystal"):
            if seen_end:
                skipping = True
                continue
        if skipping:
            if ln.startswith("--- End crystal"):
                skipping = False
            continue
        out.append(ln)
        if ln.startswith("--- End crystal"):
            seen_end = True
    return out


def cmd_keys(stream, dst):
    _, chunks = read_stream(stream)
    ks = sorted({k for k, _, n in chunks if n > 0})
    multi = sum(1 for _, _, n in chunks if n > 1)
    open(dst, "w").write("\n".join(ks) + ("\n" if ks else ""))
    print(f"keys: {len(chunks)} chunks, {len(ks)} with a crystal, {multi} with more than one -> {dst}")


def read_keys(path):
    return [ln.rstrip("\n") for ln in open(path) if ln.strip()]


def cmd_common(a, b, dst):
    ka, kb = set(read_keys(a)), set(read_keys(b))
    ks = sorted(ka & kb)
    open(dst, "w").write("\n".join(ks) + ("\n" if ks else ""))
    print(f"common: {len(ka)} & {len(kb)} -> {len(ks)} -> {dst}")


def cmd_subset(stream, keys, seed, first, dst):
    header, chunks = read_stream(stream)
    by = {}
    for k, lines, n in chunks:
        if n > 0 and k not in by:              # a frame appears once per stream; keep the first if not
            by[k] = lines
    if keys == "-":
        order = [k for k, _, n in chunks if n > 0]
        missing = []
    else:
        want = sorted(read_keys(keys))         # the SAME order for every arm -> paired splits
        missing = [k for k in want if k not in by]
        order = [k for k in want if k in by]
    seed = int(seed)
    if seed:
        rng = np.random.default_rng(seed)
        order = [order[i] for i in rng.permutation(len(order))]
    with open(dst, "w") as f:
        f.writelines(header)
        for k in order:
            f.writelines(first_crystal_only(by[k]) if first == "1" else by[k])
    print(f"subset: {len(order)} chunks (seed {seed}, first-crystal {first}), {len(missing)} keys absent -> {dst}")
    if missing:
        open(dst + ".missing", "w").write("\n".join(missing) + "\n")


def cmd_splitfile(keys, seed, dst):
    """Write a partialator --custom-split file: "<image> <event> halfA|halfB" for the COMMON set.

    Added 17 Oct as the
    fix for the N0 pairing failure. partialator's default two-way split (merged.hkl1/.hkl2) is by
    alternating chunk ORDER, so one key absent from an arm shifts every later frame into the other
    half and the arms are no longer paired -- which is why the paired-split guard had to fail 50
    `common*` merges whose arm was short 13-16 keys of 998.

    The assignment here is keyed BY KEY and derived from the COMMON set with the same seed for
    every arm, so an arm that is missing a key simply contributes nothing to its own half and every
    other frame stays where it was. Verified on r194 common_CF_s1: the order-based split reproduces
    the first merge pass's CC 0.4770808 / 127 pairs exactly, the key-paired split gives 0.4932408 / 125, and
    dropping 16 of 114 keys still assigns all 98 survivors to two populated halves.
    """
    want = sorted(read_keys(keys))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(want))
    with open(dst, "w") as f:
        for rank, i in enumerate(perm):
            k = want[i]
            img, ev = k.rsplit(" //", 1) if " //" in k else (k, "")
            f.write("%s //%s %s\n" % (img, ev, "halfA" if rank % 2 == 0 else "halfB"))
    print(f"splitfile: {len(want)} keys, seed {seed} -> {dst} (halfA/halfB, keyed by key not order)")


# ----------------------------------------------------------------------------------------------------------------
# cell check of a solution file against the reference (the bar for the BLIND cell)
def sol_real_bases(path):
    for img, ev, v, *_ in read_sol(path):
        B = np.array([v[0:3], v[3:6], v[6:9]]).T / 10.0       # columns a*, b*, c* in 1/A
        yield (img, ev), np.linalg.inv(B).T                  # real basis, columns a, b, c in A


def cmd_cellcheck(sol, cell):
    from glint.lattice import cell_to_Ar
    from glint.multishot import same_lattice
    ref = cell_to_Ar(*[float(x) for x in cell.split()])
    n = ok = 0
    lens = []
    for _, M in sol_real_bases(sol):
        n += 1
        ok += bool(same_lattice(M, ref))
        lens.append(sorted(np.linalg.norm(M, axis=0)))
    med = np.median(np.array(lens), axis=0).round(2).tolist() if lens else []
    res = {"n_solutions": n, "n_same_lattice_as_reference": ok,
           "frac_same_lattice": (ok / n if n else None), "median_sorted_edges_A": med,
           "reference": cell, "predicate": "glint.multishot.same_lattice (rtol 0.05, ctol 0.06, vtol 0.10)"}
    print(json.dumps(res))
    return res


# ----------------------------------------------------------------------------------------------------------------
# misorientation between two arms on their common frames (diagnostic only)
def stream_bases(path):
    """key -> real basis (columns a,b,c in A) of the FIRST crystal of each chunk (CrystFEL astar/bstar/cstar, nm^-1)."""
    out, key, a, b = {}, None, None, None
    img = ev = ""
    for ln in open(path, errors="replace"):
        if ln.startswith("Image filename:"):
            img = ln.split(":", 1)[1].strip(); ev = ""
        elif ln.startswith("Event:"):
            ev = ln.split(":", 1)[1].strip()
        elif ln.startswith("astar ="):
            a = [float(x) for x in ln.split("=")[1].split()[:3]]
        elif ln.startswith("bstar ="):
            b = [float(x) for x in ln.split("=")[1].split()[:3]]
        elif ln.startswith("cstar ="):
            c = [float(x) for x in ln.split("=")[1].split()[:3]]
            k = f"{img} {ev}".strip()
            if k not in out and a and b:
                B = np.array([a, b, c]).T / 10.0
                out[k] = np.linalg.inv(B).T
            a = b = None
    return out


def laue_ops(cell, tol=0.01):
    """Signed permutation matrices S (crystal basis) that preserve the reference metric: the holohedry ops.
    tol 1%: btx's r0016 cell has a, b 0.17% apart (tetragonal); r194's b, c differ by 2.6% and beta = 90.8 deg,
    neither of which a 1% tolerance merges."""
    from itertools import permutations, product
    from glint.lattice import cell_to_Ar
    A = cell_to_Ar(*[float(x) for x in cell.split()])
    G = A.T @ A
    s = np.sqrt(np.outer(np.diag(G), np.diag(G)))
    ops = []
    for perm in permutations(range(3)):
        for sg in product((1, -1), repeat=3):
            S = np.zeros((3, 3))
            for i, j in enumerate(perm):
                S[j, i] = sg[i]
            if np.all(np.abs(S.T @ G @ S - G) <= tol * s):
                ops.append(S)
    return ops


def rot_angle_deg(Q):
    U, _, Vt = np.linalg.svd(Q)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def misorientation(path_a, path_b, keys, cell):
    A, B = stream_bases(path_a), stream_bases(path_b)
    ops = laue_ops(cell)
    ang = []
    for k in keys:
        if k in A and k in B:
            ang.append(min(rot_angle_deg(A[k] @ np.linalg.inv(B[k] @ S)) for S in ops))
    if not ang:
        return None
    ang = np.array(ang)
    return {"n": int(len(ang)), "median_deg": float(np.median(ang)), "frac_le_1deg": float(np.mean(ang <= 1.0)),
            "frac_le_3deg": float(np.mean(ang <= 3.0)), "n_laue_ops": len(ops)}


# ----------------------------------------------------------------------------------------------------------------
# log parsing
def _f(x):
    try:
        return float(x)
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
    for name, key in (("<snr>", "mean_I_over_sigma"), ("redundancy", "redundancy"), ("completeness", "completeness_pct")):
        m = re.search(r"Overall %s = ([-\d.eE+naninf]+)" % re.escape(name), t)
        if m:
            r[key] = _f(m.group(1))
    return r


def parse_partialator(path):
    if not os.path.exists(path):
        return {}
    t = open(path, errors="replace").read()
    ok = re.findall(r"^\s*(\d+) OK\s*$", t, flags=re.M)
    res = re.findall(r"\((\d+) crystals\)", t)
    r = {"n_ok_final": int(ok[-1]) if ok else None, "n_residual_final": int(res[-1]) if res else None}
    m = re.search(r"Log residual went from .*?, (\d+) crystals", t)
    if m:
        r["n_in"] = int(m.group(1))
    return r


def parse_shells(path):
    """[(d_hi_A, value, nref)] low -> high resolution, from a compare_hkl --shell-file."""
    rows = []
    if not os.path.exists(path):
        return rows
    for ln in open(path):
        p = ln.split()
        if len(p) < 6 or not re.match(r"^[\d.]+$", p[0]):
            continue
        v = _f(p[1])
        rows.append((10.0 / float(p[5]), v, int(float(p[2]))))
    return rows


def cc_half_resolution(path, thresh=0.143):
    """High-resolution edge of the last shell, counting from low resolution, before CC1/2 first drops below
    `thresh` (CC* = 0.5). None if the first shell already fails."""
    rows = parse_shells(path)
    last = None
    for d_hi, v, n in rows:
        if v is None or not np.isfinite(v) or v < thresh:
            break
        last = d_hi
    return last


def parse_glint_log(path):
    r = {}
    if not os.path.exists(path):
        return r
    for ln in open(path, errors="replace"):
        s = ln.strip()
        for tag in ("blind indexed", "consensus cell", "consensus REFUSED", "N-best recovered", "rescued failures",
                    "gate (strict)", "FINAL indexed", "solution file"):
            if s.startswith(tag):
                r[tag] = s.split(":", 1)[1].strip() if ":" in s else s
    return r


def indexamajig_final(path):
    if not os.path.exists(path):
        return None
    last = None
    for ln in open(path, errors="replace"):
        if ln.startswith("Final:"):
            last = ln.strip()
    return last


# ----------------------------------------------------------------------------------------------------------------
# the results table
ROW_ORDER = ["CF", "CFnr", "GLb", "GLk"] + [f"N{s}" for s in (1, 2, 3)]          # own merges (B0/N0 exist on C only)


def merge_record(d):
    """FOMs of one merge directory written by merge_one.sh."""
    rec = {"dir": d}
    rec.update({"partialator": parse_partialator(os.path.join(d, "partialator.log"))})
    for fom in ("CC", "CCstar", "Rsplit"):
        rec[fom] = parse_compare(os.path.join(d, f"compare_{fom}.log"))
    rec["check"] = parse_check(os.path.join(d, "check.log"))
    rec["d_cchalf_0.143_A"] = cc_half_resolution(os.path.join(d, "shells_CC.dat"))
    if os.path.exists(os.path.join(d, "n_in.txt")):
        rec["n_chunks_in"] = int(open(os.path.join(d, "n_in.txt")).read().split()[0])
    return rec


def ms(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return None, None, 0
    return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0), len(xs)


def paired(a, b):
    """Paired difference a - b over seeds present in both: mean, sd, n, sign count."""
    ks = sorted(set(a) & set(b))
    dd = [a[k] - b[k] for k in ks if a[k] is not None and b[k] is not None]
    if not dd:
        return None
    m, s, n = ms(dd)
    return {"mean": m, "sd": s, "n": n, "n_pos": int(sum(x > 0 for x in dd)), "n_neg": int(sum(x < 0 for x in dd))}


def verdict(pd, sd_ctrl):
    """Decision rule reused from the pre-registered integration matrix (r51/r194, Oct 2026): a difference counts when the
    paired delta has one sign in >= 8 of 10 orders AND |mean| exceeds one split-sd of the control (CF)."""
    if pd is None or sd_ctrl is None:
        return "n/a"
    k = max(pd["n_pos"], pd["n_neg"])
    if k >= 8 and abs(pd["mean"]) > sd_ctrl:
        return ("higher" if pd["mean"] > 0 else "lower") + f" ({k}/{pd['n']} orders, |mean| > CF split-sd)"
    return f"no detected difference ({pd['n_pos']}+/{pd['n_neg']}- of {pd['n']}; CF split-sd {sd_ctrl:.3f})"


def cmd_collect(outdir):
    res = {"commit": None, "crystfel": None, "datasets": {}}
    cfile = os.path.join(outdir, "provenance.json")
    if os.path.exists(cfile):
        # Fixed 17 Oct.
        # provenance.json carries "datasets": "r51 r194 r0016" -- a STRING, the argv of the run --
        # so a plain res.update() replaced the dict initialised above and the first
        # res["datasets"][ds] = D below raised TypeError: 'str' object does not support item
        # assignment. That killed job 41037263 AFTER all 206 merges were on disk, so no RESULTS.md
        # was ever written. Keep the provenance value under its own name and leave the dict intact.
        _prov = json.load(open(cfile))
        _ds_argv = _prov.pop("datasets", None)
        res.update(_prov)
        res["datasets"] = {}                      # never let provenance overwrite the results dict
        if _ds_argv is not None:
            res["datasets_argv"] = _ds_argv
    lines = ["# Merge results (generated by merge_tools.py collect; every value read from the files under this directory)", ""]
    lines.append(f"GLINT commit {res.get('commit')}; {res.get('crystfel')}; merge = partialator "
                 f"{res.get('merge_argv', '(see commands.log)')}; argv of every step in commands.log.")
    lines.append("")
    for ds in sorted(os.listdir(outdir)):
        dd = os.path.join(outdir, ds)
        if not os.path.isdir(os.path.join(dd, "merges")):
            continue
        D = {"glint": {}, "indexamajig": {}, "own": {}, "common": {}, "commonk": {}, "cellcheck": {}, "misorientation": {}}
        for arm in ("GLb", "GLk"):
            D["glint"][arm] = parse_glint_log(os.path.join(dd, f"{arm}_glint.log"))
            cc = os.path.join(dd, f"{arm}_cellcheck.json")
            if os.path.exists(cc):
                D["cellcheck"][arm] = json.load(open(cc))
        for log in sorted(glob.glob(os.path.join(dd, "idx_*.log"))):
            D["indexamajig"][os.path.basename(log)[4:-4]] = indexamajig_final(log)
        for kf in sorted(glob.glob(os.path.join(dd, "*.keys"))):
            D.setdefault("n_keys", {})[os.path.basename(kf)[:-5]] = len(read_keys(kf))
        for m in sorted(glob.glob(os.path.join(dd, "merges", "own_*"))):
            D["own"][os.path.basename(m)[4:]] = merge_record(m)
        for setname in ("common", "commonk", "commonnodcc"):
            D.setdefault(setname, {})
            for m in sorted(glob.glob(os.path.join(dd, "merges", f"{setname}_*_s*"))):
                b = os.path.basename(m)
                arm, seed = b[len(setname) + 1:].rsplit("_s", 1)
                D[setname].setdefault(arm, {})[int(seed)] = merge_record(m)
        for pair, kf in (("GLb_vs_CF", "C.keys"), ("GLk_vs_CF", "Ck.keys")):
            try:
                a = os.path.join(dd, f"{pair.split('_vs_')[0]}.stream")
                b = os.path.join(dd, "CF.stream")
                cell = open(os.path.join(dd, "cell.txt")).read().strip()
                if os.path.exists(a) and os.path.exists(os.path.join(dd, kf)):
                    D["misorientation"][pair] = misorientation(a, b, read_keys(os.path.join(dd, kf)), cell)
            except Exception as e:                                   # diagnostic only; never blocks the table
                D["misorientation"][pair] = {"error": repr(e)}
        res["datasets"][ds] = D

        nsol = {a: sum(1 for _ in open(os.path.join(dd, f"{a}.sol"))) for a in ("GLb", "GLk") + tuple(f"N{s}" for s in (1, 2, 3))
                if os.path.exists(os.path.join(dd, f"{a}.sol"))}
        acc = {a: {"solutions": n, "accepted_by_crystfel": D.get("n_keys", {}).get(a),
                   "frac": (D.get("n_keys", {}).get(a) / n if n and D.get("n_keys", {}).get(a) is not None else None)}
               for a, n in nsol.items()}
        D["acceptance"] = acc
        lines += [f"## {ds}", ""]
        lines.append("Frame flow: " + "; ".join(f"{k}: {v}" for k, v in D.get("n_keys", {}).items()))
        for a, r in acc.items():
            lines.append(f"CrystFEL acceptance of {a}: {r['accepted_by_crystfel']}/{r['solutions']}"
                         + (f" ({100 * r['frac']:.1f}%)" if r["frac"] is not None else "")
                         + ("  <- chance acceptance of a wrong orientation (frame-level null)" if a.startswith("N") else ""))
        for arm, g in D["glint"].items():
            if g:
                lines.append(f"GLINT {arm}: " + "; ".join(f"{k} {v}" for k, v in g.items()))
        for arm, c in D["cellcheck"].items():
            lines.append(f"cell check {arm}: {c.get('n_same_lattice_as_reference')}/{c.get('n_solutions')} solutions are the "
                         f"reference lattice; median sorted edges {c.get('median_sorted_edges_A')} A")
        for k, v in D["misorientation"].items():
            lines.append(f"misorientation {k}: {v}")
        lines.append("")
        for setname, pairs in (("common", ("GLb", "B0", "N0_1", "N0_2", "N0_3")), ("commonk", ("GLk",)),
                               ("commonnodcc", ("GLb", "N0_1"))):
            S = D[setname]
            if "CF" not in S:
                continue
            n_in = [r.get("n_chunks_in") for r in S["CF"].values()]
            lines.append(f"### {setname}: the same {n_in[0] if n_in else '?'} frames in every arm, "
                         f"{len(S['CF'])} half-split orders (mean +- sd)")
            lines.append("")
            lines.append("| arm | crystals merged (median) | d at CC1/2 0.143 (A, median) | CC1/2 | CC* | Rsplit % | <I/sigma> | completeness % |")
            lines.append("|---|---|---|---|---|---|---|---|")
            sd_cf = ms([r["CC"].get("value") for r in S["CF"].values()])[1]
            for arm in ("CF",) + pairs:
                if arm not in S:
                    continue
                R = S[arm]
                cc = ms([r["CC"].get("value") for r in R.values()])
                cs = ms([r["CCstar"].get("value") for r in R.values()])
                rs = ms([r["Rsplit"].get("value") for r in R.values()])
                sn = ms([r["check"].get("mean_I_over_sigma") for r in R.values()])
                cp = ms([r["check"].get("completeness_pct") for r in R.values()])
                nok = [r["partialator"].get("n_ok_final") for r in R.values() if r["partialator"].get("n_ok_final")]
                dres = [r.get("d_cchalf_0.143_A") for r in R.values() if r.get("d_cchalf_0.143_A")]
                f = lambda t, p=3: "-" if t[0] is None else f"{t[0]:.{p}f} +- {t[1]:.{p}f}"
                lines.append(f"| {arm} | {int(np.median(nok)) if nok else '-'} | {np.median(dres):.2f} | {f(cc)} | {f(cs)} | {f(rs, 1)} | {f(sn, 2)} | {f(cp, 1)} |"
                             if dres else
                             f"| {arm} | {int(np.median(nok)) if nok else '-'} | - | {f(cc)} | {f(cs)} | {f(rs, 1)} | {f(sn, 2)} | {f(cp, 1)} |")
            lines.append("")
            cfv = {k: r["CC"].get("value") for k, r in S["CF"].items()}
            for arm in pairs:
                if arm in S:
                    av = {k: r["CC"].get("value") for k, r in S[arm].items()}
                    pd = paired(av, cfv)
                    lines.append(f"- CC1/2 {arm} - CF: {pd and round(pd['mean'], 4)} +- {pd and round(pd['sd'], 4)}; {verdict(pd, sd_cf)}")
            if "N0_1" in S:
                n0 = {k: r["CC"].get("value") for k, r in S["N0_1"].items()}
                for arm in ("CF", "GLb", "B0"):
                    if arm in S:
                        av = {k: r["CC"].get("value") for k, r in S[arm].items()}
                        pd = paired(av, n0)
                        lines.append(f"- CC1/2 {arm} - null N0_1 (measured minus the null): "
                                     f"{pd and round(pd['mean'], 4)} +- {pd and round(pd['sd'], 4)}")
            lines.append("")
        lines.append("### each arm's own crystals (the LUTE-style merge; different frame sets, not a like-for-like row)")
        lines.append("")
        lines.append("| arm | chunks with a crystal | crystals merged | d at CC1/2 0.143 (A) | CC1/2 | CC* | Rsplit % | <I/sigma> | completeness % |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for arm in ROW_ORDER:
            r = D["own"].get(arm)
            if not r:
                continue
            g = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"
            lines.append(f"| {arm} | {r.get('n_chunks_in', '-')} | {r['partialator'].get('n_ok_final', '-')} | "
                         f"{g(r.get('d_cchalf_0.143_A'), 2)} | {g(r['CC'].get('value'))} | {g(r['CCstar'].get('value'))} | "
                         f"{g(r['Rsplit'].get('value'), 1)} | {g(r['check'].get('mean_I_over_sigma'), 2)} | "
                         f"{g(r['check'].get('completeness_pct'), 1)} |")
        lines.append("")
        for k, v in D["indexamajig"].items():
            lines.append(f"- indexamajig {k}: {v}")
        lines.append("")
    json.dump(res, open(os.path.join(outdir, "RESULTS.json"), "w"), indent=1, default=str)
    open(os.path.join(outdir, "RESULTS.md"), "w").write("\n".join(lines) + "\n")
    print(f"collect: {len(res['datasets'])} datasets -> {outdir}/RESULTS.md, RESULTS.json")


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    cmd, a = argv[1], argv[2:]
    if cmd == "shuffle":
        cmd_shuffle(*a)
    elif cmd == "splitfile":
        cmd_splitfile(*a)
    elif cmd == "sollist":
        cmd_sollist(*a)
    elif cmd == "keys":
        cmd_keys(*a)
    elif cmd == "common":
        cmd_common(*a)
    elif cmd == "subset":
        cmd_subset(*a)
    elif cmd == "cellcheck":
        r = cmd_cellcheck(a[0], a[1])
        if len(a) > 2:
            json.dump(r, open(a[2], "w"))
    elif cmd == "collect":
        cmd_collect(*a)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
