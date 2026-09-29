"""The PREREG.md harness: hybrid_index(select="first") vs select="matched" on the cxidb-17 120 and 480 sets.

Three stages, each a separate command so every hybrid_index call runs in its own single-threaded process:

  run    one arm x one set x one select -> <out>/<arm>_<set>_<select>.json (the matrices, the stats, the wall time)
  all    every `run` in parallel (arm null waits for clean_480_first, whose consensus cell it uses), then `score`
  score  strict / observable flags from the runs + the 23 Sep reference flags -> <out>/summary.json, printed

Arms (PREREG.md): clean; n50 = experiments/joint_ceiling/stress.py's perturb(frames, "N50", 20260925), copied
below verbatim; null = every frame of the 480 set azimuth-scrambled (rng [20260928, i]) with Mc_known = the clean
480 consensus cell. The reference flags for P1 are read from exp/joint-ceiling's results_{120,480}.json with
`git show` (the branch is not merged), so this needs that ref locally.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python experiments/select_matched/measure.py all \\
      --f120 experiments/frames_cxidb_clean.txt --f480 q480_fix.txt --out experiments/select_matched/runs
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np                                                          # noqa: E402

ARMS = ("clean", "n50", "null")
SETS = ("120", "480")
SELECTS = ("first", "matched")
STRESS_SEED = 20260925
STRESS_CONDS = ("clean", "S20", "S12", "N50", "N100", "F10", "F20")          # stress.py's CONDS: rng uses the index
NULL_SEED = 20260928
MD5 = {"120": "f89fa57d", "480": "d6d86c1b"}
REF_REF = "exp/joint-ceiling"


def load_set(path, tag):
    from glint.glint_fast import load
    raw = open(path, "rb").read()
    md5 = hashlib.md5(raw).hexdigest()
    if not md5.startswith(MD5[tag]):
        sys.exit(f"{path}: md5 {md5[:8]} is not the {tag} set ({MD5[tag]})")
    return [np.asarray(q, float) for q in load(path) if len(q) >= 6], md5


def perturb(frames, cond, seed):
    """stress.py's perturb, verbatim (exp/joint-ceiling 3d0ad38), N50/N100 branch only."""
    from glint.multilattice import scramble_azimuth
    out = []
    n = len(frames)
    for i, q in enumerate(frames):
        rng = np.random.default_rng([seed, STRESS_CONDS.index(cond), i])
        q = np.asarray(q, float)
        if cond in ("N50", "N100"):
            m = int(round(int(cond[1:]) / 100 * len(q)))
            others = rng.integers(0, n - 1, m)
            others = others + (others >= i)                     # uniform over OTHER frames
            extra = np.array([frames[j][rng.integers(len(frames[j]))] for j in others], float).reshape(-1, 3)
            extra = scramble_azimuth(extra, rng)
            q = np.concatenate([q, extra])
            q = q[rng.permutation(len(q))]
        else:
            raise ValueError(cond)
        out.append(q)
    return out


def scrambled(frames):
    from glint.multilattice import scramble_azimuth
    return [scramble_azimuth(q, np.random.default_rng([NULL_SEED, i])) for i, q in enumerate(frames)]


def git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:                                                   # noqa: BLE001
        return None


def cmd_run(a):
    import torch
    from glint.hybrid_stream import hybrid_index
    path = a.f120 if a.set == "120" else a.f480
    orig, md5 = load_set(path, a.set)
    Mc_known = None
    if a.arm == "clean":
        seen = orig
    elif a.arm == "n50":
        seen = perturb(orig, "N50", STRESS_SEED)
    else:
        if a.set != "480":
            sys.exit("arm null is the 480 set only")
        ref = json.load(open(os.path.join(a.out, "clean_480_first.json")))
        Mc_known = np.asarray(ref["stats"]["Mc"], float)
        seen = scrambled(orig)
    t0 = time.time()
    res, st = hybrid_index(seen, Mc_known=Mc_known, select=a.select)
    wall = time.time() - t0
    keep = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in st.items()
            if k not in ("escalation",)}
    out = dict(header=dict(arm=a.arm, set=a.set, select=a.select, frames=os.path.basename(path), md5=md5,
                           n=len(orig), git=git_head(), prereg="experiments/select_matched/PREREG.md",
                           torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                           env={k: os.environ.get(k) for k in ("STEPS", "OMP_NUM_THREADS",
                                                              "KC_FP", "NC", "CUDA_VISIBLE_DEVICES")},
                           wall_s=round(wall, 1)),
               stats=keep,
               M=[None if r["M"] is None else np.asarray(r["M"], float).tolist() for r in res],
               n_seen=[len(q) for q in seen])
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"{a.arm}_{a.set}_{a.select}.json"), "w"))
    print(f"{a.arm}_{a.set}_{a.select}: idx {st['n_idx']}/{len(orig)}  kc {st['n_kc_searches']}  "
          f"swaps {st['n_select_swaps']}  {wall:.0f} s", flush=True)


def _spawn(a, arm, tag, sel):
    env = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1", STEPS="8", CUDA_VISIBLE_DEVICES="")
    return subprocess.Popen([sys.executable, os.path.abspath(__file__), "run", "--arm", arm, "--set", tag,
                             "--select", sel, "--f120", a.f120, "--f480", a.f480, "--out", a.out], env=env)


def cmd_all(a):
    os.makedirs(a.out, exist_ok=True)
    first = [(arm, tag, sel) for arm in ("clean", "n50") for tag in SETS for sel in SELECTS]
    procs = {k: _spawn(a, *k) for k in first}
    done = set()
    null_procs = {}
    while len(done) < len(procs) + len(null_procs) or not null_procs:
        for k, p in list(procs.items()) + list(null_procs.items()):
            if k in done or p.poll() is None:
                continue
            if p.returncode != 0:
                sys.exit(f"run {k} failed ({p.returncode})")
            done.add(k)
        if ("clean", "480", "first") in done and not null_procs:
            null_procs = {("null", "480", sel): _spawn(a, "null", "480", sel) for sel in SELECTS}
        time.sleep(5)
    cmd_score(a)


def cmd_score(a):
    from glint.glint_fast import GATE_FRAC, GATE_MIN, LYSO, matched_strict
    from glint.multishot import same_lattice
    LY = np.asarray(LYSO, float)
    origs = {t: load_set(a.f120 if t == "120" else a.f480, t)[0] for t in SETS}

    def get(arm, tag, sel):
        return json.load(open(os.path.join(a.out, f"{arm}_{tag}_{sel}.json")))

    def strict(M, q):
        if M is None:
            return 0
        M = np.asarray(M, float)
        m = matched_strict(M, q)
        return int(same_lattice(M, LY) and m >= GATE_MIN and m >= GATE_FRAC * len(q))

    def obs(M, q):
        if M is None:
            return 0
        m = matched_strict(np.asarray(M, float), q)
        return int(m >= GATE_MIN and m >= GATE_FRAC * len(q))

    def ref_flags(tag):
        d = json.loads(subprocess.check_output(["git", "-C", ROOT, "show",
                                                f"{REF_REF}:experiments/joint_ceiling/results_{tag}.json"]))
        return d["per_frame"]["hyb_Mc"], d["per_frame"]["admm"][0]

    S = dict(header=dict(git=git_head(), prereg="experiments/select_matched/PREREG.md",
                         out=os.path.basename(os.path.normpath(a.out))))
    for arm in ("clean", "n50"):
        for tag in SETS:
            q0 = origs[tag]
            F = {sel: [strict(M, q) for M, q in zip(get(arm, tag, sel)["M"], q0)] for sel in SELECTS}
            f, m = np.array(F["first"]), np.array(F["matched"])
            row = dict(first=int(f.sum()), matched=int(m.sum()), diff=int(m.sum() - f.sum()),
                       gained=int(((m == 1) & (f == 0)).sum()), lost=int(((m == 0) & (f == 1)).sum()),
                       wall_s={sel: get(arm, tag, sel)["header"]["wall_s"] for sel in SELECTS},
                       kc_searches={sel: get(arm, tag, sel)["stats"]["n_kc_searches"] for sel in SELECTS},
                       swaps=get(arm, tag, "matched")["stats"]["n_select_swaps"],
                       support=get(arm, tag, "first")["stats"]["support"],
                       flags=dict(first=f.tolist(), matched=m.tolist()))
            if arm == "clean":
                rh, r0 = ref_flags(tag)
                row["ref"] = dict(hyb_Mc=int(sum(rh)), admm0=int(sum(r0)),
                                  mismatch_first=int(sum(x != y for x, y in zip(rh, f))),
                                  mismatch_matched=int(sum(x != y for x, y in zip(r0, m))))
            S[f"{arm}_{tag}"] = row
    null = {sel: get("null", "480", sel) for sel in SELECTS}
    scr = scrambled(origs["480"])
    nf = np.array([obs(M, q) for M, q in zip(null["first"]["M"], scr)])
    nm = np.array([obs(M, q) for M, q in zip(null["matched"]["M"], scr)])
    S["null_480"] = dict(first=int(nf.sum()), matched=int(nm.sum()), diff=int(nm.sum() - nf.sum()),
                         registered={sel: int(sum(M is not None for M in null[sel]["M"])) for sel in SELECTS},
                         wall_s={sel: null[sel]["header"]["wall_s"] for sel in SELECTS})
    c1, c4 = S["clean_120"], S["clean_480"]
    n1, n4 = S["n50_120"], S["n50_480"]
    S["predictions"] = {
        "P1": dict(ok=bool(c1["matched"] == 91 and c4["matched"] == 370 and c1["ref"]["mismatch_matched"] == 0
                           and c4["ref"]["mismatch_matched"] == 0 and c1["ref"]["mismatch_first"] == 0
                           and c4["ref"]["mismatch_first"] == 0),
                   value=dict(matched=[c1["matched"], c4["matched"]], first=[c1["first"], c4["first"]],
                              mismatches=[c1["ref"], c4["ref"]])),
        "P2": dict(ok=bool(n4["diff"] >= 3 and n4["lost"] <= 1 and n1["diff"] >= 0 and n1["lost"] <= 1),
                   value=dict(d480=n4["diff"], lost480=n4["lost"], d120=n1["diff"], lost120=n1["lost"])),
        "P3": dict(ok=bool(S["null_480"]["diff"] <= 3), value=S["null_480"]["diff"]),
        "P4": dict(ok=bool(c4["diff"] >= 2 and c4["lost"] == 0 and c1["lost"] == 0),
                   value=dict(d480=c4["diff"], lost480=c4["lost"], lost120=c1["lost"])),
    }
    json.dump(S, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for k in ("clean_120", "clean_480", "n50_120", "n50_480"):
        r = S[k]
        print(f"{k:10s} first {r['first']:4d}  matched {r['matched']:4d}  diff {r['diff']:+d} "
              f"(+{r['gained']}/-{r['lost']})  swaps {r['swaps']}  kc {r['kc_searches']}  wall {r['wall_s']}"
              + (f"  ref {r['ref']}" if "ref" in r else ""))
    print(f"null_480   first {S['null_480']['first']}  matched {S['null_480']['matched']}  "
          f"registered {S['null_480']['registered']}  wall {S['null_480']['wall_s']}")
    for p, v in S["predictions"].items():
        print(f"  {p}: {'HELD' if v['ok'] else 'FAILED'}  {v['value']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("run", "all", "score"))
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--set", choices=SETS)
    ap.add_argument("--select", choices=SELECTS)
    ap.add_argument("--f120", default=os.path.join(ROOT, "experiments", "frames_cxidb_clean.txt"))
    ap.add_argument("--f480", default="q480_fix.txt")
    ap.add_argument("--out", default=os.path.join(HERE, "runs"))
    a = ap.parse_args(argv)
    {"run": cmd_run, "all": cmd_all, "score": cmd_score}[a.stage](a)


if __name__ == "__main__":
    main()
