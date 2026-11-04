"""GLINT arms on the same 907 x 170-peak cxidb-45 lists, pinned tree (95d51ce), shipped defaults.

Re-measures the earlier GLINT rows on these lists so that they can be scored at the
strict bar together with the DIALS arms (the earlier 32.0% and 97.8% used a length-only 5% gate,
prok_rerun_s3df.py:len_match). Same entry points as that harness: index_blind_fast per frame, and
hybrid_index(frames) at its defaults. Adds hybrid_index(frames, Mc_known=ProK), GLINT given the same
target cell as DIALS's K arms, and the three arms again on the azimuth-scrambled lists (the null).
Saves each frame's M (real basis, columns; NaN when None); stills_score.py applies the gate."""
import argparse
import glob
import hashlib
import json
import os
import platform
import re
import sys
import time

import numpy as np
import torch

import glint
from glint.glint_fast import index_blind_fast
from glint.hybrid_stream import hybrid_index
from glint.lattice import cell_to_Ar

PROK = cell_to_Ar(68.7, 68.7, 108.6, 90.0, 90.0, 90.0)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load(path):
    D = np.load(path)
    return [D["q_%d" % i] for i in range(int(D["n"]))]       # as stored, like prok_rerun_s3df.py


def stack(Ms):
    out = np.full((len(Ms), 3, 3), np.nan)
    for i, M in enumerate(Ms):
        if M is not None:
            out[i] = np.asarray(M, float)
    return out


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 64 else "array%s" % (x.shape,)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return repr(x)[:200]


def arm_pf(frames):
    index_blind_fast(frames[0]); sync()                        # warm-up, not timed
    Ms, ts = [], []
    for q in frames:
        t0 = time.perf_counter()
        M = index_blind_fast(q)
        sync()
        ts.append(time.perf_counter() - t0)
        Ms.append(M)
    return Ms, np.array(ts), {}


def arm_hyb(frames, Mc_known=None):
    sync()
    t0 = time.perf_counter()
    results, stats = hybrid_index(frames, Mc_known=Mc_known)
    sync()
    wall = time.perf_counter() - t0
    assert len(results) == len(frames), (len(results), len(frames))
    return [r["M"] for r in results], np.array([wall]), stats


def glint_env_names(tree):
    """Every environment variable the glint package reads (os.environ.get / getenv / setdefault / [...]), found by
    scanning its source, so a tuning variable such as GLINT_CONSENSUS_MIN_FRAC cannot change between a cached
    arm and a new one unnoticed."""
    pat = re.compile(r"""os\.(?:environ\.get|getenv|environ\.setdefault)\(\s*["']([A-Z0-9_]+)|os\.environ\[["']([A-Z0-9_]+)""")
    names = set()
    for p in glob.glob(os.path.join(tree, "glint", "**", "*.py"), recursive=True):
        names.update(a or b for a, b in pat.findall(open(p).read()))
    return sorted(names)


def _match(prov):
    """The provenance fields a cached arm must share with this invocation to be reused: the input hashes
    (not their paths), the commit, the GPU and the settings read from the environment. Unset variables are
    dropped, so an arm recorded with a shorter variable list still matches when nothing else was set."""
    env = {k: v for k, v in (prov.get("env") or {}).items()
           if v is not None and k not in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES")}
    return dict(commit=prov.get("commit"), gpu=prov.get("gpu"), env=env,
                inputs=list((prov.get("npz_sha256") or {}).values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--npz-null", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tree = os.path.dirname(os.path.dirname(os.path.abspath(glint.__file__)))
    commit = open(os.path.join(tree, "COMMIT")).read().strip() if os.path.exists(os.path.join(tree, "COMMIT")) else "?"
    prov = dict(argv=sys.argv, tree=tree, commit=commit, host=platform.node(), torch=torch.__version__,
                cuda=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                env={k: os.environ.get(k) for k in sorted(set(glint_env_names(tree)) | {
                    "STEPS", "TOL", "REFINER", "QDIST", "OMP_NUM_THREADS", "PYTHONPATH"})},
                npz_sha256={p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (a.npz, a.npz_null)},
                time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    print("glint_samepeaks", json.dumps(prov), flush=True)
    assert prov["cuda"], "no GPU: refusing to time GLINT on CPU"
    # provenance.json is the run that first computed the arms; a resumed run keeps it and logs itself to
    # provenance_resume.jsonl. Each arm's npz carries the provenance of the invocation that computed it,
    # and an existing arm is reused only if it was computed from the same inputs, commit, GPU and settings.
    pfile = os.path.join(a.out, "provenance.json")
    if os.path.exists(pfile):
        with open(os.path.join(a.out, "provenance_resume.jsonl"), "a") as f:
            f.write(json.dumps(prov) + "\n")
    else:
        json.dump(prov, open(pfile, "w"), indent=1)
    real, null = load(a.npz), load(a.npz_null)
    arms = [("G_pf", real, lambda f: arm_pf(f)),
            ("G_hyb", real, lambda f: arm_hyb(f)),
            ("G_kc", real, lambda f: arm_hyb(f, Mc_known=PROK)),
            ("G_pf_null", null, lambda f: arm_pf(f)),
            ("G_hyb_null", null, lambda f: arm_hyb(f)),
            ("G_kc_null", null, lambda f: arm_hyb(f, Mc_known=PROK))]
    for name, frames, fn in arms:
        dst = os.path.join(a.out, name + ".npz")
        if os.path.exists(dst):
            with np.load(dst) as Z:     # arms written before the per-arm record fall back to provenance.json
                old = json.loads(str(Z["prov"])) if "prov" in Z.files else json.load(open(pfile))
            bad = [k for k in sorted(_match(prov)) if _match(old).get(k) != _match(prov)[k]]
            if bad:
                raise SystemExit(f"{dst} was computed with different {bad}; move it aside to recompute it")
            print("skip (exists, same inputs and settings)", dst, flush=True)
            continue
        Ms, t, stats = fn(frames)
        np.savez(dst, M=stack(Ms), t=t, stats=json.dumps(jsonable(stats)), prov=json.dumps(prov))
        print("%-11s frames %d  M not None %d  time %.1f s  stats %s" % (
            name, len(frames), sum(M is not None for M in Ms), float(t.sum()),
            json.dumps(jsonable({k: stats.get(k) for k in ("support", "n_idx", "n_blind", "n_rescued", "edges")}))),
            flush=True)
    print("GLINT_DONE", flush=True)


if __name__ == "__main__":
    main()
