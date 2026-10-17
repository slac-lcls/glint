"""Driver: the DIALS routes behind the two earlier DIALS rows (1D-FFT blind 61.9%, grid with the cell 49.9%),
re-run on S3DF so that the strict scorer can grade them. Run with the DIALS 3.dev.1332 build's dials.python (as
dials_samepeaks.py).

ROUTE "bvs" -- the earlier rows. A port of the two NERSC drivers that made them (not in this folder):
  index_prok_dials.py     -> "DIALS 1D-FFT blind 61.9%"        (561/907, length gate, index_dials.out)
  index_prok_dials_kc.py  -> "DIALS grid, cell supplied 49.9%"  (453/907, length gate, index_kc.out)
Per frame, exactly as there:
  rlp = flex.vec3_double(q)                    q in GLINT's lab frame as stored, no rotation, no detector
  vectors, used = FFT1D(max_cell=200).find_basis_vectors(rlp)
               or RealSpaceGridSearch(max_cell=200, target_unit_cell=ProK).find_basis_vectors(rlp)
  for cm in combinations.candidate_orientation_matrices(vectors, max_combinations=25):
      frac = mean over the frame's peaks of all(|inv(A) q - round(inv(A) q)| < 0.15)
  keep the candidate with the largest frac (strict ">", so the first one wins a tie)
No refinement, no outlier rejection, no DIALS model evaluation. One strategy object per run, reused across
frames, as there. The single change: the kept candidate's A is stored, as M_glint = inv(A)^T (q @ M = hkl,
GLINT's convention), so that the strict bar can be applied. The originals kept only its cell. The originals'
own two gates (len_match, full_match) are recomputed on that cell for the regression check.

ROUTE "indexer" -- DIALS's own indexer, as the 28 Sep fairness harness drove it (glint branch exp/dials-stills,
9af3aec, experiments/dials_stills/dials_stills_bench.py:cmd_index; pointer not re-checked 16 Oct): Indexer.from_parameters(refl, expts,
params) + index(), as stills_process.Processor.index() calls them, with stills_process's phil_scope plus the
arm's overrides. That is the core of dials.index with the stills indexer. The spots reach DIALS through the
virtual panel of dials_samepeaks.py: build_frame and crystal_record, imported unchanged (per-frame lambda,
55 mm, 0.05 mm pixels, 3000 x 3000, variance 1/12 px^2, round trip checked per frame).

Output: one JSON line per frame, the same keys stills_score.py's loader reads (frame, ok, M_glint, t_proc, err), plus the
route's own fields. A re-run skips frames already in --out.
"""
import argparse
import copy
import hashlib
import json
import logging
import os
import platform
import sys
import time

import numpy as np

PROK = (68.7, 68.7, 108.6, 90.0, 90.0, 90.0)
MAXCELL = 200.0           # the NERSC drivers' MAXCELL
MAX_COMB = 25             # the NERSC drivers' max_combinations
TOL = 0.15                # the NERSC drivers' near-integer window ("<"), identical to GATE_TOL
HERE = os.path.dirname(os.path.abspath(__file__))   # dials_samepeaks.py sits beside this file

# arm -> route and settings. Fixed before any full-run frame is indexed (README).
ARMS = {
    # the two earlier rows
    "V1_bvs_fft1d_blind":      dict(route="bvs", strategy="fft1d", input="real"),
    "V2_bvs_rsgs_kc":          dict(route="bvs", strategy="rsgs", input="real"),
    # their nulls (the azimuth-scrambled lists of prep_null.py)
    "V1n_bvs_fft1d_blind_null": dict(route="bvs", strategy="fft1d", input="null"),
    "V2n_bvs_rsgs_kc_null":     dict(route="bvs", strategy="rsgs", input="null"),
    # DIALS's own indexer (28 Sep harness), bridging the bare route to the stills_process arms
    "S1_idx_fft1d_rac":        dict(route="indexer", method="fft1d", known=False, rac=True, max_cell=None, input="real"),
    "S2_idx_fft1d_norac":      dict(route="indexer", method="fft1d", known=False, rac=False, max_cell=None, input="real"),
    "S3_idx_fft1d_rac_mc200":  dict(route="indexer", method="fft1d", known=False, rac=True, max_cell=MAXCELL, input="real"),
    "S4_idx_rsgs_kc_norac":    dict(route="indexer", method="real_space_grid_search", known=True, rac=False, max_cell=None,
                                    input="real"),
    # null for S3, the only blind indexer arm that can pass on these frames (dry run, 16 Oct: DIALS's own max_cell
    # estimate was 60-80 A against c = 108.6 A); added 16 Oct before any full-run frame was indexed
    "S3n_idx_fft1d_rac_mc200_null": dict(route="indexer", method="fft1d", known=False, rac=True, max_cell=MAXCELL,
                                         input="null"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def len_match(lens, ltol=0.05):           # index_prok_dials.py:len_match, verbatim logic
    lens = np.asarray(lens, float)
    if lens.shape[0] < 3 or np.any(np.isnan(lens)):
        return False
    L = sorted(lens[:3]); P = sorted(PROK[:3])
    return all(abs(l - p) <= ltol * p for l, p in zip(L, P))


def full_match(uc, ltol=0.05, atol=2.0):  # index_prok_dials.py:full_match, verbatim logic
    if uc is None or not len_match(uc[:3], ltol):
        return False
    return all(abs(x - 90.0) <= atol for x in uc[3:6])


# ------------------------------------------------------------------------------------------------ route "bvs"
def make_strategy(name):
    from cctbx import uctbx
    from dials.algorithms.indexing import basis_vector_search
    if name == "fft1d":
        return basis_vector_search.FFT1D(max_cell=MAXCELL)
    return basis_vector_search.RealSpaceGridSearch(max_cell=MAXCELL, target_unit_cell=uctbx.unit_cell(PROK))


def bvs_frame(strategy, q):
    from dials.array_family import flex
    from dials.algorithms.indexing.basis_vector_search import combinations
    rlp = flex.vec3_double([tuple(map(float, v)) for v in q])
    vectors, used = strategy.find_basis_vectors(rlp)
    best_frac, best_uc, best_A, ncand = -1.0, None, None, 0
    for cm in combinations.candidate_orientation_matrices(vectors, max_combinations=MAX_COMB):
        ncand += 1
        A = np.array(cm.get_A()).reshape(3, 3)
        hkl = (np.linalg.inv(A) @ q.T).T
        frac = float(np.mean(np.all(np.abs(hkl - np.round(hkl)) < TOL, axis=1)))
        if frac > best_frac:
            best_frac, best_uc, best_A = frac, cm.get_unit_cell().parameters(), A
    rec = dict(n_vectors=len(vectors), n_candidates=ncand)
    if best_A is None:
        return rec, None
    M = np.linalg.inv(best_A).T                                   # q @ M = inv(A) q = hkl
    rec.update(best_frac=best_frac, m_best=int(round(best_frac * len(q))), cell=list(best_uc),
               orig_len_gate=bool(len_match(best_uc[:3])), orig_full_gate=bool(full_match(best_uc)))
    return rec, M


# -------------------------------------------------------------------------------------------- route "indexer"
def indexer_params(spec):
    from dials.command_line.stills_process import phil_scope
    from libtbx.phil import parse
    lines = ["indexing.method=%s" % spec["method"],
             "indexing.stills.refine_all_candidates=%s" % spec["rac"],
             "indexing.nproc=1",
             "refinement.parameterisation.scan_varying=False"]          # as Processor.index() sets it
    if spec["known"]:
        lines += ["indexing.known_symmetry.space_group=P43212",
                  "indexing.known_symmetry.unit_cell=" + ",".join(str(x) for x in PROK)]
    if spec["max_cell"] is not None:
        lines += ["indexing.max_cell=%s" % spec["max_cell"]]
    work = phil_scope.fetch(sources=[parse("\n".join(lines))])
    return work.extract(), lines, phil_scope.fetch_diff(source=work).as_str()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--npz", required=True, help="the 907 x 170-peak lists (real) for input=real arms")
    ap.add_argument("--npz-null", required=True, help="the azimuth-scrambled copy (prep_null.py) for input=null arms")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = ARMS[a.arm]
    npz = a.npz if spec["input"] == "real" else a.npz_null
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    D = np.load(npz)
    n = int(D["n"])
    stop = n if a.stop is None else min(a.stop, n)
    try:
        from dials.util.version import dials_version
        dver = dials_version()
    except Exception:                                                    # noqa: BLE001
        dver = None

    phil_diff = None
    if spec["route"] == "bvs":
        strategy = make_strategy(spec["strategy"])
    else:
        logging.disable(logging.WARNING)
        sys.path.insert(0, HERE)
        from dials_samepeaks import RT_TOL, build_frame, crystal_record   # unchanged
        from dials.algorithms.indexing.indexer import Indexer
        params, overrides, phil_diff = indexer_params(spec)

    prov = dict(argv=sys.argv, arm=a.arm, spec=spec, npz=os.path.abspath(npz), npz_sha256=sha256(npz),
                driver_sha256=sha256(os.path.abspath(__file__)), dials=dver, host=platform.node(),
                python=sys.version.split()[0], start=a.start, stop=stop, MAXCELL=MAXCELL, MAX_COMB=MAX_COMB, TOL=TOL,
                phil_diff=phil_diff, slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_step=os.environ.get("SLURM_STEP_ID"),
                threads=os.environ.get("OMP_NUM_THREADS"), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if spec["route"] == "indexer":
        prov["dials_samepeaks_sha256"] = sha256(os.path.join(HERE, "dials_samepeaks.py"))
    with open(os.path.abspath(a.out) + ".provenance.json", "w") as f:
        json.dump(prov, f, indent=1)
    print("routes", json.dumps({k: prov[k] for k in ("arm", "npz", "npz_sha256", "driver_sha256", "dials", "host",
                                                    "start", "stop")}), flush=True)
    if phil_diff:
        print("effective phil diff:\n" + phil_diff, flush=True)

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["frame"])
            except Exception:                                            # noqa: BLE001
                pass
    with open(a.out, "a") as fo:
        for i in range(a.start, stop):
            if i in done:
                continue
            q = D["q_%d" % i]                                            # float32, as the NERSC drivers read it
            rec = dict(frame=i, arm=a.arm, route=spec["route"], n=int(len(q)))
            if spec["route"] == "bvs":
                t0 = time.perf_counter()
                try:
                    extra, M = bvs_frame(strategy, q)
                    rec.update(extra)
                    rec.update(ok=M is not None, err=None,
                               M_glint=None if M is None else M.ravel().tolist())
                except Exception as e:                                   # a miss, recorded (as the originals)
                    rec.update(ok=False, err=("%s: %s" % (type(e).__name__, e))[:300])
                rec["t_proc"] = time.perf_counter() - t0
            else:
                qd = np.asarray(q, float)
                tag = "prok_%05d" % i
                expts, strong, geo = build_frame(tag, qd)
                rec.update(geo)
                if geo["rt_err"] > RT_TOL or geo["n_inside"] != geo["n"]:
                    rec.update(ok=False, t_proc=None,
                               err="build: round trip %.2e or %d/%d inside" % (geo["rt_err"], geo["n_inside"], geo["n"]))
                else:
                    p = copy.deepcopy(params)
                    idxr = None
                    t0 = time.perf_counter()
                    try:
                        idxr = Indexer.from_parameters(strong, expts, params=p)
                        idxr.index()
                        rec["t_proc"] = time.perf_counter() - t0
                        rec.update(ok=True, err=None)
                        rec.update(crystal_record(idxr.refined_experiments, idxr.refined_reflections, qd))
                    except Exception as e:                               # a miss, recorded
                        rec["t_proc"] = time.perf_counter() - t0
                        rec.update(ok=False, err=("%s: %s" % (type(e).__name__, e))[:300])
                    mc = getattr(getattr(idxr, "params", p), "max_cell", None)   # Auto -> the estimate DIALS used
                    rec["max_cell_used"] = float(mc) if isinstance(mc, (int, float)) else str(mc)
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
    print("ROUTES_DONE", a.arm, a.start, stop, flush=True)


if __name__ == "__main__":
    main()
