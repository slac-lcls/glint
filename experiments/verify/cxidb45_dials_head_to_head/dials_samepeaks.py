"""Driver: dials.stills_process on the SAME 170-peak cxidb-45 lists GLINT is scored on.

Run with the DIALS 3.dev.1332 build's interpreter (`dials.python`, /sdf/group/lcls/ds/tools/cctbx-psana2/build).

What runs is dials.stills_process's own Processor.process_experiments (pre_process -> hit finder ->
index -> refine; integrate off), with its own phil scope and program defaults. ONE thing is replaced:
Processor.find_spots returns the frame's 170 peaks from experiments/prok_q.npz (GLINT 95d51ce) instead of
running DIALS spot finding on pixels. The raw SACLA h5 images are on NERSC only, and the question here is
indexing on identical peaks, so no pixels are needed.

How the q-vectors become DIALS spots. GLINT's q = (s_hat - z_hat)/lambda (beam along +z,
glint/geom.py:_q_from_panels). Each frame's lambda is recovered from its own peaks by least squares on the
Ewald condition |q|^2 + 2 q_z / lambda = 0 (per-peak spread ~1e-7 relative on these frames). s1 = q + s0 is
rotated into the DIALS lab frame (beam along -z) by R = diag(-1, 1, -1) (180 deg about y, det +1, so the
hand is kept) and intersected with a virtual flat PAD panel (55 mm, 0.05 mm pixels, 3000 x 3000, beam at
the centre, linear px<->mm), i.e. the real MPCCD's pixel size and camera length. DIALS's own
centroid_px_to_mm + map_centroids_to_reciprocal_space is then run on the table and must reproduce R q to
RT_TOL per frame, or the frame is recorded as a build failure and not given to DIALS.

Output: one JSON line per frame (resumable: frames already in --out are skipped). The crystal is stored
as M_glint = R @ [a b c] (real-space basis, columns, GLINT lab frame), so q_glint @ M_glint = hkl, exactly
GLINT's convention; stills_score.py applies GLINT's own gate to it.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
from dxtbx.model import BeamFactory, DetectorFactory
from dxtbx.model.experiment_list import ExperimentListFactory
from libtbx.phil import parse

from dials.array_family import flex
from dials.command_line.stills_process import Processor, phil_scope

NX = NY = 3000
PX = 0.05              # mm, MPCCD pixel
DIST = 55.0            # mm, mpccd-optimized.geom clen 0.055 m
VAR = 1.0 / 12.0       # px^2: centroid variance of an integer-pixel local maximum (peakfind 5x5 maxima)
RT_TOL = 2e-5          # 1/A, round-trip tolerance (float32 q, |q| <= 0.82)
R = np.diag([-1.0, 1.0, -1.0])


def wavelength_from_q(q):
    q2 = (q * q).sum(1)
    qz = q[:, 2]
    k = -(qz * q2).sum() / (2.0 * (qz * qz).sum())       # k = 1/lambda
    rel = np.abs(-q2 / (2.0 * qz) - k) / k
    return 1.0 / k, k, float(np.percentile(rel, 90))


def build_frame(tag, q):
    lam, k, lam_spread = wavelength_from_q(q)
    s1d = (q + np.array([0.0, 0.0, k])) @ R.T
    beam = BeamFactory.simple(lam)                         # s0 = (0, 0, -1/lambda)
    det = DetectorFactory.simple("PAD", DIST, (NX * PX / 2.0, NY * PX / 2.0), "+x", "-y",
                                 (PX, PX), (NX, NY), (-1.0, 1.0e9))
    p = det[0]
    xy = np.array([p.millimeter_to_pixel(p.get_ray_intersection(tuple(map(float, v)))) for v in s1d])
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < NX) & (xy[:, 1] >= 0) & (xy[:, 1] < NY)
    d = {"__id__": "ExperimentList",
         "experiment": [{"__id__": "Experiment", "identifier": tag, "beam": 0, "detector": 0, "imageset": 0}],
         "beam": [beam.to_dict()], "detector": [det.to_dict()],
         "imageset": [{"__id__": "ImageSet", "images": ["/nonexistent/cxidb45/%s.h5" % tag]}]}
    expts = ExperimentListFactory.from_dict(d, check_format=False)
    n = len(xy)
    r = flex.reflection_table()
    r["xyzobs.px.value"] = flex.vec3_double([(float(x), float(y), 0.0) for x, y in xy])
    r["xyzobs.px.variance"] = flex.vec3_double(n, (VAR, VAR, VAR))
    r["panel"] = flex.size_t(n, 0)
    r["id"] = flex.int(n, 0)
    r["bbox"] = flex.int6([(int(x) - 2, int(x) + 3, int(y) - 2, int(y) + 3, 0, 1) for x, y in xy])
    r["intensity.sum.value"] = flex.double([float(n - j) for j in range(n)])   # rank placeholder; unused by indexing
    r["intensity.sum.variance"] = flex.double(n, 1.0)
    r["n_signal"] = flex.int(n, 25)
    r.set_flags(flex.bool(n, True), r.flags.strong)
    r.experiment_identifiers()[0] = tag
    chk = r.copy()
    chk.centroid_px_to_mm(expts)
    chk.map_centroids_to_reciprocal_space(expts)
    rt_err = float(np.abs(np.array(chk["rlp"]) - q @ R.T).max())
    return expts, r, dict(lam=float(lam), lam_spread_p90=lam_spread, rt_err=rt_err,
                          n_inside=int(inside.sum()), n=n)


class SamePeaksProcessor(Processor):
    """stills_process's Processor with find_spots replaced by the supplied table; index() is timed."""

    def __init__(self, params, rank):
        super().__init__(params, composite_tag=None, rank=rank)
        self.reset(None)

    def reset(self, strong):
        self.strong, self.result, self.err, self.t_index = strong, None, None, None

    def find_spots(self, experiments):
        return self.strong

    def index(self, experiments, reflections):
        t0 = time.perf_counter()
        try:
            out = super().index(experiments, reflections)
        except Exception as e:
            self.t_index = time.perf_counter() - t0
            self.err = ("%s: %s" % (type(e).__name__, e))[:300]
            raise
        self.t_index = time.perf_counter() - t0
        self.result = out
        return out


def crystal_record(experiments, indexed, q):
    c = experiments[0].crystal
    a, b, cc = (np.array(v, float) for v in c.get_real_space_vectors())
    Mg = R @ np.column_stack([a, b, cc])
    H = q @ Mg
    m015 = int((np.abs(H - np.rint(H)).max(1) < 0.15).sum())
    sel = indexed["id"] == 0
    ix = indexed.select(sel)
    rmsd = None
    if len(ix) and "xyzcal.mm" in ix and "xyzobs.mm.value" in ix:
        dx = np.array(ix["xyzcal.mm"]) - np.array(ix["xyzobs.mm.value"])
        rmsd = float(np.sqrt((dx[:, :2] ** 2).sum(1).mean()))
    return dict(M_glint=Mg.ravel().tolist(), cell=list(c.get_unit_cell().parameters()),
                sg=str(c.get_space_group().info()), n_indexed=int(sel.count(True)),
                m015_driver=m015, rmsd_mm=rmsd, n_lattices=len(experiments))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--phil", nargs="+", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--rank", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.workdir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sources = [parse(open(f).read()) for f in a.phil]
    working = phil_scope.fetch(sources=sources)
    params = working.extract()
    params.output.output_dir = a.workdir
    diff = phil_scope.fetch_diff(source=working).as_str()
    D = np.load(a.npz)
    n = int(D["n"])
    stop = n if a.stop is None else min(a.stop, n)
    prov = dict(argv=sys.argv, arm=a.arm, npz=os.path.abspath(a.npz), npz_sha256=sha256(a.npz),
                phil_files=[os.path.abspath(f) for f in a.phil], phil_diff=diff, host=platform.node(),
                python=sys.version.split()[0], start=a.start, stop=stop, NX=NX, PX=PX, DIST=DIST, VAR=VAR,
                RT_TOL=RT_TOL, workdir=os.path.abspath(a.workdir), out=os.path.abspath(a.out), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(os.path.join(a.workdir, "provenance_%d.json" % a.rank), "w") as f:
        json.dump(prov, f, indent=1)
    print("dials_samepeaks", json.dumps({k: prov[k] for k in ("arm", "npz", "npz_sha256", "start", "stop", "host")}))
    print("effective phil diff:\n" + diff, flush=True)

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["frame"])
            except Exception:
                pass
    proc = SamePeaksProcessor(params, rank=a.rank)
    with open(a.out, "a") as fo:
        for i in range(a.start, stop):
            if i in done:
                continue
            q = np.asarray(D["q_%d" % i], float)
            tag = "prok_%05d" % i
            t0 = time.perf_counter()
            expts, strong, geo = build_frame(tag, q)
            t_build = time.perf_counter() - t0
            rec = dict(frame=i, arm=a.arm, t_build=t_build, **geo)
            if geo["rt_err"] > RT_TOL or geo["n_inside"] != geo["n"]:
                rec.update(ok=False, err="build: round trip %.2e or %d/%d inside" % (geo["rt_err"], geo["n_inside"], geo["n"]),
                           t_proc=None, t_index=None)
            else:
                proc.reset(strong)
                t1 = time.perf_counter()
                try:
                    proc.process_experiments(tag, expts)
                except Exception as e:                       # squash_errors=True should catch everything
                    proc.err = proc.err or ("process: %s: %s" % (type(e).__name__, e))[:300]
                rec.update(t_proc=time.perf_counter() - t1, t_index=proc.t_index, err=proc.err,
                           ok=proc.result is not None)
                if proc.result is not None:
                    rec.update(crystal_record(proc.result[0], proc.result[1], q))
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
    print("DIALS_DONE", a.arm, a.start, stop, flush=True)


if __name__ == "__main__":
    main()
