"""Speed and agreement of the profile-fit port on REAL frames, against laue-dials itself.

Indexes the first N frames of a run, predicts at the benchmark's gate, then integrates the same
predictions with: laue-dials' numpy Integrator, driven per image as laue.integrate does
(LAUE_DIALS_INTEGRATION must point at its integration.py); the torch port on CPU; the port on
the GPU in float64 and float32; and GLINT's box integrator. Reports ms per frame and, for the
port, the worst relative disagreement with laue-dials over the reflections both integrated.

    python bench_speed.py --images RUN.h5 --geom A.geom -N 64 --adu-per-photon 10
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np


def load_ld():
    spec = importlib.util.spec_from_file_location("ld", os.environ["LAUE_DIALS_INTEGRATION"])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ld_frame(ld, img, pred, panels, radius=None, k=5, isigi=3.0, maxiter=2):
    """laue.integrate's integrate_image on one GLINT frame (no mask). Returns (I, sigma)."""
    from glint.profile_integrate import lab_pixel_coords
    n = len(pred)
    I = np.zeros(n); U = np.zeros(n)
    pix = [img[p["min_ss"]:p["max_ss"] + 1, p["min_fs"]:p["max_fs"] + 1] for p in panels]
    pid = pred["panel"].astype(int)
    mn_fs = np.array([panels[i]["min_fs"] for i in pid]); mn_ss = np.array([panels[i]["min_ss"] for i in pid])
    spots = np.stack([pred["fs"] + 0.5 - mn_fs, pred["ss"] + 0.5 - mn_ss], 1)
    nbc = lab_pixel_coords(pred, panels)
    keep = np.arange(n)
    R = radius if radius is not None else ld.estimate_integration_radius(nbc)
    kw = dict(panel_ids=pid, neighbor_coords=nbc, radius=R, k=k, isigi_cutoff=isigi)
    integ = ld.Integrator(pix, spots, **kw)
    enough = integ.window_valid.sum(-1) >= max(6, 0.25 * integ.m)
    if not enough.all():
        keep, spots, pid, nbc = keep[enough], spots[enough], pid[enough], nbc[enough]
        kw.update(panel_ids=pid, neighbor_coords=nbc)
        integ = ld.Integrator(pix, spots, **kw)
    try:
        integ.fit(maxiter=maxiter)
    except RuntimeError:
        return I, U
    I[keep] = integ.intensity; U[keep] = integ.uncertainty
    return I, U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--geom", required=True)
    ap.add_argument("-N", type=int, default=64)
    ap.add_argument("--adu-per-photon", type=float, default=10.0)
    ap.add_argument("--cell", nargs=6, type=float, default=[79.2, 79.2, 37.9, 90, 90, 90])
    ap.add_argument("--tol", type=float, default=0.002)
    ap.add_argument("--dmin", type=float, default=1.9)
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    import h5py
    import torch
    from glint.lute_bridge import frames_from_cxi, parse_geom, lambda_from_eV
    from glint.lattice import cell_to_Ar
    from glint.hybrid_stream import hybrid_index, gate_results
    from glint.predict import predict_spots, integrate_spots, _canonical_axes
    from glint.profile_integrate import profile_integrate_frames

    frames, images = frames_from_cxi(a.images, a.geom, n=a.N, min_peaks=10, sig_floor=10, abs_thr=100)
    res, _ = hybrid_index(frames, images, Mc_known=cell_to_Ar(*a.cell), nbest=3)
    gate_results(res, frames, "strict")
    panels, glob = parse_geom(a.geom)
    f = h5py.File(a.images, "r")
    d = f["/entry_1/data_1/data"]
    imgs, preds = [], []
    for r in res:
        if r.get("M") is None:
            continue
        ev = int(r["event"])
        lam = lambda_from_eV(float(f["/LCLS/photon_energy_eV"][ev]))
        M = _canonical_axes(r["M"])
        preds.append(predict_spots(M, panels, float(glob["clen"]), lam, dmin=a.dmin, tol=a.tol))
        imgs.append(np.asarray(d[ev], np.float32) / a.adu_per_photon)
    nf = len(imgs)
    out = {"n_frames": nf, "refl_per_frame": float(np.mean([len(p) for p in preds]))}

    t = time.time(); box = [integrate_spots(i, p) for i, p in zip(imgs, preds)]
    out["box_cpu_ms"] = 1e3 * (time.time() - t) / nf
    ld = load_ld()
    t = time.time(); ref = [ld_frame(ld, i.astype(np.float64), p, panels) for i, p in zip(imgs, preds)]
    out["laue_dials_numpy_ms"] = 1e3 * (time.time() - t) / nf
    runs = [("port_cpu_f64", "cpu", torch.float64)]
    if torch.cuda.is_available():
        runs += [("port_gpu_f64", "cuda", torch.float64), ("port_gpu_f32", "cuda", torch.float32)]
    for name, dev, dt in runs:
        profile_integrate_frames(imgs[:2], preds[:2], panels, device=dev, dtype=dt)     # warm-up
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        got = profile_integrate_frames(imgs, preds, panels, device=dev, dtype=dt, batch=32)
        if dev == "cuda":
            torch.cuda.synchronize()
        out[name + "_ms"] = 1e3 * (time.time() - t) / nf
        both = [(g[1] > 0) & (r[1] > 0) for g, r in zip(got, ref)]
        same = sum(int(np.sum((g[1] > 0) != (r[1] > 0))) for g, r in zip(got, ref))
        worst = max(float(np.max(np.abs(g[0][m] - r[0][m]) / np.maximum(r[1][m], 1e-12), initial=0))
                    for g, r, m in zip(got, ref, both))
        out[name + "_max_dI_over_sigma_vs_laue_dials"] = worst
        out[name + "_kept_set_differences"] = same
    out["refl_integrated_box"] = int(sum(int((b[1] > 0).sum()) for b in box))
    out["refl_integrated_profile"] = int(sum(int((r[1] > 0).sum()) for r in ref))
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
