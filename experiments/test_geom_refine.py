"""CPU tests for glint.geom_refine (live geometry refinement). Uses the REAL glint.predict
projector; no GPU, no data files. Run: `python experiments/test_geom_refine.py` or `pytest`."""
import numpy as np
from glint.predict import project_q, predict_spots, panels_from_geom
from glint.geom_refine import GeomRefiner, refine_geometry


def _synth(n=1024):
    geom = {"global": {"clen": 0.10, "coffset": 0.0, "res": 1e4, "data": "/data/data"},
            "wavelength_A": 1.32,
            "panels": {"p0": {"fsx": 1.0, "fsy": 0.0, "ssx": 0.0, "ssy": 1.0,
                              "corner_x": -n / 2, "corner_y": -n / 2, "res": 1e4,
                              "min_fs": 0, "max_fs": n - 1, "min_ss": 0, "max_ss": n - 1}}}
    panels, clen = panels_from_geom(geom)
    return panels, clen, geom["wavelength_A"]


def _rot(seed):
    q = np.random.default_rng(seed).normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _frames(panels, lam, geom_of_f, nfr=90, seed0=100, noise=0.3, spur=0.1):
    """Simulate indexed stills; each frame observed at its own true geometry geom_of_f(f)=(clen,dfs,dss)."""
    base = np.diag([1 / 79.0, 1 / 79.0, 1 / 38.0]); rng = np.random.default_rng(0)
    frames, truth = [], []
    for f in range(nfr):
        cl, dfs, dss = geom_of_f(f)
        R = _rot(seed0 + f) @ base
        pr = predict_spots(R, panels, cl, lam, is_recip=True)
        q = np.stack([pr["h"], pr["k"], pr["l"]], 1).astype(float) @ R
        fs, ss, pan = project_q(q, panels, cl, lam)
        m = np.stack([fs, ss], 1) + np.array([dfs, dss])
        keep = (pan >= 0) & np.isfinite(m).all(1); m = m[keep]
        if len(m) < 6:
            continue
        m = m + rng.normal(0, noise, m.shape)
        ns = max(1, int(spur * len(m)))
        frames.append((R, np.vstack([m, rng.uniform(0, 1024, (ns, 2))])))
        truth.append((cl, dfs, dss))
    return frames, truth


def test_static_recovery():
    """From a wrong geometry (+2% distance, 5/-4 px beam offset), pooled refinement recovers truth."""
    panels, clen, lam = _synth()
    frames, _ = _frames(panels, lam, lambda f: (clen, 5.0, -4.0))        # true beam offset (5,-4)
    gr = refine_geometry(panels, clen * 1.02, lam, frames, passes=6)     # start 2% high in distance
    c = gr.correction()
    assert abs(c["clen_m"] - clen) / clen < 1.5e-3, c                    # distance back to truth
    assert abs(c["dfs"] - 5.0) < 0.4 and abs(c["dss"] + 4.0) < 0.4, c    # beam offset recovered
    return c


def test_drift_tracking():
    """A drifting detector (beam wander + distance ramp): a forgetting tracker follows it live."""
    panels, clen, lam = _synth()
    drift = lambda f: (clen * (1 + 8e-5 * f), 4.0 * np.sin(2 * np.pi * f / 80.0), 0.0)
    frames, truth = _frames(panels, lam, drift, nfr=150)
    gr = GeomRefiner(panels, clen, lam, gamma=0.8, update_every=1, min_frames=3, gate=30.0)
    e_fs, e_cl = [], []
    for i, (R, obs) in enumerate(frames):
        gr.add_frame(R, obs)
        cl_t, dfs_t, _ = truth[i]
        if i >= 20:                                                    # after burn-in
            e_fs.append(abs(gr.dfs - dfs_t))                           # beam-center wander (+-4 px)
            e_cl.append(abs(gr.clen_m - cl_t) / cl_t)                  # distance ramp (~1%)
    mfs, mcl = float(np.mean(e_fs)), float(np.mean(e_cl))
    assert mfs < 0.3, mfs                                              # tracks the wander to <0.3 px
    assert mcl < 5e-4, mcl                                             # tracks the ramp to <0.05%
    return dict(mean_beam_px=round(mfs, 4), mean_dist_frac=round(mcl, 6))


if __name__ == "__main__":
    ok = 0
    for t in (test_static_recovery, test_drift_tracking):
        try:
            r = t(); ok += 1; print(f"PASS  {t.__name__}: {r}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"{ok}/2 passed")
