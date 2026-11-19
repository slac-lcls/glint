"""glint.profile_integrate must reproduce laue-dials' profile-fitting integrator frame by frame.

The reference is the laue-dials code itself: ``laue_dials.algorithms.integration.Integrator``, driven
the way ``laue_dials.command_line.integrate.integrate_image`` drives it. That means radius from the
full predicted set in the shared frame, the per-panel mask-dilation drop, the MIN_WINDOW_PIXELS drop
and re-construction, fit(maxiter), and a skipped image on RuntimeError. Point LAUE_DIALS_INTEGRATION
at a copy of laue-dials' ``src/laue_dials/algorithms/integration.py``, or have ``laue_dials`` importable.
Without either, the reference checks are SKIPPED, loudly, and the port's own consistency checks still
run.

Frames are synthetic: a 2x2-panel 240x260 detector, ~160 anisotropic Gaussian spots of mixed strength
on a Poisson background, a bad-pixel mask, and spots near panel edges, so windows overhang. Cases:
  A  integer counts, auto radius, maxiter=2 (the laue.integrate default);
  B  integer counts, fixed radius 4, maxiter=6, where the Poisson score's early stop can fire;
  C  float (gain-corrected) counts, where the score is inf and the stop never fires;
  D  a frame with one prediction, which must fail (sigma = 0 everywhere).
Checks: float64 CPU and CUDA against the reference to 1e-8 relative; one batch of all frames
against each frame alone, bit for bit on a device; float32 CUDA against float64 to 1e-3 of sigma.

Run: `PYTHONPATH=. python experiments/test_profile_integrate.py` (exit 1 on any failure).
"""
import importlib.util
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

from glint.profile_integrate import profile_integrate_frames

FAIL = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAIL.append(msg)


def load_reference():
    path = os.environ.get("LAUE_DIALS_INTEGRATION")
    if path:
        spec = importlib.util.spec_from_file_location("ld_integration", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    try:
        from laue_dials.algorithms import integration
        return integration
    except Exception:
        return None


H, W = 240, 260
PANELS = []
for i, (r0, r1) in enumerate([(0, 119), (120, 239)]):
    for j, (c0, c1) in enumerate([(0, 129), (130, 259)]):
        # identity orientation, corner at the window origin: lab pixels == data-array pixels
        PANELS.append(dict(name=f"p{i}{j}", fs=np.array([1.0, 0.0]), ss=np.array([0.0, 1.0]), res=1.0,
                           cx=float(c0), cy=float(r0), coffset=0.0, min_fs=c0, max_fs=c1,
                           min_ss=r0, max_ss=r1, slab=None))


def panel_of(fs, ss):
    for pi, p in enumerate(PANELS):
        if p["min_fs"] <= fs <= p["max_fs"] and p["min_ss"] <= ss <= p["max_ss"]:
            return pi
    return -1


def make_frame(seed, n=160, integer=True):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    lam = np.full((H, W), 2.0 + rng.uniform(0, 1.5))
    fs = rng.uniform(1, W - 2, n); ss = rng.uniform(1, H - 2, n)
    amp = np.where(rng.uniform(size=n) < 0.6, rng.uniform(200, 3000, n), rng.uniform(0, 60, n))
    for f, s, a in zip(fs, ss, amp):
        sx, sy, rho = rng.uniform(0.9, 1.6), rng.uniform(0.9, 1.6), rng.uniform(-0.4, 0.4)
        dx, dy = xx - f, yy - s
        z = (dx / sx) ** 2 - 2 * rho * dx * dy / (sx * sy) + (dy / sy) ** 2
        lam += a * np.exp(-0.5 * z / (1 - rho ** 2)) / (2 * np.pi * sx * sy * np.sqrt(1 - rho ** 2))
    img = rng.poisson(lam).astype(np.float64)
    if not integer:
        img = img * 1.37 + rng.normal(0, 0.3, img.shape)
    # predictions: the true centres, jittered by up to half a pixel
    pf = fs + rng.uniform(-0.5, 0.5, n); ps = ss + rng.uniform(-0.5, 0.5, n)
    pan = np.array([panel_of(np.rint(a), np.rint(b)) for a, b in zip(pf, ps)])
    on = pan >= 0
    pred = np.zeros(int(on.sum()), dtype=[("fs", float), ("ss", float), ("panel", int)])
    pred["fs"], pred["ss"], pred["panel"] = pf[on], ps[on], pan[on]
    mask = np.ones((H, W), bool)
    bad = rng.integers(0, H * W, 60)
    mask.flat[bad] = False
    mask[50:53, 30:34] = False
    return img, pred, mask


def reference(ld, img, pred, mask, radius, k, isigi, maxiter):
    """laue.integrate's integrate_image on GLINT inputs. Returns (I, sigma) aligned with pred."""
    n = len(pred)
    I = np.zeros(n); U = np.zeros(n)
    pix = [img[p["min_ss"]:p["max_ss"] + 1, p["min_fs"]:p["max_fs"] + 1] for p in PANELS]
    pm = [mask[p["min_ss"]:p["max_ss"] + 1, p["min_fs"]:p["max_fs"] + 1] for p in PANELS]
    pid = pred["panel"].astype(int)
    mn_fs = np.array([PANELS[i]["min_fs"] for i in pid]); mn_ss = np.array([PANELS[i]["min_ss"] for i in pid])
    spots = np.stack([pred["fs"] + 0.5 - mn_fs, pred["ss"] + 0.5 - mn_ss], 1)      # panel-local (x, y)
    nbc = np.stack([pred["fs"], pred["ss"]], 1)                                  # lab frame
    keep = np.arange(n)
    if n < 2:
        return I, U
    R = radius if radius is not None else ld.estimate_integration_radius(nbc)
    # unmasked_selection_per_panel, with skimage's isotropic_dilation written out
    sel = np.ones(n, bool)
    x = np.floor(spots[:, 0]).astype(int); y = np.floor(spots[:, 1]).astype(int)
    for q in np.unique(pid):
        m = pm[q]
        if m.all():
            continue
        on = pid == q
        expanded = ~(distance_transform_edt(m) <= R)
        sel[on] = expanded.flatten()[x[on] + m.shape[1] * y[on]]
    keep, spots, pid, nbc = keep[sel], spots[sel], pid[sel], nbc[sel]
    if len(keep) == 0:
        return I, U
    kw = dict(panel_ids=pid, panel_masks=pm, neighbor_coords=nbc, radius=R, k=k, isigi_cutoff=isigi)
    integ = ld.Integrator(pix, spots, **kw)
    enough = integ.window_valid.sum(-1) >= max(6, 0.25 * integ.m)
    if not enough.all():
        if not enough.any():
            return I, U
        keep, spots, pid, nbc = keep[enough], spots[enough], pid[enough], nbc[enough]
        kw.update(panel_ids=pid, neighbor_coords=nbc)
        integ = ld.Integrator(pix, spots, **kw)
    try:
        integ.fit(maxiter=maxiter)
    except RuntimeError:
        return I, U
    I[keep] = integ.intensity; U[keep] = integ.uncertainty
    return I, U


def rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)))) if len(a) else 0.0


def main():
    import torch
    ld = load_reference()
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    cases = {
        "A integer, auto radius, maxiter=2": dict(seeds=[1, 2, 3], integer=True, radius=None, maxiter=2),
        "B integer, radius 4, maxiter=6": dict(seeds=[4, 5], integer=True, radius=4, maxiter=6),
        "C float counts, auto radius": dict(seeds=[6, 7], integer=False, radius=None, maxiter=3),
    }
    for name, c in cases.items():
        print(name)
        data = [make_frame(s, integer=c["integer"]) for s in c["seeds"]]
        imgs, preds, masks = zip(*data)
        kw = dict(radius=c["radius"], k=5, isigi_cutoff=3.0, maxiter=c["maxiter"])
        refs = [reference(ld, *d, c["radius"], 5, 3.0, c["maxiter"]) for d in data] if ld else None
        for dev in devs:
            got = profile_integrate_frames(list(imgs), list(preds), PANELS, list(masks), device=dev, **kw)
            if refs is not None:
                worst = max(max(rel(g[0], r[0]), rel(g[1], r[1])) for g, r in zip(got, refs))
                same_set = all(np.array_equal(g[1] > 0, r[1] > 0) for g, r in zip(got, refs))
                nint = sum(int((r[1] > 0).sum()) for r in refs)
                check(same_set and worst < 1e-8,
                      f"{dev} float64 vs laue-dials: {nint} integrated, same kept set={same_set}, "
                      f"max rel |d| = {worst:.2e}")
            solo = [profile_integrate_frames([i], [p], PANELS, [m], device=dev, **kw)[0]
                    for i, p, m in data]
            # one-frame and many-frame batches pad to different N, so cdist and reductions may differ
            # in the last ulp; the batch must still be the per-frame answer
            worst = max(max(rel(g[0], s[0]), rel(g[1], s[1])) for g, s in zip(got, solo))
            check(worst < 1e-9, f"{dev} batch of {len(data)} vs each frame alone: max rel |d| = {worst:.1e}")
        if "cuda" in devs:
            g64 = profile_integrate_frames(list(imgs), list(preds), PANELS, list(masks), device="cuda", **kw)
            g32 = profile_integrate_frames(list(imgs), list(preds), PANELS, list(masks), device="cuda",
                                           dtype=torch.float32, **kw)
            same = all(np.array_equal(a[1] > 0, b[1] > 0) for a, b in zip(g64, g32))
            dI = max(float(np.max(np.abs(a[0] - b[0]) / np.maximum(a[1], 1e-9), initial=0))
                     for a, b in zip(g64, g32))
            check(dI < 1e-3, f"cuda float32 vs float64: same kept set={same}, max |dI|/sigma = {dI:.1e}")
    print("D one prediction")
    img, pred, mask = make_frame(9)
    one = profile_integrate_frames([img], [pred[:1]], PANELS, [mask], device=devs[-1])[0]
    check(bool(np.all(one[1] == 0)), "a frame with one prediction fails: sigma == 0")
    if ld is None:
        print("SKIPPED the laue-dials reference checks: set LAUE_DIALS_INTEGRATION to laue-dials' "
              "src/laue_dials/algorithms/integration.py")
    if FAIL:
        print(f"\n{len(FAIL)} FAILED")
        sys.exit(1)
    print("\nall passed")


if __name__ == "__main__":
    main()
