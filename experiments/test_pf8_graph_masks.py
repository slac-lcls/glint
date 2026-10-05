"""PeakFinder8's CUDA-graph path, reused over frames whose NaN/Inf pixels move: every frame must give
exactly what the eager path gives on that frame.

The graph replays the fp32 background loop on fixed buffers: each find() writes the frame and that
frame's usable-pixel mask (mask AND finite) into them, then replays on a separate stream. If the replay
is not ordered after those writes it can read the previous frame's mask, or an uninitialised one on the
first call (Copilot review on #231). Neither shows on the CPU, so this needs a GPU.

One graph=True finder runs over a sequence of frames, each with its own NaN/Inf pixels, some on peaks.
The sequence runs forwards and then backwards, so every mask differs from the one before it. Each
result is compared with the eager finder (graph=False) and with a fresh graph finder on the same frame.

The fp32 forward sums with atomics, so two eager runs on the same frame already differ in the last
bits (measured here as an A/A control). "Equal" therefore means the same peaks: identical counts,
centroids within 0.01 px, intensity and SNR within a relative tolerance set above that A/A floor. A
negative control replays the graph with the PREVIOUS frame's usable mask, the stale-mask failure
itself, and must differ well beyond the tolerance, so the check can fail.
SKIPS, exit 0, without cupy or a GPU, or if the graph path is unavailable for this geometry.

  PYTHONPATH=. python experiments/test_pf8_graph_masks.py
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except Exception as e:                                   # no cupy, no driver, no device
    print(f"SKIP {os.path.basename(__file__)} -- no usable cupy GPU ({type(e).__name__})")
    sys.exit(0)
import numpy as np
from glint.peakfinder8 import PeakFinder8

FAILS = []


def check(name, ok, detail=""):
    print(("ok    " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


H = W = 384
rng = np.random.default_rng(7)
yy, xx = np.mgrid[0:H, 0:W]
R = np.hypot(yy - H / 2 + 0.3, xx - W / 2 + 0.7)
base = (200.0 * np.exp(-R / 90.0) + rng.normal(0, 3.0, (H, W))).astype(np.float32)
peaks = rng.uniform(20, H - 20, (40, 2))
for py, px in peaks:
    base += (400.0 * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / 2.0)).astype(np.float32)
mask = np.ones((H, W), bool); mask[:, W // 2 - 1:W // 2 + 1] = False          # a masked seam

frames = []
for k in range(8):
    f = base.copy()
    bad = rng.integers(0, H * W, 400)
    f.ravel()[bad[:200]] = np.nan; f.ravel()[bad[200:300]] = np.inf; f.ravel()[bad[300:]] = -np.inf
    py, px = np.rint(peaks[k]).astype(int); f[py, px] = np.nan if k % 2 else np.inf   # one on a peak's centre
    qy, qx = np.rint(peaks[k + 8]).astype(int)
    f[max(qy - 12, 0):qy + 12, max(qx - 12, 0):qx + 12] = np.nan     # a dead block over another peak
    frames.append(f)
frames.append(base.copy())                                                         # clean, after dirty ones

q = cp.asarray(R); m = cp.asarray(mask)
kw = dict(dtype=cp.float32, thr_snr=5.0, min_snr=5.0, min_pix=1, max_pix=200)
eager = PeakFinder8(q, m, graph=False, **kw)
graphed = PeakFinder8(q, m, graph=True, **kw)
if not graphed._graph_want:
    print(f"SKIP {os.path.basename(__file__)} -- the graph path is unavailable for this geometry")
    sys.exit(0)


def host(pk):
    return {k: cp.asnumpy(v) for k, v in pk.items()}


def diff(a, b):
    """-> (same peak list?, max |d xy| px, max rel |d intensity|, max rel |d snr|); peaks sorted by position."""
    if any(len(a[k]) != len(b[k]) for k in a):
        return False, np.inf, np.inf, np.inf
    if not len(a["x"]):
        return True, 0.0, 0.0, 0.0
    ia, ib = np.lexsort((a["x"], a["y"])), np.lexsort((b["x"], b["y"]))
    dxy = float(max(np.max(np.abs(a["x"][ia] - b["x"][ib])), np.max(np.abs(a["y"][ia] - b["y"][ib]))))
    rel = lambda k: float(np.max(np.abs(a[k][ia] - b[k][ib]) / np.maximum(np.abs(b[k][ib]), 1e-6)))  # noqa: E731
    return bool(np.array_equal(a["npix"][ia], b["npix"][ib])), dxy, rel("intensity"), rel("snr")


XY_TOL, REL_TOL = 0.01, 1e-3


def same(a, b):
    ok_n, dxy, di, ds = diff(a, b)
    return (ok_n and dxy <= XY_TOL and di <= REL_TOL and ds <= REL_TOL,
            f"npix same {ok_n}, max|dxy| {dxy:.3g} px, rel dI {di:.3g}, rel dsnr {ds:.3g}")


order = list(range(len(frames))) + list(range(len(frames) - 1, -1, -1))
ref = {i: host(eager.find(cp.asarray(frames[i]))) for i in range(len(frames))}
aa = [diff(host(eager.find(cp.asarray(frames[i]))), ref[i]) for i in range(len(frames))]
print("INFO  A/A, eager twice on each frame: max |dxy| %.3g px, rel dI %.3g, rel dsnr %.3g, npix equal %s"
      % (max(d[1] for d in aa), max(d[2] for d in aa), max(d[3] for d in aa), all(d[0] for d in aa)))
check("A/A eager-vs-eager is inside the tolerance (else the tolerance is below the atomics noise)",
      all(same(host(eager.find(cp.asarray(frames[i]))), ref[i])[0] for i in range(len(frames))))
bad_reuse, bad_fresh, worst, seen = [], [], "", []
for step, i in enumerate(order):
    got = host(graphed.find(cp.asarray(frames[i])))
    ok, why = same(got, ref[i]); seen.append(why)
    if not ok:
        bad_reuse.append((step, i)); worst = why
    if step < len(frames):                                       # a fresh graph finder on the same frame
        ok2, _ = same(host(PeakFinder8(q, m, graph=True, **kw).find(cp.asarray(frames[i]))), ref[i])
        if not ok2:
            bad_fresh.append(i)
check("the graph path was captured", graphed._graph is not None)
check(f"one reused graph finder == eager on every frame ({len(order)} finds, masks change every call)",
      not bad_reuse, f"differing (step, frame): {bad_reuse[:6]} {worst}")
print("INFO  reused graph vs eager, worst step: " + max(seen, key=lambda w: float(w.split("rel dI ")[1].split(",")[0])))
check("a fresh graph finder == eager on every frame", not bad_fresh, bad_fresh)

# negative control: the stale-mask failure itself -- frame i replayed with frame i-1's usable mask
stale = PeakFinder8(q, m, graph=True, **kw)
real_usable = stale._usable
prev = {"g": None}


def stale_usable(Iflat):
    g, Iz = real_usable(Iflat)
    out = (prev["g"] if prev["g"] is not None else g), Iz
    prev["g"] = g
    return out


stale._usable = stale_usable
n_stale = sum(not same(host(stale.find(cp.asarray(frames[i]))), ref[i])[0] for i in range(1, len(frames)))
check(f"negative control: replaying with the previous frame's mask is caught ({n_stale} of {len(frames) - 1} "
      "frames differ beyond the tolerance)", n_stale >= len(frames) - 2, n_stale)
check("the frames are not trivial (eager finds peaks on each)", all(len(ref[i]["x"]) >= 20 for i in ref),
      [len(ref[i]["x"]) for i in ref])

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
