"""StreamDriver must refuse a frame its ring cannot hold losslessly, not cast it silently.

WHY THIS EXISTS. The driver's resident ring (and the rescue-pixel store) is preallocated at the
constructor's `dtype`, uint16 by default -- sized for raw ADU -- and `push()` copied every frame in
with `ring[slot][...] = xp.asarray(frame)`, which casts without a word. A calibrated frame
(det.calib is float32, pedestal-subtracted, so about half its pixels are negative) lost its
fractions and had its negatives clamped to 0 before the peak finder saw it: on a sigma=1 frame with
6 planted spots the default ring found 93 peaks where a float32 ring found 6, and a BLANK frame
became a 93-peak "hit". float64 and signed-int frames wrapped their negatives to ~65535 instead
(0 of 30 planted spots found, no warning at all). docs/stream_driver_options.md's "Construct, feed,
read" example omits dtype, so following it with calibrated frames hit this directly.

The fix refuses instead of guessing: push() (both the known-cell and the blind warm-up path) and
the rescue-pixel store raise TypeError, before any state changes, when the frame's dtype does not
cast losslessly into the ring's (an integer ring takes numpy-"safe" casts only; a float ring takes
any integer or float frame). The default stays uint16 -- it is what raw-ADU throughput and the
bit-exact benchmarks are measured with -- and a uint16 frame goes through exactly as before.

What is pinned:
  * the default is still uint16, and a float32 / float64 / int16 / int32 / uint32 frame pushed into
    it raises TypeError naming dtype=np.float32 -- known-cell and blind paths -- with no frame
    counted and the driver still usable afterwards;
  * the rescue-pixel store refuses the same, directly and through warmup_batch(rescue_pixels=...);
  * a float32 ring takes float32 frames (6 peaks on the sigma=1 frame, all 6 planted spots) and
    float64 frames (same peaks as the float32 cast of the frame);
  * UNCHANGED: a uint16 frame into the default ring lands bit-for-bit, its peak list is the finder's
    on the frame itself, and uint8/bool frames are still accepted.

  PYTHONPATH=. python experiments/test_ring_dtype.py     # exit 0 = all pass
"""
from __future__ import annotations

import contextlib
import inspect
import os
import sys
import types
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint import stream_driver as sd

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)[:300]}")
    if not ok:
        FAILS.append(name)


@contextlib.contextmanager
def no_torch_needed():
    """A Mc=None driver imports glint.glint_fast lazily (torch); stand in for it during construction,
    as test_cell_registry.py and test_lock_gate_wiring.py do."""
    stub = types.ModuleType("glint.glint_fast")
    stub.index_blind_nbest = lambda q, k: []
    prev = sys.modules.get("glint.glint_fast")
    sys.modules["glint.glint_fast"] = stub
    try:
        yield
    finally:
        if prev is not None:
            sys.modules["glint.glint_fast"] = prev
        else:
            sys.modules.pop("glint.glint_fast", None)


N = 200
PANELS = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=1e4,
               cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5), coffset=0.0,
               min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
M_KC = np.diag([1 / 79.0, 1 / 79.0, 1 / 38.0])
CENTRES = [(30, 40), (60, 150), (100, 100), (140, 60), (170, 170), (50, 90)]


def calib_frame(sigma=1.0, amp=60.0, seed=12345):
    """Pedestal-subtracted calibrated frame: zero-mean Gaussian noise + 6 Gaussian Bragg spots."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, sigma, (N, N)).astype(np.float32)
    yy, xx = np.mgrid[0:N, 0:N]
    for (y, x) in CENTRES:
        img += (amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.2 ** 2))).astype(np.float32)
    return img


def raw_frame(seed=7):
    """Raw-ADU uint16 frame: pedestal ~1000 + noise + the same spots (what the default ring is for)."""
    return np.clip(np.rint(calib_frame(sigma=25.0, amp=1800.0, seed=seed) + 1000.0), 0, 65535).astype(np.uint16)


def driver(Mc=M_KC, **kw):
    kw.setdefault("B", 64); kw.setdefault("use_gpu", False)
    if Mc is None:
        with no_torch_needed():
            return sd.StreamDriver(None, PANELS, 0.1, 1.3, (N, N), **kw)
    return sd.StreamDriver(Mc, PANELS, 0.1, 1.3, (N, N), **kw)


def n_planted(pk):
    fs, ss = np.asarray(pk["x"], float), np.asarray(pk["y"], float)
    return 0 if len(fs) == 0 else sum(int(np.min(np.hypot(fs - x, ss - y)) < 2.0) for (y, x) in CENTRES)


def push_outcome(drv, frame):
    """(exception or None, peaks the finder saw on the ring slot, frames counted)."""
    seen = []
    orig = drv.finder.find
    drv.finder.find = lambda img, *a, **k: seen.append(orig(img, *a, **k)) or seen[-1]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # main's lossy cast warns once; not the signal here
            drv.push(frame)
        exc = None
    except Exception as e:                           # noqa: BLE001 - the exception IS the observation
        exc = e
    finally:
        drv.finder.find = orig
    return exc, (len(seen[-1]["x"]) if seen else None), drv.n_pushed


fr32 = calib_frame()
print(f"constructor default dtype: {inspect.signature(sd.StreamDriver.__init__).parameters['dtype'].default}")
check("the default ring dtype is still uint16 (raw-ADU throughput; not silently changed)",
      inspect.signature(sd.StreamDriver.__init__).parameters["dtype"].default is np.uint16)

# ---- THE DEFECT: a calibrated frame into the default ring -----------------------------------------
print("calibrated frames into the DEFAULT (uint16) ring")
for label, Mc in (("known-cell push()", M_KC), ("blind warm-up push() -> _push_blind", None)):
    drv = driver(Mc)
    n0 = drv.n_pushed
    exc, npk, n1 = push_outcome(drv, fr32)
    check(f"[{label}] float32 frame -> TypeError naming dtype=np.float32",
          isinstance(exc, TypeError) and "dtype=np.float32" in str(exc),
          repr(exc) if exc else f"accepted: the finder saw {npk} peaks (6 planted)")
    check(f"[{label}] ...raised before any state changed (no frame counted)", n1 == n0, (n0, n1))
    exc2, npk2, _ = push_outcome(drv, raw_frame())
    check(f"[{label}] ...and the driver still takes a uint16 frame afterwards", exc2 is None, repr(exc2))

drv = driver()
for dt in (np.float64, np.int16, np.int32, np.uint32):
    # the unsigned frame is non-negative: uint32 is refused for what its TYPE can hold (> 65535), not its values
    fr = (fr32.astype(dt) if np.dtype(dt).kind == "f"
          else np.rint(np.abs(fr32) * 30 if np.dtype(dt).kind == "u" else fr32 * 30).astype(dt))
    exc, npk, _ = push_outcome(drv, fr)
    check(f"{np.dtype(dt).name} frame into the uint16 ring -> TypeError",
          isinstance(exc, TypeError), repr(exc) if exc else f"accepted: the finder saw {npk} peaks")

blank = np.random.default_rng(777).normal(0.0, 1.0, (N, N)).astype(np.float32)
exc, npk, _ = push_outcome(driver(), blank)
check("a BLANK float32 frame is refused, not turned into a many-peak hit",
      isinstance(exc, TypeError), f"accepted: {npk} peaks on a blank (hit threshold 6)")

# ---- the rescue-pixel store --------------------------------------------------------------------
print("rescue-pixel store")
store = sd._PixelStore(2, (N, N), np.uint16, np)
try:
    store.put(0, fr32)
    exc = None
except Exception as e:                               # noqa: BLE001
    exc = e
check("_PixelStore(uint16).put(float32 frame) -> TypeError", isinstance(exc, TypeError),
      repr(exc) if exc else f"stored; min {store.get(0)[0].min()} (input min {fr32.min():.2f})")
frames = np.stack([calib_frame(seed=s) for s in range(3)])
drv = driver(None, rescue_pixels=4, warmup_rescue=True)
try:
    drv.warmup_batch(frames)
    exc = None
except Exception as e:                               # noqa: BLE001
    exc = e
check("warmup_batch(float32 frames, rescue_pixels=4) on the default ring -> TypeError",
      isinstance(exc, TypeError) and "dtype=np.float32" in str(exc),
      repr(exc) if exc else f"stored {len(drv._pix)} frames, truncated")
drv = driver(None, rescue_pixels=4, warmup_rescue=True, dtype=np.float32)
try:
    drv.warmup_batch(frames)
    exc = None
except Exception as e:                               # noqa: BLE001
    exc = e
check("...the same with dtype=np.float32 keeps the picks' pixels exactly",
      exc is None and len(drv._pix) > 0
      and all(np.array_equal(drv._pix.get(ev)[0], frames[ev]) for ev in list(drv._pix._at)),
      repr(exc) if exc else len(drv._pix))

# ---- a float32 ring: what calibrated frames should use ------------------------------------------
print("dtype=np.float32 ring")
exc, npk32, _ = push_outcome(driver(dtype=np.float32), fr32)
d = driver(dtype=np.float32)
push_outcome(d, fr32)
check("float32 frame into a float32 ring: accepted, 6 peaks, all 6 planted spots",
      exc is None and npk32 == 6 and n_planted(d.finder.find(d._ring[0])) == 6, (repr(exc), npk32))
exc64, npk64, _ = push_outcome(driver(dtype=np.float32), fr32.astype(np.float64))
check("float64 frame into a float32 ring: accepted (precision, not data) with the same peaks",
      exc64 is None and npk64 == npk32, (repr(exc64), npk64, npk32))

# ---- UNCHANGED: raw uint16 into the default ring ------------------------------------------------
print("raw uint16 frames into the default ring (unchanged)")
raw = raw_frame()
d = driver()
exc, npk, _ = push_outcome(d, raw)
direct = d.finder.find(raw)
check("a uint16 frame lands bit-for-bit in the ring",
      exc is None and d._ring[0].dtype == np.uint16 and np.array_equal(d._ring[0], raw), repr(exc))
check("...and the finder's peaks on the ring are its peaks on the frame itself",
      exc is None and npk == len(direct["x"]) and np.array_equal(d.finder.find(d._ring[0])["x"], direct["x"]),
      (npk, len(direct["x"])))
for dt in (np.uint8, np.bool_):
    exc, _, _ = push_outcome(driver(), (raw > 1200).astype(dt))
    check(f"a {np.dtype(dt).name} frame is still accepted by the uint16 ring", exc is None, repr(exc))

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
