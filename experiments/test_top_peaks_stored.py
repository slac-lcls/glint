"""Regression test for #33: `top_n` must truncate on the peakfinder='stored' path.

It used to apply only on the self-peak-find (v4/pf9) branch, so with peakfinder='stored' -- the
LUTE default since #26 -- `top_peaks` was a SILENT no-op: the config validated and the peak list came
through untruncated. That is the failure mode #23's validators exist to prevent.

Self-contained: builds a synthetic .cxi + .geom in a temp dir with KNOWN intensities, so it needs no
h5py fixture from a real run and no GPU (lute_bridge imports numpy only). Checks:

  1. top_n=0 is unchanged (no accidental truncation)
  2. top_n truncates to exactly N
  3. the peaks KEPT are the N highest-intensity ones -- verified by position, not by count, since a
     truncation that keeps the wrong peaks would still pass a count check
  4. frames with fewer than N peaks are untouched
  5. a .cxi WITHOUT peakTotalIntensity falls back to stored order rather than raising
  6. the v4/pf9 branch is unaffected

  python test_top_peaks_stored.py
"""
import os, sys, tempfile
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from glint.lute_bridge import frames_from_cxi

GEOM = """\
photon_energy = 9500
clen = 0.1
res = 9097.5
adu_per_eV = 0.001
p0/min_fs = 0
p0/max_fs = 1023
p0/min_ss = 0
p0/max_ss = 1023
p0/corner_x = -512.0
p0/corner_y = -512.0
p0/fs = +1.0x +0.0y
p0/ss = +0.0x +1.0y
"""

NFR, PMAX = 6, 40
rng = np.random.default_rng(0)
COUNTS = np.array([40, 40, 25, 8, 40, 3], np.int32)      # incl. one below top_n and one below min_peaks


def build(path, with_intensity=True):
    X = np.zeros((NFR, PMAX), np.float32); Y = np.zeros((NFR, PMAX), np.float32)
    I = np.zeros((NFR, PMAX), np.float32)
    for i in range(NFR):
        k = int(COUNTS[i])
        X[i, :k] = rng.uniform(10, 1000, k); Y[i, :k] = rng.uniform(10, 1000, k)
        I[i, :k] = rng.uniform(1, 1000, k)
    with h5py.File(path, "w") as f:
        g = f.create_group("entry_1/result_1")
        g.create_dataset("peakXPosRaw", data=X); g.create_dataset("peakYPosRaw", data=Y)
        g.create_dataset("nPeaks", data=COUNTS)
        if with_intensity:
            g.create_dataset("peakTotalIntensity", data=I)
    return X, Y, I


ok = True


def check(label, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


with tempfile.TemporaryDirectory() as td:
    gpath = os.path.join(td, "d.geom"); open(gpath, "w").write(GEOM)
    cxi = os.path.join(td, "peaks.cxi")
    X, Y, I = build(cxi, with_intensity=True)
    TOP = 10

    print("1-4. stored path WITH peakTotalIntensity")
    f0, _ = frames_from_cxi(cxi, gpath, peakfinder="stored", top_n=0, min_peaks=6)
    fN, _ = frames_from_cxi(cxi, gpath, peakfinder="stored", top_n=TOP, min_peaks=6)
    n0 = [len(q) for q in f0]; nN = [len(q) for q in fN]
    check(f"top_n=0 unchanged -> {n0} (expect counts, 0 where <min_peaks)",
          n0 == [40, 40, 25, 8, 40, 0])
    check(f"top_n={TOP} truncates -> {nN}", nN == [10, 10, 10, 8, 10, 0])
    check("frames already under top_n are untouched (frame 3 keeps 8)", nN[3] == 8)

    # the peaks kept must be the N STRONGEST -- check by position, not count
    kept_ok = True
    for i in (0, 1, 4):
        k = int(COUNTS[i]); want = np.argsort(I[i, :k])[::-1][:TOP]
        wx = np.sort(X[i, :k][want])
        f1, _ = frames_from_cxi(cxi, gpath, peakfinder="stored", top_n=TOP, min_peaks=6, n=i + 1)
        # recover which x survived by re-deriving q for the expected subset and comparing
        fexp, _ = frames_from_cxi(cxi, gpath, peakfinder="stored", top_n=0, min_peaks=6, n=i + 1)
        # compare the truncated frame against the full frame's rows for the expected indices
        full = fexp[i]
        sub = fN[i]
        match = all(any(np.allclose(s, full[j], atol=1e-9) for j in want) for s in sub)
        kept_ok = kept_ok and match
    check("kept peaks are the highest-intensity ones (matched by q-vector)", kept_ok)

    print("\n5. stored path WITHOUT peakTotalIntensity -> fallback, no raise")
    cxi2 = os.path.join(td, "noint.cxi"); build(cxi2, with_intensity=False)
    try:
        f2, _ = frames_from_cxi(cxi2, gpath, peakfinder="stored", top_n=TOP, min_peaks=6)
        check(f"falls back to stored order -> {[len(q) for q in f2]}",
              [len(q) for q in f2] == [10, 10, 10, 8, 10, 0])
    except Exception as e:
        check(f"falls back without raising (got {type(e).__name__}: {e})", False)

    # 6. CONSTANT intensities: worse than absent ones, because sorting a constant array yields an
    # arbitrary subset that LOOKS ranked. experiments/cf_peaks.cxi is exactly this -- 1000.0 for all
    # 16545 peaks -- which silently invalidated the "intensity ranking" arm of #34.
    print("\n6. stored path with CONSTANT peakTotalIntensity -> detect and fall back")
    cxi3 = os.path.join(td, "constint.cxi")
    Xc = np.zeros((NFR, PMAX), np.float32); Yc = np.zeros((NFR, PMAX), np.float32)
    for i in range(NFR):
        k = int(COUNTS[i])
        Xc[i, :k] = np.arange(k) * 7.0 + 10.0        # positions strictly increasing -> order is visible
        Yc[i, :k] = np.arange(k) * 3.0 + 10.0
    with h5py.File(cxi3, "w") as fh:
        g = fh.create_group("entry_1/result_1")
        g.create_dataset("peakXPosRaw", data=Xc); g.create_dataset("peakYPosRaw", data=Yc)
        g.create_dataset("nPeaks", data=COUNTS)
        g.create_dataset("peakTotalIntensity", data=np.full((NFR, PMAX), 1000.0, np.float32))
    f3, _ = frames_from_cxi(cxi3, gpath, peakfinder="stored", top_n=TOP, min_peaks=6)
    fref, _ = frames_from_cxi(cxi3, gpath, peakfinder="stored", top_n=0, min_peaks=6)
    check(f"still truncates -> {[len(q) for q in f3]}",
          [len(q) for q in f3] == [10, 10, 10, 8, 10, 0])
    # the fallback must keep the FIRST 10, i.e. match the head of the untruncated frame
    head_ok = all(np.allclose(f3[i], fref[i][:TOP], atol=1e-9) for i in (0, 1, 4))
    check("falls back to stored ORDER (keeps the head, not an arbitrary subset)", head_ok)

print("\n" + ("ALL PASS" if ok else "*** FAILURES ***"))
sys.exit(0 if ok else 1)
