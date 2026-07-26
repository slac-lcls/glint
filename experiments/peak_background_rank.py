"""Does background-aware peak ranking beat raw intensity? (follow-up to #34 / #36)

#34 found that ranking peaks by `peakTotalIntensity` does NOT beat plain peakfinder8 output order.
One explanation: peakTotalIntensity is RAW. On a water ring the diffuse background is high, so a
spurious peak sitting on the ring can carry large total intensity while being statistically
insignificant -- which would make intensity a background-CONTAMINATED ranking and explain why it
underperforms. That is the "over-finding on water rings" `top_peaks` was written to guard against.

Step 1 is a diagnostic, not a ranking: do THESE frames even show ring contamination? If the radial
peak distribution is smooth, this corpus cannot test the idea and any ranking result would be noise
dressed as a finding.

  A. radial histogram of peak |q| -- a water ring shows as an excess at fixed |q|
     (water's main diffuse ring is ~3.2 A, i.e. |q| ~ 0.31 1/A; a second near ~1.9 A)
  B. median peak intensity vs |q| -- ring peaks ride a raised background
  C. does the intensity RANK correlate with |q|? if the strongest peaks cluster at one radius,
     intensity is selecting a ring rather than the best reflections

Step 2 only runs if step 1 finds something: rank by intensity RELATIVE to the radial median
(a background-normalised score computable from what the file actually carries) and compare against
raw intensity / output order / inner on the same frames.

  python peak_background_rank.py
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from glint.lute_bridge import parse_geom, peaks_to_q, lambda_from_eV, _meta

CXI = os.path.join(HERE, "cf_peaks.cxi")
GEOM = os.path.join(HERE, "cf.geom")

panels, glob = parse_geom(GEOM)
with h5py.File(CXI, "r") as f:
    npk = np.asarray(f["/entry_1/result_1/nPeaks"])
    X = np.asarray(f["/entry_1/result_1/peakXPosRaw"])
    Y = np.asarray(f["/entry_1/result_1/peakYPosRaw"])
    I = np.asarray(f["/entry_1/result_1/peakTotalIntensity"])
    clen_spec, en_spec = glob.get("clen"), glob.get("photon_energy")
    coff = float(glob.get("coffset", 0.0))
    QS, IS, FR = [], [], []
    for i in range(len(npk)):
        k = int(npk[i])
        if k < 6:
            continue
        clen = _meta(clen_spec, f, i, 0.1)
        clen = clen * (0.001 if abs(clen) > 10 else 1.0) + coff
        eV = _meta(en_spec, f, i, None)
        wl = lambda_from_eV(eV) if eV else 1.3
        q = peaks_to_q(X[i, :k], Y[i, :k], panels, clen, wl)
        good = np.isfinite(q).all(1)
        QS.append(np.linalg.norm(q[good], axis=1)); IS.append(I[i, :k][good]); FR.append(i)

qn = np.concatenate(QS); iv = np.concatenate(IS)
print(f"{len(FR)} frames, {len(qn)} peaks total, |q| range {qn.min():.4f}-{qn.max():.4f} 1/A "
      f"(d = {1/max(qn.max(),1e-9):.2f}-{1/max(qn.min(),1e-9):.1f} A)\n")

# ---- A. radial histogram: is there a ring? ---------------------------------------------------
print("A. radial peak-count distribution (a water ring = excess at fixed |q|)")
nb = 24
edges = np.linspace(qn.min(), qn.max(), nb + 1)
cnt, _ = np.histogram(qn, bins=edges)
mid = 0.5 * (edges[:-1] + edges[1:])
# expected count if peaks were uniform in RECIPROCAL AREA (shell area ~ q), the null for "no ring"
shell = mid / mid.sum()
exp = shell * cnt.sum()
peak_bin = int(np.argmax(cnt / np.maximum(exp, 1e-9)))
print(f"{'|q| 1/A':>9}{'d (A)':>8}{'peaks':>7}{'vs shell-area null':>20}")
for b in range(nb):
    r = cnt[b] / max(exp[b], 1e-9)
    bar = "#" * int(min(r, 6) * 6)
    star = "  <-- max excess" if b == peak_bin else ""
    print(f"{mid[b]:9.3f}{1/max(mid[b],1e-9):8.2f}{cnt[b]:7d}   {r:5.2f}x {bar}{star}")

# ---- B. does intensity ride the background? --------------------------------------------------
print("\nB. median peak intensity vs |q| (ring peaks ride a raised background)")
print(f"{'|q| 1/A':>9}{'d (A)':>8}{'n':>7}{'median I':>11}{'p90 I':>11}")
for b in range(0, nb, 3):
    m = (qn >= edges[b]) & (qn < edges[min(b + 3, nb)])
    if m.sum() > 4:
        print(f"{mid[b]:9.3f}{1/max(mid[b],1e-9):8.2f}{int(m.sum()):7d}"
              f"{np.median(iv[m]):11.1f}{np.percentile(iv[m],90):11.1f}")

# ---- C. do the STRONGEST peaks cluster at one radius? ----------------------------------------
print("\nC. where do the top-ranked peaks live? (intensity selecting a ring would show here)")
for frac, lbl in ((0.10, "top 10% by intensity"), (0.25, "top 25%"), (1.0, "all peaks")):
    k = max(1, int(len(iv) * frac))
    sel = np.argsort(iv)[::-1][:k]
    print(f"  {lbl:<22} median |q| {np.median(qn[sel]):.3f}  "
          f"IQR {np.percentile(qn[sel],25):.3f}-{np.percentile(qn[sel],75):.3f}")
r_top = np.corrcoef(np.argsort(np.argsort(-iv)), qn)[0, 1]
print(f"  Spearman-ish corr(intensity rank, |q|) = {r_top:+.3f}"
      f"   ({'strong radial bias' if abs(r_top) > 0.3 else 'weak' if abs(r_top) > 0.1 else 'negligible'})")

print("\nverdict")
excess = cnt[peak_bin] / max(exp[peak_bin], 1e-9)
if excess > 1.6:
    print(f"  Ring-like excess of {excess:.2f}x at |q|={mid[peak_bin]:.3f} (d={1/mid[peak_bin]:.2f} A)"
          f" -> background-aware ranking has something to bite on; proceed to step 2.")
else:
    print(f"  No ring-like excess (max {excess:.2f}x at d={1/mid[peak_bin]:.2f} A). These frames are")
    print("  already clean, so a background-aware rank cannot be distinguished from intensity HERE.")
    print("  Testing the idea needs frames where a finder actually over-finds (v4/pf9 on a water-ring")
    print("  dataset), not this stored-peakfinder8 corpus.")
