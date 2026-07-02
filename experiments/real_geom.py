"""Real-geometry accuracy check: does radial.py's area (bbox) split match pyFAI's full 2-D splitting
profile BIN-FOR-BIN? We take pyFAI's OWN per-pixel q (and its deltaQ) so the geometry + binning are
identical, then compare I(q) directly (RMS). Close detector -> high angles -> curvature; OVERSAMPLED bins
(npt > radial pixels) -> pixels straddle several bins, the regime where splitting matters. Run in a
cupy+pyFAI env on a GPU node with OCL_ICD_VENDORS set. Credit: pyFAI (Kieffer et al., JAC 48, 510, 2015)."""
import sys, numpy as np
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
from radial import RadialLUT
import logging; logging.getLogger("pyFAI").setLevel(logging.ERROR)
try:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
except Exception:
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

H = W = 2048; px = 100e-6
ai = AzimuthalIntegrator(dist=0.10, poni1=(H // 2) * px, poni2=(W // 2) * px,
                         pixel1=px, pixel2=px, wavelength=1e-10)
qpp = np.asarray(ai.qArray((H, W)), np.float64)                 # per-pixel q (nm^-1), pyFAI's geometry
try:
    dqpp = np.asarray(ai.deltaQ((H, W)), np.float64)            # pyFAI's per-pixel q half-extent
except Exception:
    dqpp = None
qmin, qmax = float(qpp.min()), float(qpp.max())
th_max = np.degrees(np.arctan((H // 2) * px * np.sqrt(2) / 0.10))

rings = np.linspace(qmin + 0.10 * (qmax - qmin), qmax - 0.05 * (qmax - qmin), 6)
sig = 0.003 * (qmax - qmin)
img = np.zeros((H, W), np.float32)
for qr in rings:
    img += np.exp(-((qpp - qr) ** 2) / (2 * sig ** 2)).astype(np.float32)

npt = 2000                                                      # oversampled (> ~1024 radial pixels)
sa = np.asarray(ai.solidAngleArray((H, W)), np.float64)         # per-pixel solid angle (pyFAI's correction)
print(f"geometry: dist=0.10 m, 2theta_max~{th_max:.0f} deg, q={qmin:.2f}-{qmax:.2f} nm^-1, npt={npt} (oversampled)")
lin = RadialLUT(qpp, nbin=npt, qmin=qmin, qmax=qmax, split="linear")
aG = RadialLUT(qpp, nbin=npt, qmin=qmin, qmax=qmax, split="area")
linS = RadialLUT(qpp, nbin=npt, qmin=qmin, qmax=qmax, split="linear", norm=sa)  # solid-angle-corrected
aGS = RadialLUT(qpp, nbin=npt, qmin=qmin, qmax=qmax, split="area", norm=sa)
print(f"nnz/pixel: linear {lin.M.nnz/(H*W):.2f}   area {aG.M.nnz/(H*W):.2f}")

for corr in (False, True):
    _, I_pf = ai.integrate1d(img, npt, unit="q_nm^-1", radial_range=(qmin, qmax),
                             method=("full", "csr", "opencl"),
                             correctSolidAngle=corr, polarization_factor=None)
    I_pf = np.asarray(I_pf, np.float64)

    def rms(I):
        I = np.asarray(I, np.float64); m = np.isfinite(I) & np.isfinite(I_pf)
        return np.sqrt(np.mean(((I - I_pf)[m]) ** 2)) / np.nanmax(I_pf) * 1e3

    print(f"\n== pyFAI correctSolidAngle={corr} ==   RMS vs pyFAI-full [x1e-3, rel to peak]:")
    L, A = (linS, aGS) if corr else (lin, aG)                   # match the correction
    print(f"  linear split   {rms(L.integrate(img)[1]):6.2f}")
    print(f"  area   split   {rms(A.integrate(img)[1]):6.2f}")
