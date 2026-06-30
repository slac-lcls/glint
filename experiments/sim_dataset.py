"""Phase A (cctbx): simulate a small SFX still dataset with nanoBragg for the end-to-end merge demo.

K lysozyme stills at uniform-random orientations, one ground-truth structure-factor set (random
amplitudes with a Wilson-like resolution falloff, default_F=0 so only real reflections diffract),
flat background + Poisson noise. Saves the image stack + truth (orientations, hkl, |F|^2, geom) so
Phase B can GLINT-index -> predict -> integrate and Phase C can partialator-merge, and we can score
the merge against the known |F|. Detector = the validated single panel (0.2mm px, 150mm, 1024^2).

  source /sdf/group/lcls/ds/tools/cctbx/psana2/build/setpaths.sh
  libtbx.python sim_dataset.py [K]
"""
from __future__ import annotations
import os, sys
import numpy as np
from simtbx.nanoBragg import nanoBragg, shapetype
from cctbx import crystal, miller
from cctbx.array_family import flex
from scitbx.matrix import sqr

CELL = (79.0, 79.0, 38.0, 90.0, 90.0, 90.0)
SG = "P43212"
DMIN = 1.9                 # cover the detector corners
DET_N = 1024
PIX_MM = 0.2
DIST_MM = 150.0
WAVE_A = 1.32
NCELL = 12
BFAC = 18.0                # resolution falloff of the random |F|
BG = 8.0                   # flat background photons/pixel
TARGET = 4000.0           # ~photons in a strong spot after scaling
K = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = "/sdf/home/s/smarches/glint_sim"
os.makedirs(OUT, exist_ok=True)

# ---- ground-truth structure factors ----
symm = crystal.symmetry(unit_cell=CELL, space_group_symbol=SG)
ms = miller.build_set(symm, anomalous_flag=False, d_min=DMIN)
d = np.array(ms.d_spacings().data())
ss2 = (1.0 / d) ** 2
rng = np.random.default_rng(7)
amps = np.abs(rng.normal(size=ms.size())) * np.exp(-BFAC * ss2 / 4.0) * 1000.0 + 1.0
F = miller.array(ms, data=flex.double(amps)).set_observation_type_xray_amplitude()
hkl = np.array(ms.indices())
print(f"Fhkl: {ms.size()} refl to d_min={DMIN}; saving truth")

uc = symm.unit_cell()
Bn = np.array(sqr(uc.fractionalization_matrix()).transpose()).reshape(3, 3)   # cols a*,b*,c* (1/A)


def rand_rot(r):
    """Uniform random rotation (Shoemake)."""
    u1, u2, u3 = r.random(3)
    q = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                  np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)])
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


rr = np.random.default_rng(123)
nrng = np.random.default_rng(999)
stack = np.zeros((K, DET_N, DET_N), np.uint16)
Amats = np.zeros((K, 3, 3))
scale = None
for k in range(K):
    U = rand_rot(rr)
    A_recip = U @ Bn
    Amats[k] = A_recip
    SIM = nanoBragg(detpixels_slowfast=(DET_N, DET_N), pixel_size_mm=PIX_MM, verbose=0)
    SIM.wavelength_A = WAVE_A; SIM.distance_mm = DIST_MM
    SIM.Ncells_abc = (NCELL, NCELL, NCELL); SIM.xtal_shape = shapetype.Gauss
    SIM.Fhkl = F; SIM.default_F = 0
    SIM.Amatrix = sqr(tuple(A_recip.T.flatten()))   # nanoBragg wants the transpose
    SIM.add_nanoBragg_spots()
    raw = SIM.raw_pixels.as_numpy_array().reshape(DET_N, DET_N)
    if scale is None:
        scale = TARGET / np.percentile(raw, 99.99)
    noisy = nrng.poisson(np.clip(raw * scale + BG, 0, None)).astype(np.float64)
    stack[k] = np.clip(noisy, 0, 65535).astype(np.uint16)
    nb = int((stack[k] > 5 * BG).sum())
    print(f"  still {k:2d}: max={stack[k].max():5d}  npix>5bg={nb}", flush=True)

np.save(f"{OUT}/images.npy", stack)
np.savez(f"{OUT}/truth.npz", Amats=Amats, hkl=hkl.astype(np.int32), Fsq=(amps ** 2).astype(np.float32),
         cell=np.array(CELL), dmin=DMIN, det_n=DET_N, pix_mm=PIX_MM, dist_mm=DIST_MM, wave_A=WAVE_A,
         bg=BG, scale=scale, sg=SG)
print(f"\nsaved {OUT}/images.npy  ({stack.nbytes/1e6:.0f} MB)  + truth.npz  (K={K})")
