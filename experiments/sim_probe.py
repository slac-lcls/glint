"""nanoBragg API + convention probe (cctbx.python / libtbx.python).

Render ONE lysozyme still at a KNOWN orientation, dump the detector params and the brightest
pixels. Purpose: pin nanoBragg's detector/orientation convention against fftindex's predict bridge
before building the full simulate->index->integrate->merge pipeline. Run with cctbx:

  source /sdf/group/lcls/ds/tools/cctbx/psana2/build/setpaths.sh
  libtbx.python sim_probe.py
"""
from __future__ import annotations
import numpy as np
from simtbx.nanoBragg import nanoBragg, shapetype
from cctbx import crystal, miller
from cctbx.array_family import flex
from scitbx.matrix import sqr

CELL = (79.0, 79.0, 38.0, 90.0, 90.0, 90.0)
SG = "P43212"
DMIN = 3.0
DET_N = 1024
PIX_MM = 0.2
DIST_MM = 150.0
WAVE_A = 1.32

# ground-truth structure factors: random amplitudes on the lyso lattice (seeded -> reproducible)
symm = crystal.symmetry(unit_cell=CELL, space_group_symbol=SG)
ms = miller.build_set(symm, anomalous_flag=False, d_min=DMIN)
rng = np.random.default_rng(0)
amps = flex.double(np.abs(rng.normal(size=ms.size())) * 100.0 + 10.0)
F = miller.array(ms, data=amps).set_observation_type_xray_amplitude()
print(f"Fhkl: {ms.size()} reflections to d_min={DMIN}")

# reciprocal cell matrix B (columns a*,b*,c* in 1/A) for IDENTITY orientation
uc = symm.unit_cell()
# cctbx: orthogonalization matrix maps fractional->cartesian (A); its inverse-transpose gives recip
B = sqr(uc.fractionalization_matrix()).transpose()   # columns are a*,b*,c* (1/A), Cartesian
print("recip B (a*,b*,c* as columns, 1/A):")
Bn = np.array(B).reshape(3, 3)
print(np.array2string(Bn, precision=5, suppress_small=True))

# random orientation U so the diffraction pattern is ASYMMETRIC (a centered tetragonal pattern is
# invariant under the very flips we need to distinguish -> identity orientation can't pin convention)
def rot(ax, ay, az):
    cx, sx = np.cos(ax), np.sin(ax); cy, sy = np.cos(ay), np.sin(ay); cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx
U = rot(0.4, -0.7, 1.1)
A_recip = U @ Bn                                      # columns a*,b*,c* (1/A), rotated

SIM = nanoBragg(detpixels_slowfast=(DET_N, DET_N), pixel_size_mm=PIX_MM, verbose=0)
SIM.wavelength_A = WAVE_A
SIM.distance_mm = DIST_MM
SIM.Ncells_abc = (10, 10, 10)
SIM.xtal_shape = shapetype.Gauss
SIM.Fhkl = F
SIM.Amatrix = sqr(tuple(A_recip.flatten()))          # random orientation: A = U B
SIM.add_nanoBragg_spots()
img = SIM.raw_pixels.as_numpy_array().reshape(DET_N, DET_N)

print(f"\nraw_pixels: shape={img.shape} max={img.max():.1f} mean={img.mean():.3f} "
      f"nbright(>0.1*max)={(img > 0.1*img.max()).sum()}")
print("Amatrix readback:", np.array2string(np.array(SIM.Amatrix).reshape(3, 3), precision=5, suppress_small=True))
try:
    print("beam_center_mm:", SIM.beam_center_mm)
except Exception as e:
    print("beam_center_mm n/a:", e)
for attr in ("pixel_size_mm", "distance_mm", "close_distance_mm"):
    try:
        print(f"  {attr} = {getattr(SIM, attr)}")
    except Exception:
        pass

# brightest local maxima (slow=ss, fast=fs) for the overlay test
from scipy.ndimage import maximum_filter
mx = maximum_filter(img, size=7)
ys, xs = np.where((img == mx) & (img > 0.15 * img.max()))
order = np.argsort(img[ys, xs])[::-1][:15]
print("\nbrightest local maxima (ss=slow, fs=fast, value):")
for i in order:
    print(f"  ss={ys[i]:4d} fs={xs[i]:4d}  I={img[ys[i],xs[i]]:.1f}")

np.save("/sdf/home/s/smarches/sim_still0.npy", img)
np.save("/sdf/home/s/smarches/sim_still0_B.npy", Bn)
np.save("/sdf/home/s/smarches/sim_still0_Arecip.npy", A_recip)   # columns a*,b*,c* (rotated)
print("\nsaved sim_still0.npy + sim_still0_B.npy + sim_still0_Arecip.npy")
