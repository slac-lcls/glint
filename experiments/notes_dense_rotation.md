# GLINT on dense / rotation data — GPU local-cluster FFT vs DIALS/labelit DPS

**Question (user).** Do our earlier FFT methods go faster on DENSE / rotation data (regular
crystallography), and how do we compare to DIALS/phenix on rate + speed? The old `fftindex`
3D-FFT-of-the-peak-cloud IS the DIALS/labelit `fft3d` algorithm, so the real question is GPU vs CPU
on the same idea, in DIALS's home turf.

## The method: translation-invariant local-cluster FFT (user idea)
A single global 3D FFT of the whole rlp cloud explodes on dense data (peak extraction was ~40 s on a
full-rotation cloud). Because `|F(x)|` is translation-invariant, the lattice vectors are recovered
from small clusters of Bragg peaks centred *anywhere* in reciprocal space. So we FFT several SMALL
centred clusters on a fine LOCAL grid — `_cluster_fft_seeds` in `fftindex/glint_fast.py` — giving
`O(K * N_cluster)` seeds (flat in density) that feed GLINT's refine (M3) + batched-GPU assembly (M4).
Entry point `index_blind_cluster_seeded`; below `CLUSTER_MIN=3000` rlps (thin/sparse) it falls back
to the Fibonacci grid (the SFX path, untouched — zero regression).

## Head-to-head vs DPS (rstbx `DPS_primitive_lattice`, the labelit/phenix/MOSFLM 1-D FFT engine)
Same clouds indexed by GLINT (one A100) and DPS (one CPU core). DPS runtime scales with the rlp
count; GLINT is flat. Scripts: `gen_clouds.py` (density sweep, one lyso cell) / `gen_cells.py` (six
cells) generate; `glint_clouds.py`/`glint_cells.py` (GPU) and `dials_clouds.py`/`dials_cells.py`
(cctbx CPU) index the identical clouds.

### Cross-cell corroboration (six unit cells, full-rotation clouds to 3 A) — paper Table 4
| cell (A)              | rlps  | GLINT rate | GLINT ms | DPS rate | DPS ms  | speedup |
|-----------------------|-------|-----------|----------|----------|---------|---------|
| lysozyme 79/79/38     | 36.9k | 100%      | 47.0     | 100%     | 4039.8  | 86x     |
| Proteinase K 68/68/109| 79.3k | 100%      | 50.7     | 75%      | 6673.2  | 132x    |
| thaumatin 58/58/130   | 67.8k | 100%      | 46.4     | 100%     | 6015.5  | 130x    |
| hexagonal 105/105/75  | 109k  | 100%      | 52.6     | 100%     | 8961.6  | 170x    |
| cubic 78/78/78        | 73.5k | 100%      | 47.6     | 100%     | 6458.2  | 136x    |
| ortho 60/110/135      | 138k  | 100%      | 51.4     | 50%      | 11398.0 | 222x    |

GLINT: flat ~47-53 ms, **100% on all six cells, 86-222x faster than DPS**, matching DPS where it is
complete and EXCEEDING it on the two large cells where DPS drops (prok 100 vs 75, ortho 100 vs 50).

## Large-anisotropic fix (was ortho 25%, prok 75%)
Diagnosis (`tune_ortho.py`): the failure was DOWNSTREAM (refine convergence), NOT seed generation —
the seeds always covered the truth axes (`gen 4/4` at every config). The old default grid
(`fov=160`, `n_grid=64` => 5 A/voxel) under-resolved long real-space axes, so refine landed in the
wrong basin. Fix = size the seed grid for long axes: **`_cluster_fft_seeds` defaults `fov 160->200`,
`n_grid 64->96`** (~4 A/voxel, keeps axes to 0.9*fov = 180 A). Recovered BOTH weak cells (ortho
25->100%, prok 75->100%); lyso/thaum/hex/cubic stay 100%; the lyso density sweep (still/wedge/full)
stays 100% — no regression; timing still flat. Confirmed through the production entry
(`confirm_fix.py`).

## Optimizer note (CG / Newton for M3)
`refine_vec_newton` (analytic 3x3 Hessian, damped) added to `glint_index.py`, flag `REFINER=newton`.
TESTED = NEGATIVE: ~10-20% faster but LESS robust (basin-jumping on the multimodal cos comb from
imperfect seeds; damping insufficient). Kept flag-gated for the record; default stays
gradient+momentum.

Environment: NERSC Perlmutter, `module load pytorch/2.6.0`, one A100. Data + run copies at
`/pscratch/sd/s/smarches/glint_real` (`cells.npz`, `clouds.npz`, `cells_glint_fixed.txt`,
`cells_dps.txt`).
