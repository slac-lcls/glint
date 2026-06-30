# Frontier #1 real data: GLINT on cxidb-45 Proteinase K (SACLA MPCCD) at NERSC

Real-data counterpart of the nanoBragg validation ([[notes_merge_pipeline]]). Data = the CrystFEL-
tutorial **Proteinase K** SFX dataset (cxidb-45, SACLA MPCCD), staged at NERSC
`/global/cfs/cdirs/lcls/dermen/cxidb45/data/run296940-*.h5` (`tag-*/data` = (8192,512) int16, per-shot
photon energy/wavelength). Geometry = mnasser's `cxidb_62/mpccd-optimized.geom` (same MPCCD octal, 8
panels q1-q8, 50 µm px, clen 0.055 m) -- directly usable. lambda ~ 0.952 A. Code at NERSC
`/pscratch/sd/s/smarches/glint_real` (env `module load pytorch/2.6.0`).

## Result 1 -- GLINT blind-indexes real ProK (generalises beyond lysozyme)

`prok_probe.py` / `phaseB_prok.py`: peakfind -> q (`lute_bridge.peaks_to_q`, multi-panel geom,
per-shot lambda) -> `index_blind_fast`. **186/200 frames blind-indexed (93%)**; median sorted cell
**[68.7, 69.0, 108.6] A** = textbook ProK P4(3)2(1)2. GLINT's home cell is lysozyme; this is a
DIFFERENT protein on REAL SACLA data, indexed blind at the same rate it hits cxidb lyso. Of the 186,
**72** land on the full ProK lattice (`same_lattice`); the rest are sub-lattices (the known single-frame
selection scatter -- a known-cell rescue would recover them).

## Result 2 -- end-to-end merge runs on real data, but is partiality-limited

72 ProK frames -> predict_spots -> integrate_spots -> stream -> Monte-Carlo merge (`merge_stats.py`,
4/mmm, per-frame scale + inverse-variance mean; self-contained, same math as `process_hkl`):

| | value |
|---|---|
| unique reflections (4/mmm) | 16.5k |
| redundancy | 17.8x |
| ⟨I/σ⟩ | 3.2 |
| **CC1/2** | **~0.15** |
| CC* | ~0.52 |
| Rsplit | ~34% |

**Diagnosis = partiality, not the bridge.** Two controls rule out the obvious culprits:
- **Not orientation**: adding a least-squares predict-refine (`refine_orient`) left CC1/2 unchanged.
- **Not noise**: CC1/2 *drops* as the I/σ floor rises (0.15 -> 0.08), the opposite of noise dilution.

The signature is **partiality**: a still samples each reflection at a random point on its rocking curve,
so the SAME hkl is fully recorded in one frame and a sliver in another. A naive Monte-Carlo merge with
no partiality model (and `tol` predicting ~4000 geometrically-near-Ewald reflections/frame, most tiny
partials) averages inconsistent measurements -> low CC1/2. This is *exactly* the problem
**`partialator`** (per-crystal scaling + partiality + post-refinement) and **diffBragg** (forward-model
every pixel) exist to solve -- the Holton/Brewster/Sauter line. The nanoBragg case hit the same wall
(CC1/2 0.59 even with perfect orientations, no partiality model); real data with mosaicity + SASE
bandwidth pushes it lower.

So GLINT's contribution (blind index + geometry-correct, mergeable stream) is **validated end-to-end on
real data**; the absolute CC1/2 is gated by the downstream merger, which should be `partialator`/diffBragg,
not GLINT.

## Next (separable, downstream)
- Run real **`partialator`** (NERSC CrystFEL at `/global/cfs/cdirs/lcls/chuck/crystfel/bin`; needs
  `libgsl.so.19` + `libhdf5.so.7` on `LD_LIBRARY_PATH` -- hdf5 found in `chuck/hdf5/lib`, gsl TBD) or a
  conda-forge CrystFEL; expect a large CC1/2 lift from scaling+partiality.
- **Known-cell rescue** the sub-lattice frames (72 -> ~180) for redundancy.
- More frames (file 0 has 452; 3 files ~1356) and tol/partiality tuning.
- Eventually hand the stream to `diffBragg` (cctbx on NERSC) for the forward-model merge.
