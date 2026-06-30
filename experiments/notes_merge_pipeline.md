# Frontier #1: end-to-end merge (GLINT orientation -> real I/sigma -> partialator)

Closes the blocked merge: GLINT indexes -> **predict spots** (orientation -> hkl in diffracting
condition -> detector fs,ss; `fftindex.predict.predict_spots`, inverse of `lute_bridge.peaks_to_q`)
-> **box-integrate** the image (`integrate_spots`) -> **CrystFEL `.stream`** with REAL I/sigma
(`write_stream_integrated`) -> **`process_hkl`/`partialator` merge** -> CC1/2, CC*, Rsplit, and CC to
the ground-truth |F|^2. Raw pixels came from **nanoBragg** (Holton's simulator, cctbx/simtbx on S3DF),
so the true structure factors are known and the merge can be scored exactly.

## Setup
- `sim_dataset.py` (cctbx): K lysozyme stills (P43212, 79/79/38), uniform-random orientations, random
  |F| with a Wilson B=18 falloff (default_F=0), flat bg + Poisson noise; single flat panel
  (0.2 mm px, 150 mm, 1024^2, lambda 1.32 A), d_min 1.9 A. Saves images + truth (orientations, |F|^2).
- Geometry/convention **validated two ways**: vs CrystFEL ground truth (`validate_predict.py`,
  projection inverse exact to 1.7e-13 px, predicted fs,ss within 0.84 px) and vs nanoBragg
  (`stage2_solve.py`, 100% integer hkl, cell recovered as 79/79/38). nanoBragg's Amatrix-transpose +
  an improper detector-frame rotation G pinned.
- `phaseB_integrate.py` (predict+integrate -> stream), `phaseC_merge.py` (process_hkl all/odd/even
  -> compare_hkl CC*/Rsplit + check_hkl completeness; CC to truth |F|^2 with 4/mmm canonicalisation).
- Merge: CrystFEL 0.12.0 `process_hkl -y 4/mmm --scale` (Monte-Carlo, no partiality model).

## Results (K=40 stills, process_hkl)

| orientation | Rsplit | CC1/2 | CC* | compl. | ⟨I/σ⟩ | red. | CC(I, true \|F\|²) |
|---|---|---|---|---|---|---|---|
| **truth** (nanoBragg)   | 57.2% | 0.586 | 0.860 | 71.9% | 3.2 | 5.6 | **0.795** |
| **GLINT blind**         | 56.1% | 0.608 | 0.870 | 71.9% | 3.7 | 5.6 | **0.796** |

**GLINT's blind orientations merge indistinguishably from the exact nanoBragg orientations** (CC* 0.870
vs 0.860; truth-CC 0.796 vs 0.795; 40/40 indexed). The merge is limited by partiality + low redundancy
(process_hkl Monte-Carlo floor at K=40), NOT by GLINT's orientation accuracy -> GLINT output is
**integration-grade**. This is the controlled, ground-truth-checkable proof that the predict->integrate
->merge bridge is correct.

The truth-orientation row is the **machinery upper bound** at K=40: with exact orientations the merge
is limited by *partiality* (each still samples a random point on every reflection's rocking curve) and
low redundancy, not by orientation error -- `process_hkl` has no partiality model, so CC1/2~0.59 /
Rsplit~57% is the expected Monte-Carlo floor at this frame count. The merged intensities nonetheless
track the input structure factors at **CC=0.80**. The GLINT-orientation row is the actual frontier-#1
claim: if it tracks the truth row, GLINT's blind orientations are **integration-grade**.

## Caveats / next
- 40 stills is LOW (real SFX merges thousands); the absolute FOMs are redundancy-limited, so the
  TRUTH-vs-GLINT *gap* is the meaningful quantity, not the absolute CC1/2.
- `partialator` (partiality + per-crystal scaling) should lift all rows; process_hkl first for a
  convention-free baseline.
- Real cxidb-17 pixels (purged) are the eventual real-data confirmation (separate, post-MFA).
