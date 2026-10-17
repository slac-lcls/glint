# per_frame_accuracy_vs_peaks

How often the shipped per-frame blind indexer's best cell is correct, binned by peaks per frame.
`glint.glint_fast.index_blind_fast` at its defaults, top-1 cell, one frame at a time, no consensus and
no rescue; scored at the strict bar. Bins (inclusive peak counts): 6-9, 10-14, 15-19, 20-29, 30-39,
40-59, 60-99, 100-149, 150-249.

- Commit 95d51ce. No CrystFEL step. Environment `ana-4.0.58-py3-minipytorch`, one A100 per task.
  Every Python step stops if any environment variable read by the shipped `glint/` modules is set.
- Recorded run: Slurm 41006499 (array: r0016 as 41017341 on sdfampere035; r0278, r0058 and cxidb-45 as
  41017395, 41017753 and 41018032 on sdfampere030; task 5 r0278z060) and 41006500 (binning), 16 Oct 2026.

## Inputs

| run | peaks from | geometry | wavelength | events | frames |
|---|---|---|---|---|---|
| r0016 (mfxx49820) | xtc, GLINT's psana1 reader, PeakFinderV4 defaults, >= 6 peaks | btx `r0016.geom` with `--calib-dir $VERIFY_INPUTS/glint_gt_mfxx49820/calib`, zdist 0.102973 m | 1.290757 A | 6000 | 2,228 |
| r0278 (mfxl1038923) | same | btx r0016 panel layout, zdist 0.058 m | 1.26296 A | 3000 | 1,940 |
| r0278z060 | same | same, zdist 0.060 m | 1.26296 A | 3000 | 1,940 |
| r0058 (mfxl1038923) | same | same, zdist 0.050 m | per event | 3000 | 2,370 |
| cxidb-45 | `$VERIFY_GLINT/experiments/prok_q.npz` (sha256 e4ac4f54...91a75), 170 peaks per frame | applied when the file was made | per frame | | 907 |

Reference cells: r0016 btx `sample2.cell`; cxidb-45 P4(3)2(1)2 68.7 68.7 108.6. r0278 and r0058 have no
recorded cell, so their reference is GLINT's vote over every frame's best cell (`consensus_cell`, shipped
`min_frac` 0.02, `min_lead` 1.5); `../r0278_r0058_crystfel_cell` checks it with CrystFEL.

## argv

```
python -u j4_qlists.py <run> $OUT        # r0016 r0278 r0058 r0278z060
python -u j4_score.py <run> $OUT         # also cxidb45
python -u j4_bin.py $OUT
```

Strict bar: the best cell is the reference lattice (`same_lattice`) and indexes at least 25% and at least
10 of the frame's peaks within 0.15 of integer hkl.

## Null and control

- Null: the same frames azimuth-scrambled (`scramble_qlist.scramble_frames`, seed 20260928).
- Control: `glint.hybrid_stream.index_known_fast(frames, REF)`, the reference cell supplied, on the real
  and on the scrambled frames. Its own null is not zero.

## Expected (FACTS keys)

| key | value |
|---|---|
| `r0016_sf_frames` | 2228 |
| `r0016_sf_strict_of2228`, `r0016_sf_strict_pct` | 679, 30 |
| `r0016_sf_lattice_of2228` | 690 |
| `r0016_sf_bin_<lo>_<hi>_frames`, `_strict`, `_pct` | 6-9: 240, 0; 10-14: 227, 34; 15-19: 189, 40; 20-29: 328, 93; 30-39: 259, 94; 40-59: 377, 157; 60-99: 382, 178; 100-149: 163, 69; 150-249: 63, 14 |
| `r0016_sf_null_of2228` | 0 (0 in every bin of every run) |
| `r0016_known_strict_of2228`, `r0016_known_strict_pct` | 1234, 55 |
| `r0016_known_null_of2228` | 330 |
| `pk45_sf_strict_of907`, `pk45_sf_known_of907` | 324, 806 |
| `r0278_frames`, `r0278_vote_a`, `_b`, `_c` | 1940; 44.41, 70.60, 90.72 (task 1, 0.058 m) |
| `r0278_vote_support` of `r0278_vote_pool` | 635 of 1785 |
| `r0058_frames`, `r0058_vote_a`, `_b`, `_c` | 2370; 43.87, 67.68, 89.02 |
| `r0058_vote_support` of `r0058_vote_pool` | 718 of 2319 |

The frame counts are also checked against the August reads (`<run>_qlists.json`, MATCH or DIFFERS).

## Outputs

`<run>_qlists.npz` and `.json`, `<run>_frames.csv`, `<run>_summary.json`, `cxidb45_thin_frames.csv`
(random thinning of the 170-peak lists, its own table), `J4_bins.csv`, `J4_table.md`,
`J4_accuracy_vs_peaks.png`, `logs/`.

## Run

```
mkdir -p $VERIFY_OUT/per_frame_accuracy_vs_peaks/logs && cd $VERIFY_OUT/per_frame_accuracy_vs_peaks
A=$(sbatch --parsable $VERIFY_DIR/per_frame_accuracy_vs_peaks/j4_index.sbatch)
sbatch --dependency=afterany:${A%%;*} $VERIFY_DIR/per_frame_accuracy_vs_peaks/j4_bin.sbatch
```

Not run: array task 4 (the 120 cxidb-17 frames, `sbatch --array=4`). The FACTS cxidb-45 merge (CC* 0.90,
R_split 31%, <I/sigma> 7.8) is not re-derived here.
