# stopping_rule_live_and_scrambled

The shipped stopping rule, frame by frame. `StreamDriver(Mc=None)` with every default (`lock_support` 3;
`lock_gap` 2, adaptive; `lock_frac` 0.02 and `lock_lead` 1.5 once 72 cells are pooled; `warmup_nbest` 3;
`min_peaks` 6; `alias_gate` None), read from the signature at run time. Three measurements:

1. Real runs in arrival order (psana event order; cxidb-17: file order): frames pushed before the driver
   decides, and whether the decided cell is right.
2. The null: 10,000 groups of 100 azimuth-scrambled frames (3,333 from r0278, 3,333 from r0058, 3,334 from
   refined r0016) and 20 whole scrambled runs per run size, each pushed into a fresh driver. Any decision
   is false.
3. 400 random orders per dataset, seeds 0-7, replayed on the cached cells.

A decision is correct when `same_lattice(decided cell, reference)` passes at its defaults. References:
btx `sample2.cell` for r0016, lysozyme 79.02 79.02 37.98 for cxidb-17, and the whole-run batch vote for
r0278 and r0058 (no independent cell exists for those two).

- Commit 95d51ce. CrystFEL not used. Seed 20261015. Environment `ana-4.0.58-py3-minipytorch`.
- Recorded run: Slurm 41011615 (prep, sdfampere023, A100), 41011616 (scramble array 0-3, sdfampere026 and
  sdfampere030), 41056365 (replay, sdfmilan027, submitted with `--time=18:00:00`; the 6 h attempt
  41011617 timed out), 16 Oct 2026.

## Inputs

| dataset | source |
|---|---|
| r0278 | xtc via `run_to_qframes_psana1`, zdist 0.060 m, wavelength 1.26296 A, 3,000 events, btx r0016 panel layout |
| r0058 | the same, zdist 0.050 m, per-event wavelength |
| r0016ref | xtc, btx `r0016.geom` (md5 0deaac06) with `$VERIFY_INPUTS/glint_gt_mfxx49820/calib`, 6,000 events |
| r0016ref_aug | `$VERIFY_INPUTS/glint_gt_mfxx49820/q6000_geom.npz` (md5 742ac14c), cross-check of the re-extraction |
| r0016unref | `$VERIFY_INPUTS/glint_gt_mfxx49820/r0016_q6000.npz` (md5 18aedece), psana's unrefined geometry |
| cxidb480 | `$VERIFY_INPUTS/q480_fix.txt` (md5 d6d86c1b), 480 cxidb-17 frames |
| harness check | `experiments/nbest_120.npz` (md5 a1c4c04f) |

Reference cell file: `/sdf/data/lcls/ds/mfx/mfxx49820/results/btx/cell/sample2.cell` (md5 70359fee).

## argv

```
python -u j5_prep.py all
python -u j5_scramble.py $SLURM_ARRAY_TASK_ID 4 --G 100 --groups 3333,3333,3334 --long 20 --hours 5.4 --procs 2
python -u j5_replay.py --procs 32 --orders 400
```

`j5_check120.py` (CPU, seconds) must first reproduce `poolgate_clean_lock_med` 6, `poolgate_clean_lock_max`
18, `poolgate_clean_maxpool` 54 and `poolgate_clean_diff_72` 0. `prep_summary.json` checks the regenerated
frames against `indexed_r0278` (1785) and `indexed_r0058` (2319).

## Expected (FACTS keys)

| key | value |
|---|---|
| `decide_r0278_frames`, `decide_r0058_frames`, `decide_r0016_frames`, `decide_cx17_frames` | 10, 6, 8, 5 |
| `decide_*_support` | 3, 3, 3, 4 |
| `decide_*_vol_ratio` | 1.0003, 0.9848, 1.0063, 0.9994 |
| `decide_scr_groups`, `decide_scr_group_frames`, `decide_scr_decided` | 10000, 100, 8 |
| `decide_scr_wald_lo_pct`, `decide_scr_wald_hi_pct` | 0.025, 0.135 |
| `decide_scr_at_support3`, `decide_scr_upper95_per1000` | 7, 1.443 |
| `decide_scr_batch_decided` | 0 |
| `decide_scr_wholerun_groups`, `decide_scr_wholerun_decided` | 60, 0 |
| `decide_rand_r0278_correct`, `_r0058_`, `_r0016_`, `_cx17_` | 393, 383, 399, 399 of 400 (seed 1) |
| `decide_rand_r0058_correct_min`, `_max` | 380, 393 over the 8 seeds |

Part C2 (unrefined r0016 in the August `wrong_sweep.py` order scheme) recorded 363 decisions and 140 wrong
for support and lead only, 4 and 3 for the shipped rule. It does not re-derive the FACTS `poolgate_bad_*`
keys (342, 194, 3, 2).

## Outputs

`prep/` (`q_*.npz`, `votes_*.npz`, `live_*.json`, `prep_summary.json`), `scramble/task<t>/`,
`replay/J5_results.json`, `replay/J5_tables.md`, `logs/`.

## Run

```
mkdir -p $VERIFY_OUT/stopping_rule_live_and_scrambled/logs && cd $VERIFY_OUT/stopping_rule_live_and_scrambled
P=$(sbatch --parsable $VERIFY_DIR/stopping_rule_live_and_scrambled/j5_prep.sbatch)
S=$(sbatch --parsable --dependency=afterok:$P $VERIFY_DIR/stopping_rule_live_and_scrambled/j5_scramble.sbatch)
sbatch --time=18:00:00 --dependency=afterany:$S $VERIFY_DIR/stopping_rule_live_and_scrambled/j5_replay.sbatch
```

Not run: `j5_scramble_cpu.sbatch`, the CPU form of the scramble step (submit one or the other).
