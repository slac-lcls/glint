# Real-data verifiers

Opt-in scripts that re-derive FACTS values (`experiments/check_numbers.py`) from real data on S3DF.
They need facility data, GPUs and Slurm, so they are not part of the CPU CI and nothing here runs on
its own. Each folder holds the scripts of one check as its recorded run used them, logic unchanged, and a
README with the protocol, inputs, argv, outputs and the values that run produced.

## How to run a check

1. Make a plain GLINT tree at the commit the recorded run used, with a `COMMIT` file. The scripts read
   that file; every check except `r51_blind_arm_with_guard` used 95d51ce.

   ```
   mkdir -p $TREE && git -C <glint checkout> archive 95d51ce | tar -x -C $TREE && echo 95d51ce > $TREE/COMMIT
   ```

2. Export four variables. There are no defaults: a script stops if one is missing.

   | variable | meaning |
   |---|---|
   | `VERIFY_GLINT` | the tree from step 1 |
   | `VERIFY_DIR` | this directory, `experiments/verify` in a checkout of this branch |
   | `VERIFY_OUT` | an output root; a check writes to `$VERIFY_OUT/<folder>` |
   | `VERIFY_INPUTS` | the inputs that are not facility data (table below) |

3. `mkdir -p $VERIFY_OUT/<folder>/logs && cd $VERIFY_OUT/<folder>`, then submit as the folder README
   says. The `#SBATCH --output` paths are relative to that directory.

Site settings: the account `lcls:prjdat21`, the partitions, the `--exclude` node lists and
`--constraint=OS_VER:8.10` are the S3DF settings of the recorded runs. Change them for another
allocation. Conda environments and the CrystFEL, DIALS and CCP4 builds are read from `/sdf/group/lcls/ds`;
raw data from `/sdf/data/lcls/ds`.

`$VERIFY_INPUTS` holds files that are not in the repository or the facility tree:

| path | md5 | used by |
|---|---|---|
| `glint_gt_mfxx49820/calib/` | | r0016 calib-dir override (0-end.data, the calibration btx refined against): `r0016_sustained_rate`, `per_frame_accuracy_vs_peaks`, `stopping_rule_live_and_scrambled` |
| `glint_gt_mfxx49820/q6000_geom.npz` | 742ac14c | `r0016_sustained_rate` (finder check), `stopping_rule_live_and_scrambled` |
| `glint_gt_mfxx49820/r0016_q6000.npz` | 18aedece | `stopping_rule_live_and_scrambled` (unrefined r0016) |
| `q480_fix.txt` | d6d86c1b | `stopping_rule_live_and_scrambled`, `xgandalf_fast_cores` (480 cxidb-17 frames) |
| `xg_fast/A_default.txt`, `xg_fast/B_fast.txt`, `xgd480_fix.txt` | | `xgandalf_fast_cores` (earlier driver outputs, controls) |
| `cxi_tools/bin/make_cxi_pinned.py` | e237baae | `r0278_r0058_crystfel_cell` |
| `cxidb17_harness/j8_score.py`, `cxidb17_harness/inputs/` | d4182cf7 (scorer) | `asdf_via_indexamajig`, `mosflm_via_indexamajig` |

## Checks

Keys not yet on main come from #264 (the r0016 rate and fast XGANDALF) and from the real-run FACTS bank
(the other checks); `python3 experiments/check_numbers.py --facts` lists the keys a checkout has.

| check | FACTS keys | expected | recorded run (Slurm) | commit |
|---|---|---|---|---|
| [`r0016_sustained_rate`](r0016_sustained_rate/) | `r0016_sustained_fps`, `r0016_hits`, `r0016_lock_frame`, `r0016_strict_of_hits`, `r0016_strict_pct`, `r0016_null_strict_of_hits`, `r0016_null_strict_pct` | 231.1 f/s; 12,133 hits; lock at frame 24; 6,653 strict (54.8%); null 2,015 (16.6%) | 41040988 | 95d51ce |
| [`xgandalf_fast_cores`](xgandalf_fast_cores/) | `xgandalf_fast_1core_ms` | 1137.7 ms per frame, one EPYC 7542 core | 41011531 | 95d51ce |
| [`merges_tofile_vs_crystfel`](merges_tofile_vs_crystfel/) | `merge_*` | r0016 CC1/2 0.370 (GLINT) against 0.323 (CrystFEL) on 998 frames, not detected as different; shuffled orientations accepted 0 of 1,778; r51 250 correct-lattice cells of 11,058 frames | 41037259-41037262, 41130408 | 95d51ce |
| [`per_frame_accuracy_vs_peaks`](per_frame_accuracy_vs_peaks/) | `r0016_sf_*`, `r0016_known_*`, `pk45_sf_*`, `r0278_*`, `r0058_*` | r0016 679 of 2,228 strict (30%), 0 in the null; cell supplied 1,234 (55%) | 41006499, 41006500 | 95d51ce |
| [`stopping_rule_live_and_scrambled`](stopping_rule_live_and_scrambled/) | `decide_*` | decides after 10, 6, 8 and 5 frames (r0278, r0058, refined r0016, 480 cxidb-17); 8 decisions in 10,000 scrambled groups of 100 | 41011615, 41011616, 41056365 | 95d51ce |
| [`cxidb45_dials_head_to_head`](cxidb45_dials_head_to_head/) | `pk45_*` | GLINT consensus 804 of 907 strict, DIALS with the cell 675, every null 0 | 41011687-41011690, 41099183, 41099184 | 95d51ce |
| [`r51_merge_vs_frames`](r51_merge_vs_frames/) | `r51_stream_*`, `r51_sweep_*` | LUTE's merge reproduced (CC1/2 0.2742221, 2,120 pairs); CC1/2 0.2890 at 251 frames | 41011666 | 95d51ce |
| [`r0278_r0058_crystfel_cell`](r0278_r0058_crystfel_cell/) | `cfmode_*` | CrystFEL's most common cell is GLINT's lattice on both runs, from 8 of 667 and 15 of 447 crystals | 41155327 | 95d51ce |
| [`asdf_via_indexamajig`](asdf_via_indexamajig/) | not yet in FACTS | strict of 120: 0 in every blind arm; with the cell 43 at defaults, 64 with every check off; nulls 0 | 41168320 | 95d51ce |
| [`mosflm_via_indexamajig`](mosflm_via_indexamajig/) | not yet in FACTS | strict of 120: 27 blind at defaults; with the cell 35 at defaults, 54 without the peak check; nulls 0 | 41172742 | 95d51ce |
| [`r51_blind_arm_with_guard`](r51_blind_arm_with_guard/) | not yet in FACTS | 2,408 solutions written, 250 on the reference lattice; CC1/2 GLINT minus CrystFEL -0.005 on 241 frames | 41172814, 41172816, 41172817 | abe330c |
