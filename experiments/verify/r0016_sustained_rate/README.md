# r0016_sustained_rate

Sustained frames per second of the shipped `StreamDriver` at its defaults on one A100, past start-up,
on every event of mfxx49820 r0016 (Epix10ka2M, lysozyme, 37,710 events in arrival order), with the
strict-bar yield of the same pass. The timed region is `push()` and the final `close()`, each bracketed
by a device synchronize, from one full batch (64 frames) after the lock frame to the end of the run.
Reading and GPU calibration are outside it, so this is not a detector readout rate.

- Commit 95d51ce. CrystFEL not used. Environment `ana-4.0.58-py3-minipytorch` (torch 2.1.0, cupy 12.3.0, psana1).
- Recorded run: Slurm 41040988, task 0 on sdfampere038 and task 1 on sdfampere020, 16 Oct 2026.

## Inputs

| input | path |
|---|---|
| raw data | `/sdf/data/lcls/ds/mfx/mfxx49820/xtc/mfxx49820-r0016-s0[0-3]-c0[01].xtc` |
| geometry | `/sdf/data/lcls/ds/mfx/mfxx49820/results/btx/geom/r0016.geom` (md5 0deaac06) |
| reference cell | btx `sample2.cell`: 79.327 79.461 38.406 90.111 89.981 90.083 |
| calib-dir override | `$VERIFY_INPUTS/glint_gt_mfxx49820/calib` |
| finder check (summary only) | `$VERIFY_INPUTS/glint_gt_mfxx49820/q6000_geom.npz` |

## argv

```
python -u j1_stream.py --glint $VERIFY_GLINT --exp mfxx49820 --run 16 --det MfxEndstation.0:Epix10ka2M.0 \
  --calib-dir $VERIFY_INPUTS/glint_gt_mfxx49820/calib --geom /sdf/data/lcls/ds/mfx/mfxx49820/results/btx/geom/r0016.geom \
  --zdist 0.102973 --wavelength 1.290757 --cell 79.327,79.461,38.406,90.111,89.981,90.083 \
  --arms defaults,events_off,wrescue,wrescue_pix,known --rescue-pixels 256 \
  --chunk 1024 --warm-events 1024 --check-calib 8 --calib gpu --window 2048 --out-dir $OUT --tag a${TASK}
python -u $VERIFY_GLINT/experiments/record_stream_replay.py --cupy --input real=$OUT/j1_a${TASK}_q.npz \
  --input scr=$OUT/j1_a${TASK}_qscr.npz --ref real=<cell> --ref scr=<cell> --B 64 --min-inliers 0 --out $OUT/j1_a${TASK}_null.json
python -u j1_summary.py $OUT
```

Strict bar: `same_lattice(M, ref)` and at least 10 and at least 25% of the frame's peaks within 0.15 of
integer hkl (constants imported from `glint.glint_fast`).

## Null and control

- Null: the `defaults` arm's per-frame q, azimuth-scrambled (`scramble_qlist.scramble_frames`, seed
  20260928), replayed by `experiments/record_stream_replay.py` after the real q.
- Controls: `known` (the reference cell from frame 0, no blind phase) and `events_off` (no event log).

## Expected

| FACTS key | value |
|---|---|
| `r0016_sustained_fps` | 231.1 (task 1; task 0 gave 255.9 with identical yields) |
| `r0016_hits` | 12133 |
| `r0016_lock_frame` | 24 |
| `r0016_strict_of_hits`, `r0016_strict_pct` | 6653, 54.8 |
| `r0016_null_strict_of_hits`, `r0016_null_strict_pct` | 2015, 16.6 |

Not in FACTS: the `known` arm locks at frame 0 and reaches 6,716 of 12,133. Task 0 ran on sdfampere038,
which the later checks exclude; FACTS quotes task 1.

## Outputs

`out/j1_a{0,1}.json` (per arm: steady-state and window rates, lock frame and cell, yields, provenance),
`j1_a*_frames.npz`, `j1_a*_q.npz` and `j1_a*_qscr.npz` (the null's inputs), `j1_a*_null.json`,
`j1_summary.json`; logs in `logs/`.

## Run

```
mkdir -p $VERIFY_OUT/r0016_sustained_rate/logs && cd $VERIFY_OUT/r0016_sustained_rate
sbatch $VERIFY_DIR/r0016_sustained_rate/j1_stream.sbatch
# after both tasks end, in ana-4.0.58-py3-minipytorch:
python $VERIFY_DIR/r0016_sustained_rate/j1_summary.py $VERIFY_OUT/r0016_sustained_rate/out
```
