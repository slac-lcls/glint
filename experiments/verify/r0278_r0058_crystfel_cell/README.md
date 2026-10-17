# r0278_r0058_crystfel_cell

CrystFEL's most common cell on mfxl1038923 r0278 and r0058, as a reference independent of GLINT's indexer.
No cell was recorded for this sample, so `../per_frame_accuracy_vs_peaks` uses GLINT's vote over every
frame's best cell. Here blind XGANDALF through `indexamajig` indexes the same frames and peaks, and
`cells.py` finds the most common of CrystFEL's cells two ways: GLINT's `consensus_cell` used as a mode
finder (with the shipped batch gate, and ungated), and a histogram of Buerger-reduced parameters at 1 A and
1 degree. Each is compared with GLINT's vote cell by `same_lattice` and the volume ratio.

- Commit 95d51ce. CrystFEL 0.12.0 (`/sdf/group/lcls/ds/tools/crystfel/0.12.0`). Environment
  `ana-4.0.58-py3-minipytorch`; the job stops if any `GLINT_*` variable is set.
- Recorded run: Slurm 41155327, sdfmilan183, 64 cores, 17 Oct 2026 (as run: `j14.sbatch` md5 875457e4,
  `cells.py` md5 69f8bfd1, before the path changes in this folder).

## Inputs

- `$VERIFY_OUT/per_frame_accuracy_vs_peaks/r0278_qlists.npz` (1,940 frames) and `r0058_qlists.npz`
  (2,370 frames): the q lists of the per-frame check, GLINT's psana1 reader and v4 peak finder, the btx
  r0016 panel layout at 0.058 and 0.050 m.
- `$VERIFY_INPUTS/cxi_tools/bin/make_cxi_pinned.py` (md5 e237baae682b05988406b7acfdd70268, checked by the
  job): `experiments/make_cxi.py` at 95d51ce with its two `sys.path` lines and two default paths pointed
  at the pinned tree. It writes the CXI peak lists, one wavelength per set (r0058: the median of a 0.8%
  spread).

## argv

```
python npz2qlist_j4.py $J4OUT/<run>_qlists.npz inputs/<run>_real.txt
python $VERIFY_GLINT/experiments/scramble_qlist.py inputs/<run>_first400.txt inputs/<run>_scr.txt --seed 20260928
python make_cxi_pinned.py ../inputs/<set>.txt <set>
indexamajig -i <set>_events.lst -g <set>.geom -o <set>.stream --peaks=cxi --no-image-data --no-revalidate \
  --indexing=xgandalf -j 64
python cells.py $OUT
```

## Null

Azimuth-scrambled copies of each run's first 400 frames through the same chain.

## Expected (FACTS keys)

| key | value |
|---|---|
| `cfmode_r0278_a`, `_b`, `_c` | 44.64, 70.84, 89.46 |
| `cfmode_r0278_support` of `cfmode_r0278_indexed` | 8 of 667 (`_support_pct` 1.2) |
| `cfmode_r0278_vol_ratio` | 0.994 |
| `cfmode_r0058_a`, `_b`, `_c` | 43.65, 67.71, 88.73 |
| `cfmode_r0058_support` of `cfmode_r0058_indexed` | 15 of 447 (`_support_pct` 3.4) |
| `cfmode_r0058_vol_ratio` | 0.992 |
| `cfmode_null_frames`; `cfmode_r0278_null_max_group`, `cfmode_r0058_null_max_group` | 400; 1, 2 |

Both modes are the same lattice as GLINT's vote. 112 and 76 of the scrambled frames index.

## Outputs

`inputs/`, `cxi/`, `cf/<set>.stream` and logs, `crystfel_cell.json`, `crystfel_cell.md` (renamed from
the recorded run's output names), `MANIFEST.md5`, `logs/`.

## Run

```
mkdir -p $VERIFY_OUT/r0278_r0058_crystfel_cell/logs && cd $VERIFY_OUT/r0278_r0058_crystfel_cell
sbatch $VERIFY_DIR/r0278_r0058_crystfel_cell/j14.sbatch
```

`dryrun.sh` runs the chain on 30 frames per set in a temporary directory.
