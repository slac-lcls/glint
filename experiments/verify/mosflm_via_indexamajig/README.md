# mosflm_via_indexamajig

MOSFLM through CrystFEL 0.12.0 `indexamajig` on the 120 cxidb-17 frames: the nine arms of
`../asdf_via_indexamajig` with `--indexing=mosflm`. `ipmosflm` is CCP4 8.0's
(`/sdf/group/lcls/ds/tools/ccp4-8.0/bin/ipmosflm`), linked alone into the run's `bin/` so no other CCP4
binary is on PATH. Time per frame is CrystFEL's own `index` time (`--profile`), one core.

- Commit 95d51ce. CrystFEL 0.12.0 (`/sdf/group/lcls/ds/tools/crystfel/0.12.0`).
- Recorded run: Slurm 41172742, milano (EPYC 7713), 17 Oct 2026 (`j17.sbatch` md5 cf564a79 as run).

## Inputs

The same as `../asdf_via_indexamajig`: the 120-frame CXI inputs, their scrambled copy and the scorer
`j8_score.py` (md5 d4182cf73c292b4e4c22d907e85d260e) under `$VERIFY_INPUTS/cxidb17_harness`, not in this
branch.

## argv

```
indexamajig -i cf120_events.lst -g cf120.geom [-p lyso.cell] --peaks=cxi --indexing=mosflm --no-image-data \
  --no-revalidate --integration=none -j 1 --profile [--no-retry --no-refine --no-check-cell] [--no-check-peaks] \
  -o <arm>.stream --temp-dir=$OUT
python j8_score.py --arm <arm> --kind stream --result <arm>.stream --profile <arm>.out --events <events> \
  --frames <qeff> --out $OUT --meta <json>
python j17_table.py $OUT
```

## Null

`null_blind_alloff` and `null_cell_alloff` on the scrambled frames.

## Expected (not yet in FACTS)

| arm | returned | lattice | strict | index ms median |
|---|---|---|---|---|
| blind_on | 44 | 27 | 27 | 6896 |
| blind_nopeaks | 77 | 42 | 39 | 792 |
| blind_alloff | 73 | 41 | 32 | 770 |
| cell_on | 35 | 35 | 35 | 3674 |
| cell_nopeaks | 62 | 61 | 54 | 2417 |
| cell_off | 23 | 23 | 23 | 807 |
| cell_alloff | 68 | 55 | 44 | 811 |
| null_blind_alloff | 2 | 0 | 0 | 2176 |
| null_cell_alloff | 0 | 0 | 0 | 690 |

## Outputs

`run_<job>/`: streams, `<arm>_frames.csv`, `<arm>_summary.json`, `ARGV.tsv`, `PROVENANCE.txt`, `mosflm.md`,
`logs/`.

## Run

```
mkdir -p $VERIFY_OUT/mosflm_via_indexamajig/logs && cd $VERIFY_OUT/mosflm_via_indexamajig
sbatch $VERIFY_DIR/mosflm_via_indexamajig/j17.sbatch
```
