# asdf_via_indexamajig

asdf through CrystFEL 0.12.0 `indexamajig` on the 120 cxidb-17 frames, nine arms at once, one core each:
blind and with the cell (`-p experiments/xgandalf/lyso.cell`), each at defaults (`_on`), without the peak
check (`_nopeaks`), and with retry, refinement, the cell check and the peak check all off (`_alloff`);
`cell_off` keeps the peak check with the other three off. Time per frame is CrystFEL's own `index` time
(`--profile`), one core.

- Commit 95d51ce. CrystFEL 0.12.0 (`/sdf/group/lcls/ds/tools/crystfel/0.12.0`).
- Recorded run: Slurm 41168320, milano (EPYC 7713), 17 Oct 2026 (41168245 on roma was cancelled while
  pending; the same script with the partition line changed, md5 fcd4fffa as run).

## Inputs

Under `$VERIFY_INPUTS/cxidb17_harness`, from a separate harness that is not in this branch:

- `inputs/frames120.txt` (the job checks it equals `experiments/xgandalf/frames_cxidb_clean.txt`),
  `inputs/cf120.{cxi,geom}`, `cf120_events.lst`, `cf120_qeff.txt`: the 120 frames back-projected to CXI by
  `experiments/make_cxi.py` at 95d51ce.
- `inputs/frames120_scr.txt`, `cf120scr.*`: the azimuth-scrambled copy, seed 20260928.

The scorer is `../common/score_arm.py` (md5 6d0e9408ecd6ebffd5ed4ddaafd64fd8, checked by the job): the
harness's scorer (md5 d4182cf73c292b4e4c22d907e85d260e) with the pinned tree read from `VERIFY_GLINT`, logic
unchanged. Strict = correct lattice (`same_lattice` against 79.02 79.02 37.98) and at least 25% and 10 peaks
within 0.15 of integer hkl, at the 95d51ce gate.

## argv

```
indexamajig -i cf120_events.lst -g cf120.geom [-p lyso.cell] --peaks=cxi --indexing=asdf --no-image-data \
  --no-revalidate --integration=none -j 1 --profile [--no-retry --no-refine --no-check-cell] [--no-check-peaks] \
  -o <arm>.stream --temp-dir=$OUT
python ../common/score_arm.py --arm <arm> --kind stream --result <arm>.stream --profile <arm>.out --events <events> \
  --frames <qeff> --out $OUT --meta <json>
python j15_table.py $OUT
```

The exact argv of each arm is written to `ARGV.tsv`.

## Null

`null_blind_alloff` and `null_cell_alloff`: the two `_alloff` arms on the scrambled frames.

## Expected (not yet in FACTS)

| arm | returned | lattice | strict | index ms median |
|---|---|---|---|---|
| blind_on | 27 | 0 | 0 | 2676 |
| blind_nopeaks | 120 | 0 | 0 | 342 |
| blind_alloff | 115 | 0 | 0 | 282 |
| cell_on | 43 | 43 | 43 | 2003 |
| cell_nopeaks | 71 | 71 | 71 | 777 |
| cell_off | 43 | 43 | 43 | 342 |
| cell_alloff | 111 | 64 | 64 | 340 |
| null_blind_alloff | 110 | 0 | 0 | 314 |
| null_cell_alloff | 98 | 0 | 0 | 395 |

## Outputs

`run_<job>/`: streams, `<arm>_frames.csv`, `<arm>_summary.json`, `ARGV.tsv`, `PROVENANCE.txt`, `asdf.md`
(the table above), `logs/`.

## Run

```
mkdir -p $VERIFY_OUT/asdf_via_indexamajig/logs && cd $VERIFY_OUT/asdf_via_indexamajig
sbatch $VERIFY_DIR/asdf_via_indexamajig/j15.sbatch
```
