# r51_merge_vs_frames

Merge statistics of mfx100848724 r51 against the number of frames merged, on LUTE's CrystFEL stream
(CrystFEL's own indexing, 251 frames with a crystal, 280 crystals). Paired random subsets of 10, 20, 35,
50, 75 and 100% of the frames, seeds 1-8 (seed 0 = stream order, 100% only), each merged with partialator
and with process_hkl; FOMs on the fixed resolution range of the 100% stream-order merge. Sweep values are
medians over the 8 seeds. No GLINT output enters stage A.

- Commit 95d51ce (the environment, numpy only). CrystFEL 0.10.2 (`/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin`).
- Recorded run: Slurm 41011666 (stage A), sdfmilan137, 16 Oct 2026.

## Inputs

- Stream: `/sdf/data/lcls/ds/prj/prjdat21/scratch/lute_sfx_val4/sfx_mfx100848724_r51_crystfel/mfx100848724_r51_crystfel_all.stream`
  (md5 8c89d5e8f72282e9b00bc677be6f37d0).
- LUTE's merge of the same stream, `mfx100848724_r51_crystfel.hkl{1,2}` in the same directory (reproduction check).
- Cell: `/sdf/group/lcls/ds/tools/lute/test_utilities/sfx/lyso_tetragonal_p43212.cell` (md5 dfb8e692b12ce33ca3d1c1dc4d75639a).

## argv

```
partialator -i in.stream -o merged.hkl -y 4/mmm --model=unity --min-measurements=2 --max-rel-B=100.0 -j 4
process_hkl -i in.stream -o merged.hkl[1|2] -y 4/mmm --scale [--odd-only|--even-only]
compare_hkl merged.hkl1 merged.hkl2 -y 4/mmm -p <cell> --fom={CC,CCstar,Rsplit} --nshells=10 [--lowres=DLO --highres=DHI]
check_hkl merged.hkl -y 4/mmm -p <cell> --nshells=10 [--lowres=DLO --highres=DHI]
```

Every call is appended to `A/commands.log`.

## Null

Each crystal's (I, sigma) permuted over its own reflections (`perm=all`), or within blocks of 20 neighbors
in 1/d, which keeps the Wilson falloff (`perm=shell`); merged with partialator.

## Expected (FACTS keys)

| key | value |
|---|---|
| `r51_stream_frames`, `r51_stream_crystals` | 251, 280 |
| `r51_stream_cchalf`, `r51_stream_ccstar`, `r51_stream_rsplit_pct`, `r51_stream_pairs` | 0.2742221, 0.6560602, 93.16, 2120 (equal to LUTE's merge) |
| `r51_sweep_lo_frames`, `_cchalf`, `_ccstar`, `_rsplit_pct` | 88, 0.1396, 0.5540, 108.08 |
| `r51_sweep_hi_frames`, `_cchalf`, `_ccstar`, `_rsplit_pct` | 251, 0.2890, 0.6697, 91.27 |
| `r51_sweep_hi_cchalf_min`, `_max` | 0.2163, 0.3665 |
| `r51_sweep_hi_rsplit_min`, `_max` | 87.57, 96.28 |
| `r51_sweep_hi_null_cchalf`, `r51_sweep_hi_null_shell_cchalf` | -0.0083, 0.0580 |
| `r51_sweep_hi_null_rsplit_pct`, `r51_sweep_hi_null_shell_rsplit_pct` | 111.78, 105.98 |

Below 88 frames only 6-7 reflection pairs remain, so those rows are not measurements. The FACTS values
are partialator's.

## Outputs

`A/r51CF/` (`frames.json`, `input.md5`, `input_indexamajig_argv.txt`, `fixed_range_A.txt`, `ref/`, `merges/`),
`A/repro.txt`, `A/results.json`, `A/sweep.tsv`, `A/analysis.json`, `A/provenance.json`, `logs/`.

## Run

```
mkdir -p $VERIFY_OUT/r51_merge_vs_frames/logs && cd $VERIFY_OUT/r51_merge_vs_frames
sbatch $VERIFY_DIR/r51_merge_vs_frames/sweep.sbatch
```

Not run: stage B (`sbatch --export=ALL,STAGE=B ...`), which reads the r51 CF and GLb streams of
`../merges_tofile_vs_crystfel`.
