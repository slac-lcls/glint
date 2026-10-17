# r51_blind_arm_with_guard

The r51 arms of `../merges_tofile_vs_crystfel`, re-run with the solution-file guard of #263
(`write_solution_file` skips a frame with no usable cell). In the first run the r51 blind arm wrote no
solution file, because the writer stopped on a degenerate cell. Same chain, r51 only: GLINT blind (`GLb`)
and known-cell (`GLk`) solution files, `indexamajig --indexing=file` with the shuffled nulls, `B0` and
`N0` on the common frames, key-paired merges, the reproduction gate on LUTE's r51 merge, and
`j2_tools.py collect`. CrystFEL's own arms (`CF`, `CFnr`) are the first run's streams, linked, not re-run.
The tools (`j2_tools.py`, `j2_merge_one.sh`) are read from `../merges_tofile_vs_crystfel`.

- GLINT abe330c, the head of #263 (95d51ce + #261 docstrings + the guard); its `glint/` package is
  identical to main at b21b165. CrystFEL 0.10.2. GPU exclude list sdfampere002, 003, 038, 042 (site setting).
- Recorded run: Slurm 41172814 (GLINT, A100) -> 41172816 (`--indexing=file`) -> 41172817 (merges),
  17 Oct 2026.

## Inputs

The r51 list, geometry and cell of `../merges_tofile_vs_crystfel/config.sh` (11,058 frames), and from
`$VERIFY_OUT/merges_tofile_vs_crystfel/r51/`: `CF.stream`, `CF.keys`, `CFnr.stream`, `CFnr.keys`,
`idx_CF.log`, `idx_CFnr.log`.

## argv

As in `../merges_tofile_vs_crystfel`; every executed argv is in `commands.log`.

## Null

Frame level: `N1`-`N3`, GLb's orientations shuffled among frames, peak checks on. Merge level: `N0_1`-`N0_3`,
the shuffled orientations on the common frames with the checks off.

## Expected (not yet in FACTS)

| quantity | value |
|---|---|
| GLb solutions written; on the reference lattice (`GLb_cellcheck.json`) | 2,408; 250 |
| GLINT's consensus on r51 | declined: best cluster 269 of 32,199 pooled cells (0.8%) |
| CrystFEL accepts GLb; GLk | 242 of 2,408; 303 of 1,510 |
| shuffled orientations accepted (`N1`-`N3`) | 0 of 2,408 each |
| common frames (`C.keys`) | 241 |
| CC1/2, common set: CF; GLb; B0 | 0.222 +- 0.027; 0.217 +- 0.034; 0.216 +- 0.028 |
| CC1/2 GLb minus CF | -0.0051 +- 0.0387, 3+/7-: not detected |
| CC1/2 nulls `N0_1`, `N0_2`, `N0_3` | 0.007, -0.029, 0.024 |
| misorientation GLb against CF | median 0.0099 deg, 98.3% within 1 deg (241 frames) |
| CC1/2 known-cell set (302 frames): CF; GLk | 0.263 +- 0.045; 0.261 +- 0.049 |

Means +- sd over the ten key-paired splits, from `RESULTS.md`.

## Outputs

`r51/` (solution files, streams, keys, cell checks, `merges/`), `RESULTS.md`, `RESULTS.json`,
`provenance.json`, `commands.log`, `logs/`.

## Run

With `VERIFY_GLINT` set to an abe330c (or b21b165) tree:

```
O=$VERIFY_OUT/r51_blind_arm_with_guard; mkdir -p $O/logs $O/r51 && cd $O
for f in CF.stream CF.keys CFnr.stream CFnr.keys idx_CF.log idx_CFnr.log; do
  ln -s $VERIFY_OUT/merges_tofile_vs_crystfel/r51/$f r51/$f; done
G=$(sbatch --parsable $VERIFY_DIR/r51_blind_arm_with_guard/j18_glint.sbatch)
T=$(sbatch --parsable --dependency=afterok:$G $VERIFY_DIR/r51_blind_arm_with_guard/j18_tofile.sbatch)
sbatch --dependency=afterok:$T $VERIFY_DIR/r51_blind_arm_with_guard/j18_merge.sbatch
```

The recorded run also linked `cell.txt`, which `j18_tofile.sbatch` rewrites with the same cell.
