# merges_tofile_vs_crystfel

GLINT's blind orientations handed to CrystFEL for integration, against CrystFEL's own indexing of the
same frames, merged the same way. GLINT writes a solution file (`glint_cli --tofile`, `--gate strict`);
`indexamajig --indexing=file` reads it, refines, checks and integrates. A row compares the frames both
routes index, split ten times into half-datasets assigned by frame key (`partialator --custom-split`,
one seeded assignment shared by every arm). CC1/2, CC* and Rsplit are means over the ten splits; crystals
merged and the resolution are medians. A difference counts as detected when one sign holds in at least 8
of 10 splits and the mean exceeds CrystFEL's own split sd (`j2_tools.py verdict`).

- Commit 95d51ce. CrystFEL 0.10.2 (`/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin`). GLINT environment
  `ana-4.0.58-py3-minipytorch`; the r0016 `.cxi` is rebuilt with btx 0f1af1d in `ana-4.0.47-py3`.
- Recorded run: Slurm 41037259 (r0016 `.cxi`), 41037260 (GLINT, sdfampere010, A100), 41037261 (CrystFEL
  indexing, sdfmilan136), 41037262 (`--indexing=file`, sdfmilan177), 16 Oct 2026; merges 41130408
  (sdfmilan135), 17 Oct 2026. The first merge pass, 41037263, split halves by chunk order and is superseded.
  The indexing jobs ran with these files before `cmd_splitfile` and the `cmd_collect` fix were added;
  only the merge step uses either.

## Inputs (paths in `config.sh`)

| dataset | frames | input | geometry, cell |
|---|---|---|---|
| r51, mfx100848724, Jungfrau-16M | 11,058 with >= 10 peaks (events 0-17,871) | LUTE FindPeaksSFX `.cxi` list (round 7) | LUTE `mfx100848724_356mm_refined.geom`, `lyso_tetragonal_p43212.cell` (79.17 79.17 37.96) |
| r194, mfx101343025, Jungfrau-16M | 278 (first 3,000 events) | LUTE `.cxi` list (round 5) | `mfx101343025_r0139_refined.geom`, `maybe_lyso.cell` (28.0 62.5 60.9, beta 90.8) |
| r0016, mfxx49820, Epix10ka2M | first 6,000 events | rebuilt by `j2_r0016_cxi.py` (btx PeakFinder, r0016 settings) | btx `r0016.geom` (md5 0deaac06), `sample2.cell` |

Reproduction gate (must pass before any comparison merge): LUTE's r51 GLINT stream merged by this
harness gives CC 0.1751131 on 2,686 pairs.

## argv

```
python -m glint.glint_cli --images LIST --geom GEOM --peakfinder stored --gate strict --lattice LATT \
  --tofile GLb.sol -o GLb_orient.stream                      # GLk: the same + --cell a b c al be ga
indexamajig -i LIST -o CF.stream --peaks=cxi -g GEOM -p CELLFILE --indexing=xgandalf --tolerance=5,5,5,1.5 --multi \
  --integration=rings-nocen --fix-divergence=0.0 --int-radius=3,4,6 --int-diag=none --push-res=infinity \
  [--no-revalidate on r0016] -j 64 --wait-for-file=0
indexamajig -i GLb.lst -o GLb.stream --peaks=cxi -g GEOM -p CELLFILE --indexing=file --tolerance=10,10,10,3 \
  --fromfile-input-file=GLb.sol <same integration flags> -j 32 --wait-for-file=0
partialator -i in.stream -o merged.hkl -y SYM --model=unity --min-measurements=2 --max-rel-B=100.0 --custom-split=split.txt -j 1
compare_hkl merged-halfA.hkl merged-halfB.hkl -y SYM -p CELLFILE --fom={CC,CCstar,Rsplit} --nshells=10
check_hkl merged.hkl -y SYM -p CELLFILE --nshells=10
```

Every executed argv is appended to `commands.log`.

## Null and controls

- Frame-level null `N1`-`N3`: GLINT's orientations shuffled among frames (seeds 1-3), same argv.
- Merge-level null `N0_1`-`N0_3`: the shuffled orientations on the common frames with
  `--no-check-peaks --no-refine --no-retry`; `B0`: the real orientations with the same flags.
- `CFnr` (CrystFEL with `--no-revalidate` on r51 and r194) and `GLk` (GLINT given the cell).

## Expected (FACTS keys)

| key | value |
|---|---|
| `merge_r0016_common_frames` | 998 |
| `merge_r0016_glint_cchalf`, `merge_r0016_cf_cchalf` | 0.370 (sd 0.055), 0.323 (sd 0.127) |
| `merge_r0016_cchalf_diff`, `_sd`, `_pos`, `_neg` | 0.0468, 0.1274, 5, 5 (not detected) |
| `merge_r0016_null1_cchalf`, `_null2_`, `_null3_` | 0.023, 0.028, 0.025 |
| `merge_r0016_misorient_le1deg_pct` | 97.4 |
| `merge_r194_common_frames` | 114 |
| `merge_r194_glint_cchalf`, `merge_r194_cf_cchalf` | 0.211, 0.413 |
| `merge_r194_cchalf_diff`, `_sd`, `_pos`, `_neg` | -0.2018, 0.2265, 3, 7 (not detected) |
| `merge_r194_glint_compl_pct`, `merge_r194_cf_compl_pct` | 2.0, 2.1 |
| `merge_r0016_glint_accepted` of `_solutions` | 1038 of 1548 |
| `merge_r194_glint_accepted` of `_solutions` | 123 of 230 |
| `merge_shuffled_checked_accepted` of `merge_shuffled_checked_of` | 0 of 1778, each shuffle seed |
| `merge_r51_frames`, `merge_r51_top_group`, `merge_r51_pooled_cells` | 11058, 269, 32199 |
| `merge_r51_correct_lattice` | 250 |

The block also carries d, CC*, Rsplit and completeness for both runs. r51 has no paired row here: the
solution-file writer stopped on a degenerate cell (fixed in #263; see `../r51_blind_arm_with_guard`), so
the r51 keys come from 41037260's log and orientation stream. r194's GLINT-vs-CrystFEL misorientation reads
179.79 degrees because each frame is written in one of two near-equivalent settings; it is not quoted.

## Outputs

`<dataset>/` (streams, keys, solution files, cell checks, `merges/<arm>/`), `RESULTS.md`, `RESULTS.json`,
`provenance.json`, `commands.log`, `logs/`.

## Run

```
bash $VERIFY_DIR/merges_tofile_vs_crystfel/submit_j2.sh          # the five jobs, chained
# to repeat only the merges once the streams exist:
cd $VERIFY_OUT/merges_tofile_vs_crystfel && sbatch --chdir=$PWD $VERIFY_DIR/merges_tofile_vs_crystfel/j2_merge.sbatch
```

`dryrun.sh` is the interactive check on a few frames. Not run here: a cxidb-45 arm (its images are at
NERSC), so the FACTS cxidb-45 merge (CC* 0.90, R_split 31%, <I/sigma> 7.8) is not re-derived.
