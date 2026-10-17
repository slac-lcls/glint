# cxidb45_dials_head_to_head

GLINT and DIALS on the same peak lists: cxidb-45 Proteinase K, 907 frames x 170 peaks
(`experiments/prok_q.npz`), reference P4(3)2(1)2 68.7 68.7 108.6. Every arm is scored by GLINT's gate
at 95d51ce with the ProK reference: correct lattice and at least 25% and 10 of the 170 peaks within 0.15
of integer hkl, so at least 43 matched peaks. DIALS gets the peaks through a virtual flat panel
(`dials_samepeaks.py`); spot finding and integration are not exercised.

Two runs share this folder:
- the `stills_process` run (`stills_*`, `submit_stills.sh`): DIALS `stills_process` arms (`phil/`), the GLINT
  arms, the null input, the scorer;
- the routes run (`routes_*`): the bare basis-vector routes behind the earlier
  length-gate DIALS rows (best of 25 candidates, no accept step) and DIALS's own `Indexer`, graded by the
  same scorer, with the `stills_process` outputs read back.

- Commit 95d51ce. No CrystFEL. DIALS 3.dev.1332-g3e144e7ea (`/sdf/group/lcls/ds/tools/cctbx-psana2/build`,
  `dials.python`), one EPYC 7713 core per arm; GLINT on one A100 in `ana-4.0.58-py3-minipytorch`.
- Recorded runs: Slurm 41011687 (null input), 41011688 (DIALS, array 0-19), 41011689 (GLINT, sdfampere035),
  41011690 (score), 16 Oct 2026; routes 41099183 (array 0-8, sdfmilan183 and sdfmilan223) and 41099184
  (score), 16 Oct 2026.

## Inputs

| input | sha256 |
|---|---|
| `$VERIFY_GLINT/experiments/prok_q.npz` | e4ac4f5486ba44005c36ab1bd2dc27c6e1527788d7c69f550ff5be5141791a75 |
| `inputs/prok_q_scrambled_s20260928.npz`, made by `prep_null.py` | 8e690c25d474ffc10b61aeb39d02d10707bdc325732f456166f19cd2396a4df6 |

The routes sbatch stops if either hash differs.

## argv

```
python -u prep_null.py --tree $VERIFY_GLINT --npz $VERIFY_GLINT/experiments/prok_q.npz --out inputs/prok_q_scrambled_s20260928.npz --seed 20260928
dials.python dials_samepeaks.py --npz <npz> --phil phil/common.phil phil/<ARM>.phil --arm <ARM> --start S --stop E --out dials/<ARM>/frames_S_E.jsonl --workdir ... --rank C
python -u glint_samepeaks.py --npz <prok_q.npz> --npz-null <scrambled> --out glint
python -u stills_score.py --npz <prok_q.npz> --npz-null <scrambled> --dials dials --glint glint --out score
dials.python -u routes_dials.py --arm <ARM> --npz <prok_q.npz> --npz-null <scrambled> --out dials_routes/out/<ARM>/frames.jsonl
python -u routes_score.py --npz ... --npz-null ... --dials dials_routes/out --stills-dials dials --stills-glint glint --out dials_routes/score
```

## Null

The lists azimuth-scrambled per peak (`scramble_qlist.scramble_frames`, seed 20260928): the
`N2_kc_xfel_scrambled` DIALS arm, `G_pf_null`, `G_hyb_null`, `G_kc_null`, and the routes' `V1n`, `V2n`, `S3n`.

## Expected (FACTS keys)

| key | arm | value |
|---|---|---|
| `pk45_glint_perframe_strict`, `_ms` | `G_pf` (`index_blind_fast` per frame) | 324, 6.0 ms |
| `pk45_glint_consensus_strict`, `_ms` | `G_hyb` (`hybrid_index`, no cell) | 804, 24.15 ms (batch wall / 907) |
| `pk45_glint_known_strict` | `G_kc` (the cell given) | 805 |
| `pk45_dials_xfel_strict`, `_ms` | `K2_kc_xfel` | 675, 1307.0 ms |
| `pk45_dials_defaults_strict` | `K1_kc_fft1d` | 605 |
| `pk45_dials_grid_refined_strict` | `K3_kc_rsgs` | 552 |
| `pk45_dials_auto_maxcell_strict` | `B0_blind_fft1d` | 42 |
| `pk45_dials_maxcell200_strict`, `_ms` | `S3_idx_fft1d_rac_mc200` | 446, 1446.0 ms |
| `pk45_dials_best25_strict`, `_ms` | `V1_bvs_fft1d_blind` | 535, 360.0 ms |
| `pk45_dials_grid_norefine_strict`, `_ms` | `V2_bvs_rsgs_kc` | 263, 319.0 ms |
| `pk45_mcnemar_glint_only`, `_dials_only`, `_p` | `G_hyb` against `K2_kc_xfel` | 138, 9, 8.2e-31 |
| `pk45_null_strict` | every null | 0 |
| `pk45_glint_known_null_lattice` | `G_kc_null`, correct lattice only | 896 |
| `pk45_*_lengate` | length-only gate | GLINT 333, 895, 893; DIALS 682, 605, 577 |

Regression checks the routes scorer prints: `V1` length gate 561 and full gate 548, `V2` length gate 453
(the earlier rows' counts).

## Outputs

`inputs/`, `dials/<ARM>/frames_*.jsonl`, `glint/<arm>.npz`, `glint/provenance.json`, `score/results.{json,md}`;
`dials_routes/out/<ARM>/frames.jsonl` (+ `.provenance.json`), `dials_routes/score/results.{json,md}`; `logs/`.

## Run

```
bash $VERIFY_DIR/cxidb45_dials_head_to_head/submit_stills.sh
# after it ends:
mkdir -p $VERIFY_OUT/cxidb45_dials_head_to_head/dials_routes/logs && cd $VERIFY_OUT/cxidb45_dials_head_to_head/dials_routes
A=$(sbatch --parsable $VERIFY_DIR/cxidb45_dials_head_to_head/routes_dials.sbatch)
sbatch --dependency=afterany:$A $VERIFY_DIR/cxidb45_dials_head_to_head/routes_score.sbatch
```

The two NERSC drivers that the `bvs` route ports are not in this folder. Neither run merges: the FACTS
cxidb-45 merge (CC* 0.90, R_split 31%, <I/sigma> 7.8) is not re-derived here.
