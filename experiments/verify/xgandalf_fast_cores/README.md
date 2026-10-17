# xgandalf_fast_cores

Fast XGANDALF timed per frame on one pinned core, and as throughput on 32 concurrent pinned physical cores,
on the cxidb-17 peak lists. XGANDALF is `libxgandalf.so` of CrystFEL 0.12.0 called through `xg_driver`
(`indexamajig` is not run); fast = sampling pitch 2 and 3 gradient-descent iterations, as CrystFEL's
`--xgandalf-fast-execution`. Host: S3DF ampere node, 2 x AMD EPYC 7542, CPUs only.

Existing keys this check sits beside: fast 96/120 at the lattice bar and 88/120 strict against the
defaults' 94/120 and 86/120 (re-scored here); against GLINT-(1)'s 92/120, 7:3 discordant frames (McNemar
p = 0.344); and `xgandalf_fast_speedup` 40.8x against GLINT's `blind_ms` (the last two not re-derived here).

- Commit 95d51ce (FACTS values and the scorer are read from the pinned tree; no GLINT indexing runs).
- CrystFEL 0.12.0 (`/sdf/group/lcls/ds/tools/crystfel/0.12.0`).
- Recorded run: Slurm 41011531 on sdfampere002, 64 logical = 32 physical cores, 16 Oct 2026.

## Inputs

- 120 frames: `$VERIFY_GLINT/experiments/frames_cxidb_clean.txt` (sha256 475f6802...).
- 480 frames: `$VERIFY_INPUTS/q480_fix.txt` (md5 d6d86c1b).
- Controls (earlier driver outputs): `experiments/xgandalf/xg_blind120.txt`, `$VERIFY_INPUTS/xg_fast/A_default.txt`,
  `$VERIFY_INPUTS/xg_fast/B_fast.txt`, `$VERIFY_INPUTS/xgd480_fix.txt`.
- The driver: build `bin/xg_driver` in this folder from `xg_driver.cpp` (sha256 cd4f6b4f..., the driver of
  job 37187585 that produced the `xgandalf_fast_*` keys). The recorded binary had sha256 13b43eac..., built
  on sdfiana027 with g++ 8.5.0:

  ```
  C=/sdf/group/lcls/ds/tools/crystfel/0.12.0
  mkdir -p bin && g++ -O2 -std=c++14 -I$C/include -I$C/include/eigen3 xg_driver.cpp -o bin/xg_driver \
    -L$C/lib64 -lxgandalf -Wl,-rpath,$C/lib64
  ```

## argv

```
taskset -c <cpu> bin/xg_driver <frames> blind 2 3          # fast; the defaults arm D1_24 uses: blind 6 4
python3 xg_analyze.py $RUN --glint $VERIFY_GLINT --f120 <120 list> --f480 <480 list> --sub24 <24-frame subset> --scr120 <scrambled 120>
```

Arms: `S1_120`, `S1_480`, `S1_120b` (one core; `S1_120b` repeats `S1_120` at the end as a drift check),
`D1_24` (defaults, one core), `P32_120`, `P32_480` (32 instances started together), `Q32_120` (x3),
`Q32_480` (a 32-worker queue), `N32_120`. Every command is written to `commands.log`. Strict bar:
`experiments/xgandalf/score_xg_gate.py`, `same_lattice` against 79.02 79.02 37.98 and at least 25% of the
frame's peaks within 0.15.

## Null and controls

- Null `N32_120`: the 120 lists q-scrambled (`scramble_qlist.py --seed 20260928`).
- Controls: the scorer reproduces the existing keys from the earlier outputs (`xgandalf_blind_strict_of120`
  86, `xgandalf_lattice_of120` 94, `xgandalf_fast_strict_of120` 88, `xgandalf_fast_lattice_of120` 96,
  `xgandalf_blind_strict_of480` 350).

## Expected

| key | value |
|---|---|
| `xgandalf_fast_1core_ms` | 1137.7 (median per frame of `S1_120`; mean 1392.2) |
| `xgandalf_fast_strict_of120`, `xgandalf_fast_lattice_of120` | 88, 96 in every 120-frame arm, bases identical to `S1_120` |
| not in FACTS | `P32_120` 22.287 frames/s (44.87 ms effective per frame); `N32_120` 0 of 120 strict and lattice; `S1_120b` / `S1_120` mean 1.006 |

## Outputs

`run_<job>/summary.md`, `summary.json`, `commands.log`, `provenance.txt` (hashes, lscpu, cores, load),
`raw/` (per-instance outputs and timestamps, a 20 s CPU clock trace), `logs/`.

## Run

```
mkdir -p $VERIFY_OUT/xgandalf_fast_cores/logs && cd $VERIFY_OUT/xgandalf_fast_cores
sbatch $VERIFY_DIR/xgandalf_fast_cores/xg_cores.sbatch
```

`DRYRUN=1` runs every arm on 4 frames with 2 instances.
