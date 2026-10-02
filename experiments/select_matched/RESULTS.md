# RESULTS: hybrid_index(select="matched") vs the shipped pick (PREREG.md, 4881899 + amendment 1aa5551)

Run 28 Sep 2026 at bb9c9ac on a MacBook (CPU, torch 2.13.0, numpy 1.26.4), 10 single-threaded processes in
parallel, `STEPS=8`, `OMP_NUM_THREADS=1`. Inputs `frames_cxidb_clean.txt` (`f89fa57d`) and `q480_fix.txt`
(`d6d86c1b`). Per-run matrices and stats in `runs/`, scores in `runs/summary.json`
(`measure.py score` recomputes it from the runs).

## Numbers (strict gate on the original peaks)

| arm | set | first | matched | matched − first | gained / lost | frames changing pick | known-cell searches first → matched |
|---|---|---|---|---|---|---|---|
| clean | 120 | 91 | 91 | 0 | 0 / 0 | 32 | 30 → 120 |
| clean | 480 | 366 | **370** | **+4** | 4 / 0 | 130 | 129 → 480 |
| N50 | 120 | 78 | 78 | 0 | 0 / 0 | 21 | 51 → 120 |
| N50 | 480 | 299 | **305** | **+6** | 6 / 0 | 111 | 198 → 480 |

Null (480 frames, every one azimuth-scrambled, `Mc_known` = the clean 480 consensus cell): both rules register
447 / 480 and 16 of those pass the observable gate (3.3 %), the SAME 16 frames under both rules.

Wall time of one `hybrid_index` call (processes sharing the CPU, so indicative only): 480 clean 712 s → 743 s
(+4 %), 120 clean 204 s → 213 s. The blind N-best search dominates; the extra known-cell searches are cheap.

## Predictions

| | prediction | outcome |
|---|---|---|
| P1 | matched = 91 / 370 and frame-for-frame = the 23 Sep "ADMM-lite round 0"; first = `hyb_Mc` | **HELD**: 0 mismatches on both sets, both rules |
| P2 | N50: ≥ +3 on 480 with ≤ 1 lost; ≥ 0 on 120 with ≤ 1 lost | **HELD**: +6 (6 / 0); 0 (0 / 0) |
| P3 | null: matched passes ≤ 3 more scrambled frames than first | **HELD**: 0 more (16 vs 16) |
| P4 | clean: ≥ +2 on 480, 0 lost on both | **HELD**: +4, 0 lost |

## Reading

* The shipped function with `select="matched"` IS the rule measured on 23 Sep: frame-for-frame identical on
  both sets, and `select="first"` is frame-for-frame the shipped pipeline.
* It never lost a frame in any arm (0 lost in 10 gained), and the gain grows under spurious peaks (+4 clean,
  +6 at N50 on 480), in line with the stress runs' polished variant (+6 on this draw). Without the polish the
  gain on this draw is the same.
* **P3 held, but this arm cannot tell the two rules apart:** no scrambled frame had a consensus-consistent blind
  cell, so both rules reduce to the known-cell fit on every null frame. What it does show is the known-cell
  tautology again: the search registers 447 of 480 lattice-free frames as the consensus cell, and 16 of those
  (3.3 %) pass the observable gate. That is a property of the known-cell rescue, not of `select`.
* The 120 set does not move: its 91 is already at the candidate-step ceiling C3 (RESULTS.md, 23 Sep).
* `n_idx` (registrations) is the same under both rules on every arm: `select` changes which registration a
  frame keeps, never whether it has one.

Scope: one protein (cxidb-17 lysozyme), one noise draw, CPU. Nothing here enters the paper, R1 or a talk without
S.M.; the default stays `select="first"`.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python experiments/select_matched/measure.py all \
    --f480 q480_fix.txt --out experiments/select_matched/runs          # ~18 min on 10 cores
```

`q480_fix.txt` is not in the repository (S3DF `~/q480_fix.txt`, md5 prefix `d6d86c1b`); the P1 reference flags
are read from `exp/joint-ceiling` with `git show`.
