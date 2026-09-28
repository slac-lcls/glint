# PREREG: per-frame selection by matched peaks in `hybrid_index` (`select="matched"`)

Written 28 Sep 2026 on `feat/select-matched` (base main 93b245f), committed before any measurement below.

## The change

`hybrid_index` step (3)–(4) today keeps, per frame, the FIRST blind N-best cell that is `same_lattice` with the
consensus cell Mc (N-best order = blind score), and runs the known-cell search `index_known_gpu_cell(q, Mc)` only
on frames that have no such cell. `select="matched"` instead forms, for every frame, the candidate set

    {every blind N-best cell same_lattice with Mc}  ∪  {index_known_gpu_cell(q, Mc), if same_lattice with Mc}

and keeps the one with the most `matched_strict` peaks (ties: first in that order, i.e. blind order, then the
known-cell fit). No polish, no cell update. `select="first"` (the default) is the shipped rule, unchanged.

This is the rule the 23 Sep harness measured as "ADMM-lite round 0" (`experiments/joint_ceiling/joint_ceiling.py`
on `exp/joint-ceiling`, prereg 5fea098, results 2cbfd29): 91 → 91 on the 120 set, 366 → 370 on the 480 set, CPU.
The 25–26 Sep stress runs measured a POLISHED variant of it (+7.0 [+5.9, +8.1] frames/480 at N50, 42 gained /
0 lost over six draws); the unpolished rule under noise has not been measured.

## Inputs

* `experiments/frames_cxidb_clean.txt` (md5 prefix `f89fa57d`, the 120 set) and `q480_fix.txt` (`d6d86c1b`, the
  480 set; not in the repository), frames with ≥ 6 peaks, as in every joint_ceiling run.
* CPU, `CUDA_VISIBLE_DEVICES=`, `STEPS=8`, `OMP_NUM_THREADS=1`, nbest 3, no triage, no alias gate, no cascade, no
  escalation.
* Strict score (truth-using, reporting only): `same_lattice(M, LYSO)` ∧ `matched_strict(M, q) ≥ GATE_MIN` ∧
  `matched_strict / len(q) ≥ GATE_FRAC`, on the frame's ORIGINAL peaks.
* Observable gate (no truth): `matched_strict ≥ GATE_MIN` ∧ `≥ GATE_FRAC · len(q)`.

## Arms

1. **Clean**, both sets, `select` ∈ {first, matched}, with `KC_FP=64` (the 23 Sep runs predate the fp32 known-cell
   default, #209) and with the default fp32.
2. **Noise N50**, both sets: `stress.py`'s `perturb(frames, "N50", 20260925)` verbatim (50 % extra peaks borrowed
   from other frames, azimuth-scrambled; rng `[20260925, 3, i]`), scored on the original peaks; default fp32.
3. **Null**, 480 set: every frame azimuth-scrambled (`glint.multilattice.scramble_azimuth`, rng
   `np.random.default_rng([20260928, i])`), indexed with `Mc_known` = the consensus cell of the clean 480 run;
   count frames passing the OBSERVABLE gate (every one of them is a chance registration); default fp32.

## Predictions

* **P1 (the code is the measured rule).** Arm 1 with `KC_FP=64`: `select="matched"` gives 91/120 and 370/480
  strict, and its per-frame strict flags equal `per_frame.admm[0]` of `results_{120,480}.json` exactly (0
  mismatches); `select="first"` equals `per_frame.hyb_Mc` exactly.
* **P2 (noise).** Arm 2, matched − first: ≥ +3 on 480 with ≤ 1 frame lost; ≥ 0 on 120 with ≤ 1 lost.
* **P3 (chance).** Arm 3: matched passes at most 3 more scrambled frames than first (of 480).
* **P4 (clean, fp32 default).** Arm 1 fp32: matched − first ≥ +2 on 480, 0 lost on both sets.

Reported, no prediction: wall time of `hybrid_index` per arm (matched runs the known-cell search on every frame
instead of only the frames without a consistent blind cell), the number of known-cell searches, and how many
frames change pick.

## Decision rules

* P1 fails → stop; the implementation is not the measured rule, and nothing else is reported until it is.
* P2, P3 or P4 fails → reported as failed in RESULTS.md and in the PR text. The PR stays opt-in (`select="first"`
  default) whatever happens; making `matched` the default is a separate decision for S.M., not this PR.
* Nothing here enters the paper, R1 or a talk without S.M.
