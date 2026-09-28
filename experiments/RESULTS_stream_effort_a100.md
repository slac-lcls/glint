# StreamDriver `effort=` on one A100 — job 39344280 (28 Sep 2026)

Branch feat/adaptive-effort at 70f5c61 (this note and `experiments/scramble_qlist.py` came after the run, 8d43cf8),
S3DF sdfampere013 (A100-SXM4-40GB, exclusive), ana-4.0.58-py3-minipytorch: cupy 12.3.0, torch 2.1.0. Outputs in
`~smarches/glint_esc_outputs/effort/` (`effort_gpu_39344280.out`, `replay_*.json`). The whole job took 2.5 minutes.

## Tests on the node

`experiments/test_stream_effort.py` 7/7 (numpy path); `GLINT_TEST_GPU=1 experiments/test_stream_hit_ring.py` 10/10
(device path; the fast-path call signature changed in this branch) and 10/10 on the numpy path.

## The published 480 streaming arm, with and without the policy

`record_stream_replay.py --input lyso=q480_fix.txt --B 20 --dmin 2.0 --warmup-rescue --adaptive-relock --min-inliers 10`,
plus `--effort '{"rate_hz": R, "n_gpu": 1}'` for the effort arms. B=20, so these check the MECHANISM (tier choice,
logging, escalation accounting); the tier costs the policy reasons with are B=120 numbers.

| arm | tier in force | strict / 480 | indexed | watchdog rescues | relocks | deep search |
|---|---|---|---|---|---|---|
| `effort=None` (gated 333 / 10 / 1) | shipped | 333 | 453 | 10 | 1 | – |
| `rate_hz=35000` (0.029 ms per hit) | 0 | 333 | 453 | 10 | 1 | off |
| `rate_hz=2000` (0.5 ms per hit) | 2 | 360 | 467 | 3 | 0 | on/off with the miss fraction; 4 tried, 1 accepted, 12 searches |
| `rate_hz=120` (8.33 ms per hit) | 3 | 376 | 467 | 1 | 0 | off (top tier, by rule) |

- The 35 kHz arm is decision-identical to the default path (same counts, same cells, same events): tier 0 is the
  shipped search with its depth written out.
- At tier 2 the deep search switched on at 125 and 385 and off at 185 and 425 (miss fraction 0 → 0.025 → 0): at a
  0.5 ms budget it needs miss ≤ 1.2 % with k_null=8. It took 1 of the 4 misses it saw (1 real + 8 copy searches for the
  accept, 1 search each for the three fits that failed the gate).
- The published arm's relock at 405 (cell1 = 87.5 / 87.6 / 109.5 Å, 3 frames) does not happen at tiers 2 and 3: the
  deeper fast path leaves the watchdog too few misses to vote on.
- Wall times from the log: 10 to 11 s per arm; the recorder does not time frames at B=20.

## The scrambled-null arm: how many lattice-free frames does each tier accept?

The same real 480 first (the driver locks on them, locked_after 5), then the same 480 with every peak rotated by its
own random azimuth about the beam (`experiments/scramble_qlist.py`, frame i with `default_rng([20260928, i])`; |q|
and q_z kept, so the Ewald excitation of every peak is unchanged; md5 bd132bd01f202178aa6b3274d8393c64) as a second
species `scr`, scored against the lysozyme cell. The scrambled frames have the real frames' peak counts (median 157).

| arm | tier | scr indexed / 480 (live gate) | of which strict | scr escalated |
|---|---|---|---|---|
| `effort=None` | shipped | 232 (48.3 %) | 26 (5.4 %) | – |
| `rate_hz=35000` | 0 | 232 (48.3 %) | 26 (5.4 %) | 0 |
| `rate_hz=2000` | 2 | 235 (49.0 %) | 38 (7.9 %) | 0 |
| `rate_hz=120, n_gpu=2` | 3 | 241 (50.2 %) | 46 (9.6 %) | 0 |

- The live gate (`min_inliers=10`, `min_inlier_frac=0.15`) accepts about half of the lattice-free frames at every
  depth, the shipped one included. That is the count bar being at chance on dense frames (the 16 Sep calibration on
  cxidb-17's 816 hits: real 764 vs null 738 under a ≥ 10 bar; floor 0.057·n + 9.3 at 1 % null), a property of the
  gate, not of this branch. Depth adds 1 to 2 points (232 → 235 → 241).
- Under the paper's strict gate the chance rate rises 5.4 → 7.9 → 9.6 % with depth: the depth sweep's 5 → 8.5 %.
- In the default-path null arm, 22 of the 232 accepts went to the spurious relock cell; at tiers 2 and 3 there is no
  such cell and all accepts go to lysozyme.
- The deep search was not exercised on the scrambled frames: at 2 kHz the policy switched it off once the misses
  became frequent, and at 120 Hz the top tier is chosen, where there is no deep search by rule. Under the default
  tiers it is a rare-miss feature (miss ≤ (budget − tier cost) / ((1 + k_null) · 0.94 ms)). Its acceptance is
  null-controlled per frame by construction: a lattice-free frame beats all k_null exchangeable copies with
  probability ≤ 1/(k_null + 1) — 1/9 at the k_null=8 this job ran with, before the live gate; glint#211 measured 0
  accepts on scrambled copies at k_null=32, now the default.

## Reading

The policy does what it says: the tier follows the budget, tier 0 is bit-for-bit the shipped path, deeper tiers
buy 27 and 43 more strict indexings on the 480 at B=20 (of which about 4 and 6 would be the chance share if the 147
frames the shipped depth leaves un-indexed behaved like lattice-free frames: the strict chance rate rises 2.5 and 4.2
points with depth; an upper estimate, since most of those frames hold real crystals, and the 480 has no per-frame
reference orientation to settle it directly), decisions are logged at flush boundaries, and the deep
search's accounting is visible. What the null arm adds is a caution that does not belong to this branch but must
travel with it: the live gate's chance-accept rate on lattice-free dense frames is ~48 % at any depth, so yields
quoted from the driver's `indexed` are live-gate yields and the strict column is the one to compare; the fix is the
calibrated per-peak-count floor in `_fits`, a follow-up. The shipped `k_null` default was changed to 32
(glint#211's measured setting) after this run, on the maintainer's decision; the job ran at the then-default 8, so a
rerun of the 2 kHz arm as recorded passes `"k_null": 8` (at 32 the deep search needs miss ≤ 0.3 % at a 0.5 ms budget
and would have stayed off).
