# The live gate on a lattice-free null, and a chance floor for it

**Question.** StreamDriver accepts a registration into its merge when the live gate `_fits` passes: at least
`min_inliers` near-integer peaks (10 in the published 480 arm) and at least `min_inlier_frac` = 0.15 of the
frame's peaks, at HKL_TOL 0.15. The 0.15 was chosen against a *wrong cell* on 16 synthetic frames. How often
does a frame with *no lattice*, searched with the *right* cell, pass it? And what gate holds that at about 1%?

Code: `experiments/live_gate_null.py` (measure / fit / confirm). Per-frame data: `live_gate_null_480.npz`,
`live_gate_null_120.npz` (arrays `n`, `inl_real`, `inl_null`, `inl_fit[k]`, `strict`, `M_real`). Opt-in gate:
`StreamDriver(null_floor=...)`, test `experiments/test_live_gate_floor.py`.

## Setup

- **Frames.** `q480_fix.txt` (S3DF `~/q480_fix.txt`, md5 `d6d86c1b…`): 480 cxidb-17 frames, peakfinder8 peaks,
  fixed λ 1.322216 Å, 38–1010 peaks, median 117. Transfer check: the committed `frames_cxidb_clean.txt`
  (120 frames, the Table 2 peak list, a different peak list from the same run).
- **Registration.** The driver's own call and count: `replica_gpu_batch.index_fused(qs, Mc, B=20)` at the
  shipped depth, a None or |det| < 1 result is a miss, else `StreamDriver._inliers`. Cell = the 480 arm's primary
  lock (78.706/78.792/37.813 Å, 89.81/90.08/90.19°, job 39344280). CPU torch 2.13, fp32 (the KC_FP default),
  where `index_fused` runs the eager batched engine.
- **Null.** Each peak rotated by its own random azimuth about the beam (`multilattice.scramble_azimuth`: |q|
  and q_z kept, so each peak's excitation error; lattice removed). Held-out copy: rng `[20260928, i]`, the
  stream of `scramble_qlist.py` (glint#213), whose `q480_scrambled.txt` (md5 `bd132bd0…`) gives identical counts
  on all 480 frames. Fit copies: 32 per frame, rng `[1, i, k]`, 15,360 in all, used only to fit.
- **CPU ↔ GPU.** Against job 39344280 (A100, fused kernels, same inputs): the live gate accepts 211 held-out
  null frames here and 210 there under the primary cell (207 in common); per-frame counts are identical on 182
  of those 210 and within ±2 on 202; on the real frames identical on 426 of 440, within ±2 on 436.

## The live gate at chance

| gate | real accepted | strict frames kept (of 327) | held-out null accepted (of 480) | fit copies |
|---|---|---|---|---|
| live (10, 0.15), shipped | 447 | 327 | **211 (44%)** | 43.7% |
| strict count/fraction (10, 0.25) | 328 | 327 | 29 (6.0%) | 4.7% |

The chance accepts are **sparse** frames, not dense ones: by peak count the live gate passes 96% of null
copies below 60 peaks, 94% at 60–90, 42% at 90–130, 2.5% at 130–200 and none above 200. Above ~130 peaks
the 0.15 fraction does its job; below, a count of 10 is what the search finds by chance. The null's 99th
percentile rises from 14 inliers at 40 peaks to 22 at 125 and 26 at 190.

The "refuses every wrong-cell frame" basis of 0.15 does not hold on real frames either: the real lysozyme
frames searched with the Proteinase K cell (68.7/68.7/108.6) pass the live gate **204/480** times.

## Candidates, fitted at a 1% null

| gate (all include the live gate) | real | strict kept | strict lost | held-out null | fit copies | runtime cost |
|---|---|---|---|---|---|---|
| (a) floor 0.0224 n + 5.32 + 1.211 √n | **335** | **305** | **22** | **2 (0.4%)** | 0.63% | none |
| (a) floor, straight line 0.0686 n + 12.51 | 332 | 302 | 25 | 3 (0.6%) | 0.61% | none |
| (b) min_inlier_frac 0.290 | 280 | 280 | 47 | 4 (0.8%) | 0.85% | none |
| (c) beat every one of 32 scrambled copies | 335 | 304 | 23 | 5 (1.0%) | — | 32 searches / frame |
| (c) … of 16 copies | 340 | 306 | 21 | 14 (2.9%) | — | 16 searches / frame |
| (c) … of 8 copies | 345 | 308 | 19 | 26 (5.4%) | — | 8 searches / frame |
| (d) min_inliers 21 | 328 | 297 | 30 | 7 (1.5%) | 0.59% | none |

(c) is escalate_batch's rule (a copy matching as many peaks rejects the frame); its bound is 1/(k+1), so 1%
needs ~32 extra searches per accepted frame. (b) pays for the sparse-frame problem with dense frames. The floor
(a) costs nothing at run time. Its **√n shape** follows the null: the best of K orientations of a Binomial(n, p)
count sits near n·p + √(2 n p ln K), and the fitted linear coefficient 0.0224 is close to p = 0.3³ = 0.027. The
straight line (the 16 Sep per-event form, which was 0.057 n + 9.3 for a different engine and peak list) holds
the average but not the bands:

| null of the floor, by peak count | < 60 | 60–90 | 90–130 | 130–200 | 200–400 | ≥ 400 |
|---|---|---|---|---|---|---|
| frames | 90 | 97 | 70 | 101 | 99 | 23 |
| √n floor | 0.97% | 0.87% | 1.16% | 0.50% | 0 | 0 |
| straight line | 0.10% | 0.71% | 1.74% | 0.90% | 0 | 0 |

Out of sample, fitting on half the **frames** and scoring the other half (20 random splits × 2): null on the
held-out frames' copies median 0.66%, p90 0.89%, max 1.03%; on the held-out null file median 0.42%, max 1.67%.

`StreamDriver(null_floor=NULL_FLOOR_CXIDB17)`, with `NULL_FLOOR_CXIDB17 = (0.0224, 5.32, 1.211)` as (a, b, c).

## Are the frames it refuses crystals?

The published xgandalf arm (`xg_driver`, libxgandalf 0.12.0, blind, on the same q vectors) is independent
evidence: finding the same lattice within 2° of the driver's registration is not a chance event.

| frames | n | xgandalf LYSO solution | agrees < 2° |
|---|---|---|---|
| sparse (< 62 peaks), strict, floor **keeps** | 27 | 23 | **23** |
| sparse, strict, floor **refuses** | 22 | 7 | **0** (the 7 are 34–87° away) |
| sparse, live gate but not strict | 47 | 21 | 0 |
| live gate accepts, floor refuses (all n) | 112 | 55 | 2 |
| floor keeps (all n) | 335 | 312 | 290 |

The sparse control carries the argument: at the same peak counts xgandalf confirms nearly every frame the floor
keeps and none of the 22 strict frames it refuses, which have 38–60 peaks and 10–16 inliers against a null 99th
percentile of 14–17. The strict gate's own 25% bar is at chance there (29/480 null accepts overall).

## Transfer

| | live-gate null | floor null | real, live → floor |
|---|---|---|---|
| 120 Table 2 frames, lysozyme cell, 16 copies (different peak list) | 52% | 0.78% (0/120 held out) | 117 → 87; strict 77 of 80 kept |
| 480, relock cell 87.5/87.6/109.5 (8 copies) | 46% | 1.22% | 458 → 358 |
| 480, Proteinase K cell (8 copies) | 46% | 0.68% | 204 → 4 (wrong cell) |

Same detector, peak finder and run in every row, so this is not evidence the constants carry to another
detector, peak finder, search depth or HKL_TOL. Re-fit there with `live_gate_null.py`.

## End to end: the published arm plus the scrambled 480, CPU

`record_stream_replay.py --input lyso=q480_fix.txt --input scr=q480_scrambled.txt --B 20 --dmin 2.0
--warmup-rescue --adaptive-relock --min-inliers 10`, the arm of job 39344280, with and without
`--null-floor cxidb17` (CPU torch, the driver's numpy path):

| | scrambled accepted | lysozyme strict | lysozyme accepted | xgandalf-confirmed accepted | watchdog rescues | frames to the watchdog |
|---|---|---|---|---|---|---|
| live gate (CPU) | 234 (211 lyso, 23 cell1) | 331 | 453 | 292 | 9 | 277 |
| (job 39344280, A100) | 232 (210, 22) | 333 | 453 | — | 10 | — |
| live + floor (CPU) | **5** (2, 3) | **332** | 378 | **315** | 33 | 605 |

The strict count does not fall. The 22 strict frames lost are the unconfirmed ones above. The 23 gained all
arrive as watchdog rescues. Without the floor, all 23 were accepted by the batched pass at chance-level counts
(10–23 inliers). Of the 22 that xgandalf also solves, 21 had the **wrong orientation** (12–92° from xgandalf's)
and one was 2.1° off. With the floor that registration fails and the frame misses. The watchdog's blind search
then registers it with 16–53 inliers, within 0.35° of xgandalf (22/22 checkable).
So at the shipped gate, real crystals in wrong orientations reach the live merge, and the floor routes them to
the rescue that corrects them. This rescue needs `adaptive_relock` (or `retry_cascade`); without either, a
refused frame is dropped.

Cost: every miss goes to the watchdog's blind search (~26 ms on an A100), and the floor turns lattice-free
frames into misses: 605 frames reached the watchdog against 277 on this 960-frame stream. On a water-dominated
run with the watchdog on, that is the price. Both runs relock once, at frame 365, to the same spurious
87.5/87.6/109.5 cell as the published arm, so the floor does not stop that relock.

For comparison on the same q: the published xgandalf arm passes the strict gate on 350/480; its own chance rate
was not measured here. On the full 816-frame pf8 set, CrystFEL's indexamajig with xgandalf and the cell indexes
305/816 (`xg_full.stream`), and CrystFEL refinement accepts 746/816 of GLINT's solutions (`glint_full_ref.stream`,
`--indexing=file`).

## A100 (S3DF job 39358491, sdfampere034, exclusive, cupy 12.3 / torch 2.1, branch 9d4fa71)

`experiments/live_gate_gpu.sbatch`, the fused kernels, the same inputs (md5-checked).

- **On the node:** `test_live_gate_floor.py` 6/6 (its real-frame half on the GPU: live gate 50.6% of scrambled
  copies, floor 0.42%, 77 of 80 strict frames kept), plus the three pins' suites.
- **Default path unchanged:** the published 480 arm gated 333/480, 10 watchdog rescues, 1 relock (at 405).
- **The measurement on the fused path** (8 fit copies): per-frame counts are identical to the committed CPU ones
  on 463/480 real and 426/480 held-out null frames, and within ±2 on 475 and 460. The live gate accepts 210
  null frames. The committed floor accepts 2/480 of them and 0.47% of the fit copies. Refitting on the GPU counts
  gives 0.0262 n + 5.74 + 1.117 √n, which is within the fit's own spread.

| arm (A100) | scrambled accepted | lysozyme strict | lysozyme accepted | xgandalf-confirmed accepted | frames to the watchdog | relock at |
|---|---|---|---|---|---|---|
| published 480, default gate | – | 333 | 453 | 296 | 32 | 405 |
| published 480, `--null-floor cxidb17` | – | **334** | 379 | **315** | 127 | 365 |
| + scrambled 480, default gate | 232 (210 lyso, 22 cell1) | 333 | 453 | 296 | 280 | 405 |
| + scrambled 480, `--null-floor cxidb17` | **6** (2, 4) | **334** | 379 | **315** | 601 | 365 |

The default scrambled arm reproduces job 39344280 exactly (232 / 26 strict). With the floor, the 22 lysozyme
frames that gain strict all arrive as watchdog rescues. Of the 21 that xgandalf also solves, 19 were accepted by
the default gate in a wrong orientation, and all 21 come back within 2° of xgandalf. None of the 21 strict frames
lost is xgandalf-confirmed; 5 have an xgandalf lysozyme solution, all elsewhere. Four of the 6 scrambled frames
the floor admits go into the relock cell. That is the larger cell's higher null (1.2% above), so with two active
cells the stream's chance rate is about the sum of the two. The floor moves the spurious relock earlier (365 instead
of 405), because more misses reach the watchdog. It admits 15 lysozyme frames into that cell against 3.

## Caveats

- Fitted on the CPU eager engine (fp32). The A100 fused path (next section) gives the same null and the same
  end-to-end effect with these constants.
- One run, one protein family. The floor is opt-in and its constants are documented as dataset-specific.
- The floor is applied wherever the live gate is (`_gate_count`). The watchdog's candidate check and the
  per-frame cascade arm search harder than the batched pass, so their own null is higher than this floor was
  fitted to; the floor only tightens them.
- The strict research gate passes 6% of the scrambled frames here (29/480, 4.7% of the fit copies), all at
  sparse peak counts. Strict-gate counts on dense cxidb-17 frames include a chance share. This note measures
  that share and does not revise any published number.

## Reproduce

```
python experiments/live_gate_null.py measure --input ~/q480_fix.txt --k-fit 32 \
    --extra-cell cell1=87.5,87.6,109.5,69.3,72.7,100.7 --extra-cell prok=68.7,68.7,108.6,90,90,90 --k-extra 8 \
    --out live_gate_null_480.npz                       # CPU torch, ~25 min
python experiments/live_gate_null.py fit experiments/live_gate_null_480.npz --xgandalf ~/xgd480_fix.txt --input ~/q480_fix.txt
python experiments/live_gate_null.py confirm experiments/live_gate_null_480.npz --floor 0.0224,5.32,1.211 --xgandalf ~/xgd480_fix.txt
python experiments/live_gate_null.py measure --input experiments/frames_cxidb_clean.txt \
    --cell 79.02,79.02,37.98,90,90,90 --k-fit 16 --out live_gate_null_120.npz
python experiments/record_stream_replay.py --input lyso=~/q480_fix.txt --input scr=q480_scrambled.txt \
    --ref scr=79.02,79.02,37.98,90,90,90 --B 20 --dmin 2.0 --warmup-rescue --adaptive-relock --min-inliers 10 \
    [--null-floor cxidb17] --out replay.json
```
