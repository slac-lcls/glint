# glint_cli's written-crystal gate on a lattice-free null, and a chance floor for it

**Question.** `glint_cli --gate strict` (glint#216) writes a frame as a crystal when at least GATE_MIN = 10 of its
peaks and at least GATE_FRAC = 25% of them are matched (`matched_strict`, GATE_TOL 0.15). Its commit said the bar
is not null-calibrated. How often does a frame with *no lattice* clear it on the route the CLI runs, and what
per-peak-count floor holds that near 1%?

Code: `experiments/cli_gate_null.py` (measure / fit / confirm; the quantile fit, bands and xgandalf reader are
`live_gate_null.py`'s, glint#214). Per-frame data: `cli_gate_null_480.npz`, `cli_gate_null_120.npz` (arrays `n`,
`m_real`, `p_real`, `m_null`, `p_null`, `m_fit[k]`, `p_fit[k]`, `M_real`; path codes 0 none, 1 nbest, 2 rescue).
Gate: `glint_cli --gate floor --floor cxidb17|a,b[,c]`, `hybrid_stream.gate_results(..., "floor", floor=...)`.
Test: `experiments/test_cli_gate_floor.py`.

## What the gate judges

With `--cell`, glint_cli runs `hybrid_index(frames, Mc_known=cell, nbest=3)`; the gate sees one registration per
frame, from one of two searches:

1. **nbest**: `index_blind_nbest(q, 3)`, a blind search over all cells; the first hypothesis with
   `same_lattice(c, cell)` is the registration.
2. **rescue**: otherwise `index_known_gpu_cell(q, cell)` (shipped depth: topa 8, the default anchor count), accepted
   whenever `same_lattice(M, cell)`, which a known-cell search passes by construction.

This is neither StreamDriver's batched `index_fused` nor its inlier count, so `NULL_FLOOR_CXIDB17` (glint#214) is
not this route's floor. On both peak lists below, **every** registration of a scrambled frame came from the rescue
(0 of 14,201 and 0 of 3,509 from nbest); the real frames split 351 nbest / 115 rescue / 14 none on the 480 and
90 / 24 / 6 on the 120.

## Setup

- **Frames.** `q480_fix.txt` (S3DF `~/q480_fix.txt`, md5 `d6d86c1b…`): 480 cxidb-17 frames, peakfinder8 peaks,
  38–1010 peaks, median 117. Transfer: the committed `frames_cxidb_clean.txt` (120 frames, md5 `f89fa57d…`, the
  Table 2 peak list from the same run), 40–554 peaks, median 100.
- **Route.** `hybrid_index` itself, as glint_cli calls it with `--cell "79.02 79.02 37.98 90 90 90"` (= `LYSO`), no
  cascade, no escalation; the rescue is wrapped only to record which frames reach it. Frames are independent under
  a known cell, so they are sharded over processes; re-running the real frames as one call gives identical counts
  and paths (480/480, 120/120). CPU torch 2.13, OMP 1 thread. 480: 42,000 CPU-s; 120: 13,000 CPU-s.
- **Null.** `multilattice.scramble_azimuth` (each peak rotated about the beam by its own azimuth; |q| and q_z, so
  each peak's excitation error, kept; lattice removed). Held-out copy rng `[20260928, i]`; fit copies rng
  `[1, i, k]`, k < 32, used only to fit (15,360 copies on the 480, 3,840 on the 120).
- **This is the CLI's route.** `glint_cli --qframes frames_cxidb_clean.txt --cell … --device cpu` writes crystals
  for exactly the 114 frames measured as registered, and `--gate strict` for exactly the 89 measured as strict. Its
  default stream (no `--gate`) is byte-identical to `origin/main`'s (md5 `9771d79f…`), as is `--gate none`'s.

## The strict gate at chance

| | real registered | real strict | held-out null, strict | fit copies, strict |
|---|---|---|---|---|
| 480 | 466 | 354 | **29 / 480 (6.0%)** | 4.5% (688) |
| 120 | 114 | 89 | 5 / 120 (4.2%) | 4.8% (186) |

The chance passes are **sparse** frames: the 29 held-out passes have 38–60 peaks, and by peak count strict passes
22.8% of the fit copies below 60 peaks, 1.0% at 60–90 and none above 90 (no copy above 88 peaks). A frame of ~45
peaks needs 12 matched to pass, and below 60 peaks the rescue's best orientation on a lattice-free frame matches a
median 11, 99th percentile 15, maximum 18. The #216 docstring and `--help` called these "dense" frames; that is corrected here.

## Candidates, fitted at a 1% null on the 480 (each applied on top of strict)

| gate | real | strict kept | strict lost | held-out null | fit copies |
|---|---|---|---|---|---|
| strict (10, 0.25) | 354 | 354 | 0 | 29 (6.0%) | 4.48% |
| **floor 0.0246 n + 5.39 + 1.188 √n** | **344** | **344** | **10** | **0** | **0.26%** |
| floor, straight line 0.0683 n + 12.80 | 341 | 341 | 13 | 0 | 0.09% |
| fraction 0.290 instead of 0.25 | 316 | 316 | 38 | 2 (0.4%) | 0.81% |

The floor is the 99th percentile of the fit copies' matched counts as a·n + b + c·√n (linear quantile regression,
the method and shape of glint#214: the best of K orientations of a Binomial(n, p) count sits near
n·p + √(2 n p ln K)). Rounding the coefficients to (0.0246, 5.39, 1.188) changes no decision on either list.

| fit-copy null by peak count | < 60 | 60–90 | 90–130 | 130–200 | 200–400 | ≥ 400 |
|---|---|---|---|---|---|---|
| frames (480) | 90 | 97 | 70 | 101 | 99 | 23 |
| strict | 22.8% (656) | 1.03% (32) | 0 | 0 | 0 | 0 |
| √n floor | **1.01%** (29) | 0.35% (11) | 0 | 0 | 0 | 0 |
| straight line | 0.17% (5) | 0.29% (9) | 0 | 0 | 0 | 0 |

Strict's 25% is the binding bar from 68 peaks up; below that the floor binds, which is where chance lives, and the √n shape holds the
sparse band at the 1% target where the straight line over-rejects it (and loses 3 more real strict frames).

Out of sample, fitting on half the **frames** and scoring the other half's copies (20 random splits × 2): √n floor
median 0.27%, p90 0.43%, max 0.48% overall; below 60 peaks median 1.14%, max 2.16%; held-out file median 0, max
0.42%; strict frames kept, median 97.2%.

## Are the strict frames it refuses crystals?

The 10 strict frames the floor refuses have 38–55 peaks and 10–15 matched (floor there 13.6–15.6); 9 came from the
rescue, 1 from nbest. Independent evidence: the published xgandalf arm (`xgd480_fix.txt`, libxgandalf 0.12.0, blind,
same q) finding the lysozyme lattice within 2° of the CLI's registration.

| strict frames | n | xgandalf lysozyme solution | agrees < 2° |
|---|---|---|---|
| sparse (< 62 peaks), floor **keeps** | 45 | 39 | **38** |
| sparse, floor **refuses** | 10 | 3 | **0** (68–75° away) |
| all, floor keeps | 344 | 325 | 305 |

At the same peak counts xgandalf confirms 38 of the 39 kept frames it solves and none of the refused ones.

## Transfer to the 120 Table 2 list (same detector, run and cell; different peak list)

The 480-fitted floor on the 120's copies: 0.34% of fit copies (13 / 3,840), 1.14% below 60 peaks, 0.47% at 60–90,
0 / 120 held out; real 88 of 89 strict kept. The one lost (frame 68: 55 peaks, 15 matched, rescue) sits under its
own copies' maximum of 16. A floor fitted on the 120 alone is 0.0160 n + 4.68 + 1.354 √n: the same decision on every real frame and the same
number of copy accepts (13, not all the same copies).

## Not the driver's floor

StreamDriver's `NULL_FLOOR_CXIDB17 = (0.0224, 5.32, 1.211)` was fitted on the same 480 frames for the batched
`index_fused` engine and `StreamDriver._inliers`. It was not reused; applied to this route's counts it comes out
close (fit copies 0.29%, held-out 1 / 480, the same 10 strict frames lost), because both are a best-of-many-orientations
known-cell count on the same peaks. That is a finding about this dataset, not a reason to share constants.

## Scope and caveats

- **Dataset-specific.** One run, one cell, one peak finder, CPU only. The null moves with the cell (a larger cell
  has more nodes to hit by chance; the driver's relock cell had a higher null in glint#214), the peak finder, the
  detector and the search depth. That is why `--gate floor` takes a named calibration or coefficients and has no
  default: on other data, run `cli_gate_null.py measure` on the run's own frames and pass `--floor a,b,c`.
- **GPU not measured.** The blind N-best search runs fp32; on a GPU its counts can differ by a peak or so. The null
  registrations all come from the rescue, whose CPU/GPU agreement was not measured here.
- **Routes not measured.** Without `--cell` the per-frame searches are the same but run against the consensus cell;
  `--escalate` (topa 128, nc 32, its own 32-copy null) and `--cascade` add registrations from deeper searches, which
  the floor also gates but was not fitted to. Dense mode (`dense_index`) is a different search; there strict's 25%
  is far above any floor value.
- **Strict counts carry a chance share.** 29 / 480 lattice-free frames pass strict on this route; this note does not
  revise any published strict count.

## Reproduce

```
python experiments/cli_gate_null.py measure --input ~/q480_fix.txt --k-fit 32 --procs 11 --check-whole \
    --out experiments/cli_gate_null_480.npz            # CPU torch: ~70 min wall on 11 processes
python experiments/cli_gate_null.py measure --input experiments/frames_cxidb_clean.txt --k-fit 32 --procs 11 \
    --check-whole --out experiments/cli_gate_null_120.npz
python experiments/cli_gate_null.py fit experiments/cli_gate_null_480.npz
python experiments/cli_gate_null.py fit experiments/cli_gate_null_120.npz --floor 0.0246,5.39,1.188
python experiments/cli_gate_null.py confirm experiments/cli_gate_null_480.npz --floor 0.0246,5.39,1.188 \
    --xgandalf ~/xgd480_fix.txt
```
