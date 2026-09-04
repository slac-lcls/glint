# FPGA-grade peak emission vs GLINT indexing — the selection half

**Question.** If Bragg peaks reach the indexer from a readout FPGA or an in-pixel sparsifier
(ePixUHR / SparkPix-class) instead of GPU peakfinder8 — integer positions, a small per-tile
buffer, seams where a local finder cannot compute its background — does GLINT still index?

**Answer (measured):** integer emission and seam masking stay within 1–2 frames of the 117/120
baseline; a per-tile buffer of 8 peaks is free (117/120), and even 2 peaks per tile — a quarter of
the available peaks — holds 115/120. The one policy that costs real frames is a resolution-biased
cap (103/120). *What* is dropped matters; *how many* barely does, over this range.

## What runs

`fpga_selection.py` reconstructs the *selection* half of the study entirely from data committed
to this repo — no raw pixels, no external staging:

1. Load the 120-frame cxidb-17 benchmark (`experiments/frames_cxidb_clean.txt`, the exact
   reciprocal-space vectors every indexer in `tab:summary` was fed).
2. Project each frame's q onto a synthetic detector panel (the geom bridge's own projection).
3. Impose a synthetic ePixUHR tile grid (ASIC 192×168; a readout-FPGA panel = 6 ASICs, 576×336).
4. Apply an FPGA emission constraint in pixel space, invert back to q, index with the shipped
   consensus path (`hybrid_index`).

A geometry self-check asserts the projection is an exact inverse (an already-projected q
round-trips to machine precision); the first-pass residual it prints is the physical excitation
error, not a bug.

## Result (A100, S3DF; glint @1226833)

Reference is the projection-round-trip baseline; each arm is a delta from it.

**The bar.** `indexed` counts frames whose picked cell is `same_lattice` with the reference — the
offline hybrid at the correct-lattice bar, with no reflection-count gate. That is the LOOSE bar; on
this set it coincides with the >=10-reflection bar and sits 4-6 frames above the >=25%-of-spots gate
the paper's Table 2 reports (FACTS `sf_lattice_of120` 85 against `sf_strict_of120` 79). These rows
are therefore comparable **with each other** — the point is the delta each emission constraint
costs — and not with a published indexing rate.

| arm | indexed | kept pk/frame |
|---|---|---|
| baseline (projection round-trip; offline, loose bar) | **117 / 120** | 137.9 |
| integer positions | 118 / 120 | 137.9 |
| seam mask w=8, panel grain (576×336) | 118 / 120 | 129.6 |
| seam mask w=8, ASIC grain (192×168) | 116 / 120 | 120.5 |
| seam mask w=12, ASIC grain | 117 / 120 | 107.9 |
| count cap 48/frame, unbiased (frame-global) | 115 / 120 | 47.6 |
| count cap 48/frame, low-\|q\| biased (frame-global) | 103 / 120 | 47.6 |
| **per-tile cap 2/tile, panel grain** | 115 / 120 | 35.0 |
| **per-tile cap 4/tile, panel grain** | 113 / 120 | 59.7 |
| **per-tile cap 8/tile, panel grain** | 117 / 120 | 91.5 |

Reproduced on two ampere nodes (`sdfampere023`, `sdfampere042`) and against two indexer
revisions; all four runs agree exactly, so the ±1–2 spread between neighbouring rows is the
scoring bar's own sensitivity, not run-to-run noise.

**Reading.**
- **Integer positions are free.** Sub-pixel precision is unnecessary for indexing (+1 frame,
  within the bar's sensitivity).
- **Consensus absorbs the seam loss.** Aggressive per-tile masking removes ~13% of peaks near
  tile edges (blind, per-frame), yet the consensus rate holds at 116/120 even at the fine ASIC
  grain; panel-grain tiling costs less. Tile coarsely and let consensus recover the rest.
- **A per-tile buffer is not worse than a frame-global cap of the same size — the locality of
  the loss does not matter here.** This is the arm the experiment's question actually needs, and
  it is not the same experiment as the frame-global cap: a real per-tile FIFO overflows *locally*,
  so the peaks it drops are spatially correlated, while a frame-global sample thins uniformly.
  Compare at matched budget: frame-global keeps 47.6 pk/frame for 115/120, and the per-tile arms
  bracket it — 35.0 pk/frame for the same 115/120, 59.7 for 113/120. Within the bar's ±1–2 the
  two policies are indistinguishable, and 8 peaks/tile (91.5 pk/frame) recovers the full baseline.
  **A per-tile FIFO of 8 costs nothing**, which is the number an edge emitter actually needs.
- **Resolution bias is the one policy that hurts.** At an identical 47.6 peaks kept, an unbiased
  cap gives 115/120 and a low-\|q\|-biased cap 103/120 — a 12-frame loss purchased entirely by
  *which* peaks were dropped, since the count is the same. Indexing needs the high-angle peaks
  for angular leverage. An occupancy/FIFO cap that drops by position or time
  (≈resolution-unbiased) is fine; one that preferentially keeps low-resolution peaks is not.

## What this does NOT cover (needs raw pixels — do not fake)

- **Intensity-weighted top-K per tile.** The committed lists carry positions only. The count-cap
  arms compare unbiased sampling with a low-\|q\| worst-case proxy; raw intensity data are required
  to place a real intensity-ranked cap relative to either result.
- **The detection model** (pf8 radial background vs a tile-local finder = *which* peaks are
  found) needs frames. A 2026-07 study on realistic simulated frames found a local-window finder
  matched pf8 and was robust to a rising water-ring background, but the experimental pixels
  (cxidb-17 raw, mfxl1038923) are not staged here. Re-staging them to close the detection half on
  real backgrounds is the open first task.

## Run it

```
# CPU (Mac, if torch present) or GPU:
GLINT_ROOT=$PWD python3 experiments/fpga_peaks/fpga_selection.py --json out.json
# S3DF A100 (torch env): experiments/fpga_peaks/run_fpga.sh under srun -p ampere.
```

Design brief for an edge peak-emitter, in one line: **emit integer positions; keep peaks spread
across all resolutions; a per-tile buffer of 8 is free; tile coarsely; let the cross-frame
consensus absorb the rest.**

("spread across all resolutions", not "the strongest": these lists carry positions only, so no arm
here ranks by intensity. What is measured is that *resolution bias* costs frames and *count* mostly
does not — an intensity-ranked cap remains untested, and could behave either way, since intensity
correlates with resolution.)
