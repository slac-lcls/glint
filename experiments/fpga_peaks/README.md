# FPGA-grade peak emission vs GLINT indexing — the selection half

**Question.** If Bragg peaks reach the indexer from a readout FPGA or an in-pixel sparsifier
(ePixUHR / SparkPix-class) instead of GPU peakfinder8 — integer positions, a small per-tile
buffer, seams where a local finder cannot compute its background — does GLINT still index?

**Answer (measured):** integer emission and seam masking stay within 1–2 frames of the 117/120
baseline, while a 48-peak frame cap reaches 110/120 or 104/120 depending on selection bias.

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
| integer positions | 116 / 120 | 137.9 |
| seam mask w=8, panel grain (576×336) | 116 / 120 | 129.6 |
| seam mask w=8, ASIC grain (192×168) | 116 / 120 | 120.5 |
| seam mask w=12, ASIC grain | 115 / 120 | 107.9 |
| count cap 48/frame, unbiased | 110 / 120 | 47.6 |
| count cap 48/frame, low-\|q\| biased | 104 / 120 | 47.6 |

**Reading.**
- **Integer positions are free.** Sub-pixel precision is unnecessary for indexing (−1 frame,
  within run noise).
- **Consensus absorbs the seam loss.** Aggressive per-tile masking removes ~13% of peaks near
  tile edges (blind, per-frame), yet the consensus rate holds at 116/120 even at the fine ASIC
  grain; panel-grain tiling costs less. Tile coarsely and let consensus recover the rest.
- **Keep the strongest peaks across all resolutions.** A count cap costs little when unbiased
  (110) but measurably more when biased toward low resolution (104): indexing needs the
  high-angle peaks for angular leverage. This is the one selection policy that hurts — an
  occupancy/FIFO cap that drops by position or time (≈resolution-unbiased) is fine.

## What this does NOT cover (needs raw pixels — do not fake)

- **Intensity-weighted top-K per tile.** The committed lists carry positions only. The count-cap
  arms here are unbiased vs a low-\|q\| worst-case proxy; a real intensity cap keeps strong peaks
  at all resolutions and should land at or above the unbiased result.
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

Design brief for an edge peak-emitter, in one line: **emit integer positions; keep the strongest
peaks across all resolutions; tile coarsely; let the cross-frame consensus absorb the rest.**
