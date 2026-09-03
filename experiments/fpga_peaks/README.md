# FPGA-grade peak emission vs GLINT indexing — the selection half

**Question.** If Bragg peaks reach the indexer from a readout FPGA or an in-pixel sparsifier
(ePixUHR / SparkPix-class) instead of GPU peakfinder8 — integer positions, a small per-tile
buffer, seams where a local finder cannot compute its background — does GLINT still index?

**Answer (measured):** yes, to within run noise of the undegraded rate. The cross-frame
consensus absorbs the per-frame losses an edge emitter introduces.

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

| arm | indexed | kept pk/frame |
|---|---|---|
| baseline (projection round-trip) | **117 / 120** | 137.9 |
| integer positions | 116 / 120 | 137.9 |
| seam mask w=8, panel grain (576×336) | 116 / 120 | 129.6 |
| seam mask w=8, ASIC grain (192×168) | 115 / 120 | 120.5 |
| seam mask w=12, ASIC grain | 111 / 120 | 107.9 |
| count cap 48/frame, unbiased | 115 / 120 | 47.6 |
| count cap 48/frame, low-\|q\| biased | 106 / 120 | 47.6 |

**Reading.**
- **Integer positions are free.** Sub-pixel precision is unnecessary for indexing (−1 frame,
  within run noise).
- **Consensus absorbs the seam loss.** Aggressive per-tile masking removes ~13% of peaks near
  tile edges (blind, per-frame), yet the consensus rate holds at 115/120 even at the fine ASIC
  grain; panel-grain tiling costs less. Tile coarsely and let consensus recover the rest.
- **Preserve peaks across resolutions.** A count cap costs little when unbiased (115) but
  measurably more when biased toward low resolution (106): indexing needs the high-angle peaks
  for angular leverage. These arms measure resolution diversity only; they do not validate
  intensity-ranked or FIFO/position-based dropping.

## What this does NOT cover (needs raw pixels — do not fake)

- **Intensity-weighted top-K per tile.** The committed lists carry positions only. The count-cap
  arms here are unbiased vs a low-\|q\| worst-case proxy; where a real intensity cap lands is
  unmeasured here.
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

Design brief for an edge peak-emitter, in one line: **emit integer positions; preserve peaks
across resolutions; tile coarsely; let the cross-frame consensus absorb the rest.**
