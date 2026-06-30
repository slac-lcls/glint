# Severe-mosaic consensus recovery (frontier #2)

*Does cross-frame consensus recover the cell after the single-frame rate collapses?* D2 says severe
mosaic (σ≥.0015) destroys the long-axis phase (δ(q·v) ~ σ|v| on the 79 Å axis) so **generation**
fails — no single frame solves. This quantifies whether pooling many frames still recovers the cell.

Script: `severe_mosaic_consensus.py` (120 cxidb frames, N-best=3, `consensus` over R=40 random
K-frame subsets, `min_support=3`). **PICK** = the dominant support≥3 cluster is LYSO (what production
`consensus_cell` returns); **PRESENT** = *any* support≥3 cluster is LYSO (recoverable even if
out-voted); **reach** = % of frames whose top-N candidate set contains LYSO *at all*.

| σ (broadening) | per-frame top-1 | N-best reach | consensus → 100% at K | notes |
|---|---|---|---|---|
| .0008 (mild)   | 38% | 49% | K≈20 (90% by K=8)  | saturates fast |
| **.0015 (severe)** | **9%** | **11%** | **K≈80 (87% by K=40)** | recovers despite 91% single-frame failure |
| .0020 (severe) | 0% | **0%** | never | hard wall |
| .0030 (severe) | 0% | **0%** | never | hard wall |

Consensus(K) for σ=.0015: K=12→15%, 20→32%, 40→87%, 80→100%; LYSO cluster support grows 3→14 as
K 12→120 (the ~11% of frames that surface LYSO accumulate while spurious cells scatter below
support 3).

## The lesson: recoverability is gated by REACH, not by solve rate

- At σ=.0015 only **9%** of frames solve and only **11%** even surface LYSO in their top-3 — yet
  pooling ~40–80 frames recovers the cell at 87–100%. Consensus recovers what **no single frame can**,
  because the few frames that *do* surface the truth coincide (support climbs past the min-support
  floor) while spurious near-degenerate cells differ frame-to-frame and never reach support 3.
- At σ≥.0020 **reach = 0**: the long-axis phase is destroyed beyond candidate *generation*, so LYSO
  never appears even as a hypothesis → consensus has nothing to vote on → 0% at every K. **Generation
  wall** — the recoverability cliff sits sharply between σ=.0015 and σ=.0020.
- **PRESENT ≈ PICK** throughout (only a 3-pt gap at σ=.0015, K=40): when the truth is present it is
  essentially always the *dominant* cluster. The severe-mosaic failure mode is **absence, not
  out-voting**, so a cell-prior tiebreak buys nothing here; the only lever is raising **reach**
  (wide smooth window + QDIST, per D2) so the cell surfaces in more frames.

## Connection to the Fienup voting lesson ([[phase-retrieval-stagnation-lessons]])

This is the indexing instance of the boundary of Fienup's stripes-voting: voting recovers what
single-instance perturbation cannot — **but only while the true solution is present in *some*
instances** (reach > 0). When every instance stagnates without ever expressing the truth (reach = 0),
voting is powerless. For the PR/SAD ensemble: monitor the analog of *reach* (does any restart ever
express the correct enantiomorph/cell), not just the per-instance success rate — that is what predicts
whether consensus can rescue the run.
