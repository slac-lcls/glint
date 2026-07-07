# Results — dense cluster-FFT seeding research plan (companion to PLAN_coherent_seed_clustering.md)

**Run 2026-07-06, S3DF A100, n=150 (10 dense cells × 15 rotations), STEPS=8. Verdict: DOCUMENT_NEGATIVE.**
Anchors reproduced in every axis run — pooled `_cluster_fft_seeds` = **150/150 (100%)**, coherent
`_cluster_fft_seeds_coherent` = **146/150 (97.3%)** missing **tricl ×3 + trig_ob ×1** — so the harness /
`is_correct` (== `glint_cells.py:14`, `sort(norm(M,axis=0))` within 5%) is sound; no axis-bug recurrence.

## Headline
The coherent **97.3% wall is SELECTION-limited, not deposit / aperture / grouping-limited.** The
**100% ↔ speed tradeoff is fundamental**: 100% requires the K *separate* per-cluster peak-picks (each isolating
one cluster's local structure = the ~30 ms cost); any single summed volume is ~20 ms but caps at ~97%. No
deposit/window/grouping trick escapes that. The pooled default (100%) and the `COHERENT_SEEDS` option
(97%, ~1.5× M1, ~7× fewer seeds) stand as the two honest operating points. **Nothing new to wire.**

## Per-axis (real A100, n=150)
| axis | best config | rate | M1 ms | verdict | mechanism |
|---|---|---|---|---|---|
| **A2** sub-pixel deposit | coherent-gaussian (σ0.7 splat) | 147/150 | 32.7 | **NEUTRAL** | Gaussian splat de-quantizes the `round`-snap phase error (~`dq·|x|`) and **uniquely recovers all 3 tricl + trig_ob** — but newly breaks ortho_lg ×3 = lateral trade. Trilinear = 147 too (recovers only trig_ob, 5× slower). Pooled path gains nothing. |
| **F1** local apodization | none (coherent-ones baseline best) | 146/150 | 22 | **NEGATIVE** | No window (1/\|q_local\|, gaussian, hann) recovers tricl; all ≤ baseline and *add* misses (hex_P6, cubic_ins). Edge-tapering discards high-freq peak content the matched filter needs. The wall is **not** ripple-limited. |
| **C** position-aware grouping | C2-spread-G2 (Welch) | 142/150 | 20.9 | **NEGATIVE** | Spatial-adjacent (k-means, aperture synthesis) AND spatial-spread (round-robin, Welch) both < full-coherent 146; more groups strictly worse. Splitting fragments the √K matched-filter background cancellation. Position-awareness does not rescue the (already-negative) random grouping. |
| **E** coherence annealing | E1 pooled-detect→coherent-localize | **150/150** | **72.5** | **NEUTRAL** | E1 hits **100%** (recovers all incl tricl) with only **65 seeds** (10× fewer than pooled) — but at 2.4× pooled M1 (builds both volumes) → robustness, not speed; plain pooled already = 150/150@30 ms. E2 blend homotopy = NEGATIVE (144; the incoherent t=0 term buries weak vectors, poisons selection). |

## Corrections to the pre-run framing (both from real data)
- The **long-axis cells** (large_tet 140/140/150, prok 68/68/109, thaum 58/58/130) are **never** the coherent
  limiter — always 15/15. The single coherent miss is **tricl** (low-symmetry triclinic 45/55/65 80/85/95).
- tricl's miss is **not** a sidelobe ripple you can taper away (F1 made it worse). It IS sub-pixel-phase-sensitive
  (A2 gaussian fixes it) — but fixing it displaces ortho_lg, so it's a lateral move within the selection wall.

## The one genuinely interesting operating point
**E1 (pooled-detect → coherent-localize):** 100% at 65 seeds — the 10×-smaller seed set could cut downstream
M4 cost even though M1 is 2.4× pooled. Not a speed win, but a *seed-economy* knob if M4 ever dominates. Untested
downstream; parked.

## Not run
The final **combined SYNTH** verification (coherent+gaussian, gaussian∪pooled) was authored but not executed —
the phase-2 synthesis agent self-restricted on stale plan-mode context even though its 4 siblings ran. It is
**confirmatory-only** (can't change the verdict: any 100% needs the pooled searches = the speed cost).

## Scripts / provenance
S3DF `/sdf/home/s/smarches/`: `A2soft.py`, `F1apod.py`, `Cgroup.py`, `Eanneal.py` (+ `_run.sh` wrappers, `.out`);
jobs 30837576/30837496/30837698/etc. Harness `dense_seed_group.py`. Workflow run `wf_a3d8ce56-1f5`. Memory:
`glint-dense-coherent-seeds`. Earlier verified study (pooled 100 / coherent 97 / grouping negative) in the same memory.

## Workflow meta-lesson
Background subagents doing remote (S3DF) execution will **self-restrict citing plan mode** if that context leaks,
EVEN with an explicit "plan mode is OFF" prompt directive (phase-1 agents ran; the phase-2 agent still refused).
**Clear the actual plan-mode flag (ExitPlanMode) before launching an exec workflow** — don't rely on a prompt override.
