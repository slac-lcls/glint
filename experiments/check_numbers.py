#!/usr/bin/env python3
"""Guard the GLINT deliverables against number drift.

Every problem the 2026-07-19 audit found was drift BETWEEN deliverables that each looked fine on its
own: `34 ms` was corrected by hand in the paper and the derived quantities (frames/s, speedup ratios)
were left stranded at the old value in four other files. This turns that class of error into a failing
script.

It checks four things, in rough order of how much damage each does:

  1. RETIRED values      -- a number we have superseded, appearing anywhere. The 15 ms / 21 ms /
                            47 f/s / 160x family. Each carries what it should say instead. Also the
                            71%-attributed-to-GLINT swap: 71% is XGANDALF's blind rate, GLINT-(1) is
                            76%, and conflating them once cost two decks a wrong headline.
  2. OVERCLAIMS          -- language asserting end-to-end real-time / live merge, which the measured
                            179 frames/s (vs ~3500 hits/s needed) does not support.
  3. AMBIGUITY           -- `0.33 ms` means fp32 INDEXING at B=32 *and* fused INTEGRATION per frame.
                            A bare one is a defect; it must sit near a word that says which.
  4. ARITHMETIC          -- the facts table self-checks: frames/s must equal 1000/ms, speedups must
                            equal before/after, the DRP tier sizing must equal its own formula. This
                            is what catches a half-applied edit, because you cannot change `34 ms`
                            without `29 f/s` going red.

Usage
    python check_numbers.py                 # check sources (LaTeX + deck builders)
    python check_numbers.py --pdf           # also check the BUILT PDFs (catches a stale export)
    python check_numbers.py --facts         # print the authoritative table and exit
    python check_numbers.py path [path ...] # check specific files instead of the defaults

Exit status is 1 if anything fails, so it drops straight into a pre-push hook or CI.

When a measurement legitimately changes: edit FACTS, run with --facts, and fix every file the run
flags. Do not silence a rule to make a deliverable pass.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import unicodedata
from math import comb
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path.home()

# --------------------------------------------------------------------------------------- the facts
# Measured, and the single source of truth. Provenance in the trailing comment: PR number where
# merged, or "open" where it is not yet.
FACTS: dict[str, float | str] = {
    # indexing -------------------------------------------------------------------------------------
    "blind_ms":            25.7,    # blind pipeline GLINT-1 (hybrid_index), per frame, one A100; re-measured 2026-07-23 HEAD be55302, 3 reps; paper displays ~26
    "blind_fps":           39.0,    # = 1000/blind_ms
    "known_perframe_ms":   16.5,    # per-frame known-cell rescue (replica_gpu.index_known_gpu_cell)
    "graph_ms":            1.46,    # batched + CUDA graph                                    (#14)
    "fused_b32_ms":        0.45,    # fused kernels, fp64, batch 32                           (#16)
    "fused_b120_ms":       0.26,    # fused kernels, fp64, batch 120                          (#16)
    "fused_b32_fp32_ms":   0.33,    # fp32 INDEXING at batch 32                               (#15/#16)
    "fused_b120_fp32_ms":  0.16,    # fp32 INDEXING at batch 120                              (#15/#16)
    "fused_fps":           3800.0,  # = 1000/fused_b120_ms
    "saturating_batch":    64,      # one block per frame; 108 SMs on an A100
    "indexing_rate":       "75/114",
    # Percentages are ROUNDED, not floored (changed 2026-08-02). tab:summary previously mixed the two:
    # DIALS printed 27% for 32/120 = 26.67 (rounded) while GLINT-(1) printed 76% for 92/120 = 76.67
    # (floored), so three "correct" values for one measurement were in circulation. Counts are now
    # shown inline in the table, which makes the convention checkable instead of inferred.
    # The COUNTS are the measurement; the percentages are DERIVED from them and checked below. Until
    # 2026-08-02 the counts lived only in these trailing comments and both percentage keys were read by
    # nothing, which is precisely how the pair drifted: 76/71 stayed written into the rule text below
    # while the measurement moved to 77/72, and every run stayed green because no code connected them.
    "glint1_strict_of120":         92,  # GLINT-(1), blind + cross-frame consensus, >=25%-of-spots bar
    "glint_blind_rate_pct":        77,  # = round(100 * glint1_strict_of120 / 120)
    "xgandalf_blind_strict_of120": 86,  # xgandalf blind, SAME bar, SAME peak list -- a different indexer
    "xgandalf_blind_rate_pct":     72,  # = round(100 * xgandalf_blind_strict_of120 / 120)
    # The SAME two blind arms extended to 480 frames of the same run (2026-08-17). Identical peak
    # finder (pf8 out of the CrystFEL stream), identical gate, and the published 120 embedded as a
    # subset that reproduces 92 and 86 EXACTLY -- that reproduction is the control that makes these
    # rows quotable. The 8:2 discordant split at n=120 did NOT persist: over the 360 added frames it
    # runs 21:31, so 4x the frames CONFIRM the tie instead of resolving it. Quote the n you mean.
    # GLINT-(1) IS `hybrid_index(Mc_known=None)` -- N-best blind, consensus over the POOLED N-best,
    # best N-best cell consistent with it, then cell-general rescue. NOT compare3.py's blind-top-1 +
    # consensus + rescue. The two are indistinguishable on the 120 subset (both 92) and 15 frames
    # apart at n=480 (361 vs 346), so the choice was invisible until the set grew. Settled 2026-08-18.
    "glint1_strict_of480":         361,  # GLINT-(1) = hybrid_index(None), >=25%-of-spots bar, n=480
    "glint_blind_rate_pct_480":     75,  # = round(100 * glint1_strict_of480 / 480)
    "xgandalf_blind_strict_of480": 350,  # xgandalf blind, same bar, same peak list, n=480
    "xgandalf_blind_rate_pct_480":  73,  # = round(100 * xgandalf_blind_strict_of480 / 480)
    "mcnemar480_glint_only":        34,  # discordant frames GLINT-(1) indexes and xgandalf does not
    "mcnemar480_xgandalf_only":     23,  # and the other way -- exact two-sided McNemar p = 0.18
    # sec:consensus's "consensus recovers everything knowing the cell is worth" is BAR-DEPENDENT and
    # was stated at the bar where it fails. Same pipeline, cell derived vs handed.
    "truecell_strict_of480":       357,  # hybrid_index(LYSO), strict bar -- BELOW consensus's 361
    "cons_vs_true_strict_cons":     16,  # discordant, consensus-only        (p = 0.57, a tie)
    "cons_vs_true_strict_true":     12,
    "cons_10refl_of480":           458,  # at the >=10-reflection bar the handed cell PULLS AHEAD
    "truecell_10refl_of480":       468,
    # sec:streaming at n=480 (2026-08-17). The published 120 reproduces EXACTLY as a control
    # (91 / 73 / 78 / 88), so these are the same arms on 4x the frames, not a re-definition.
    "stream_rate_of480":       323,  # StreamDriver baseline, strict gate
    "stream_rate_rescue_of480": 331,  # + warmup_rescue + adaptive_relock
    "offline_rate_of480":      357,  # offline hybrid_index(Mc_known=LYSO), same gate
    # THE inversion: the blind retry stops being a patch that closes a gap and becomes a net WIN.
    "retry_rate_of480":        365,  # streaming + blind retry on gate-failing frames -- PAST offline
    "retry_rate_of120":         88,  # the same retry on the 120 subset, still SHORT of offline's 91
    "stream_fail_of480":       152,  # frames streaming fails at the strict gate
    "stream_fail_nobody480":    65,  # ...that NO indexer tested recovers (43%)
    "union_all_indexers_480":  407,  # union of GLINT blind/(1), xgandalf blind+known, ffbidx, offline
    # sec:streaming's "ingests essentially every frame" clause, MEASURED off the driver's own
    # post-lock accept counter. It used to read "114 of 115", which was the misreading `indexing_rate`
    # is annotated against below: 114 is the LOOSE half of the (strict, loose) pair 75/114 over 120
    # PUSHED frames, not a numerator over the 115 post-lock ones. The real value there is 111.
    "driver_accept_of115":     111,  # post-lock frames the driver accepts at its count-only gate, n=120
    "driver_accept_of475":     442,  # ...and at n=480 (5 warm-up frames in both, so 115 and 475)
    # The sequential-stop trial, re-run at n=480 over 400 random arrival orders. Blind N-best is
    # deterministic per frame, so the candidates are cached once and replayed shuffled -- the trials
    # differ ONLY in order, which is what the claim is about. The 120 reproduces (median 6, 0/400).
    "seqstop_median_lock":       6,   # frames to reach the batch consensus cell, median, n=480
    "seqstop_p90_lock":         12,   # 90th percentile (was 10 at n=120 -- longer pool, longer tail)
    "seqstop_false_locks":       0,   # of 400 orderings, vs the batch consensus cell AND the textbook
    "seqstop_trials":          400,
    # The n_pool-keyed streaming gate (pool_switch). Two 400-order replays of RunningConsensus over
    # CACHED per-frame N-best candidates, so the arms differ only in the acceptance rule:
    #   clean = experiments/nbest_120.npz (cxidb-120, truth LYSO)
    #   bad   = mfxx49820 r0016, 2228 frames on UNREFINED psana geometry (the q-set alias_test.py
    #           cached with no geom= -- psana's own), N-best dumped on S3DF by cand_r0016.py to
    #           ~/glint_gt_mfxx49820/cand_r0016_fixmap.npz; 2098 frames indexed -> 6294 hypotheses
    # The bad run is the item-7 failure: its POOLED vote returns the true lattice under glint#102
    # densest seeding (27/6294) and a DOUBLED cell under arrival order (23/6294, vol/truth 2.06,
    # reproducible today only with GLINT_CONSENSUS_STABLE=0). RunningConsensus groups in arrival
    # order, so the streaming lock still meets the second one.
    "poolgate_clean_trials":       400,
    "poolgate_clean_lock_med":       6,  # frames to lock, gap-only AND switch=72 -- unchanged
    "poolgate_clean_lock_max":      18,  # worst order, gap-only and switch=72
    "poolgate_clean_lock_max_onb":  42,  # ...and with frac/lead applied at EVERY pool size
    "poolgate_clean_maxpool":       54,  # largest n_pool at lock on clean data -> why 72 clears it
    "poolgate_clean_diff_72":        0,  # of 400 orders differing from gap-only (48 differs on 5)
    "poolgate_bad_trials":         400,
    "poolgate_bad_locks_gaponly":  342,  # orders where the SHIPPED gap-only rule locks
    "poolgate_bad_true_gaponly":   148,  # ...of which on the true lattice
    "poolgate_bad_wrong_gaponly":  194,  # = 342 - 148; median vol/truth 1.94, i.e. the doubled cell
    "poolgate_bad_minpool":         54,  # earliest lock -- the tail OVERLAPS clean's max of 54
    # The gated arms. switch=72 and frac/lead-always-on land on the SAME three locks here, so the
    # switch costs nothing in protection while costing nothing on clean data either (diff_72 = 0).
    "poolgate_bad_locks_gated":      3,  # orders that still lock, with frac .02 / lead 1.5
    "poolgate_bad_true_gated":       1,  # ...of which on the true lattice
    "poolgate_bad_wrong_gated":      2,  # the residual: locks at n_pool 54-96, where a 2% share
                                         # test is trivially satisfied (2% of 54 = 1.08 < 3), so
                                         # neither scale-free test protects the small-pool tail
    "poolgate_bad_wrong_removed":  192,  # = 194 - 2
    # M3 ascent steps, re-measured at n=480 (experiments/steps_sweep.py, 13 arms 2->80, strict bar,
    # exact McNemar vs the shipped default). The point of the block is that the knob is INERT above
    # 4, so the numbers worth pinning are the plateau's ends and the one arm that is not on it.
    "m3_steps_default":            8,   # shipped; a throughput choice, not an accuracy one
    "m3_blind_steps8_of480":     282,
    "m3_blind_steps4_of480":     284,   # p = 0.894 vs 8 -- saturation is at 4, not 8
    "m3_blind_steps80_of480":    293,   # p = 0.169 vs 8 -- 10x the work buys nothing measurable
    "m3_blind_steps2_of480":     257,   # p = 0.008 vs 8 -- the ONLY significant arm, and it is worse
    "m3_hybrid_steps8_of480":    361,
    "m3_hybrid_steps2_of480":    342,   # p = 0.003 vs 8
    "m3_hybrid_steps32_of480":   366,   # the max, p = 0.458 vs 8 -- i.e. not a better setting
    "m3_ms_steps8":              6.1,   # ms/frame blind, A100
    "m3_ms_steps80":             9.8,   # = 1.62x for a rate that does not move
    # Batched vs per-frame known-cell, at the gate. The paper used to call these "rate-identical in
    # aggregate" on the strength of the n=120 split being EXACTLY 9-9. That symmetry is the sample,
    # not the algorithm: at n=480 it is 23-29. The totals agree only to within the discordant noise.
    "bvp_disagree_of120":       18,  # frames where batched and per-frame gate differently
    "bvp_batched_only_120":      9,
    "bvp_perframe_only_120":     9,
    "bvp_disagree_of480":       52,
    "bvp_batched_only_480":     23,
    "bvp_perframe_only_480":    29,
    "ffbidx_known_strict_of480":   373,  # ffbidx known-cell, n=480
    "xgandalf_known_strict_of480": 397,  # xgandalf known-cell, n=480
    # integration ----------------------------------------------------------------------------------
    "integ_before_ms":     585.0,   # 16 Mpix / 800 reflections, whole-frame float64 upcast
    "integ_after_ms":      7.6,     # upcast removed, bit-identical                           (#17)
    "integ_speedup":       76.8,    # = integ_before_ms / integ_after_ms
    "integ_fused_ms":      0.33,    # fused GPU box-integration, per frame                    (#18)
    "h2d_16mpix_ms":       4.0,     # ~3-5 ms; why the fused kernel only pays device-resident
    # streaming (NOT merged) -----------------------------------------------------------------------
    # Re-measured 2026-07-21 after the fused peakfind reduction, as ONE coherent set: both arms in the
    # same job on the same A100, interleaved, steady state (driver built OUTSIDE the timer), 5 warmups
    # + 40 reps per stage, over the 40-frame 1024^2 sim.
    #
    # WARMUP MATTERS MORE THAN EXPECTED. An earlier pass warmed each stage ONCE and inflated the whole
    # stage block -- integrate read 0.41 instead of 0.31 (-22%), peakfind 1.58 instead of 1.16. Two
    # internal checks say this warmed set is the trustworthy one: the stages the change cannot affect
    # measure IDENTICALLY across both arms (index 0.535/0.535, integrate 0.312/0.313, peaks_to_q
    # 0.082/0.082), and the end-to-end delta (1.066 ms) matches the peakfind delta (1.087 ms) to 2%,
    # where the cold numbers mismatched by 14%.
    #
    # Same-protocol PRE-change arm: stream 5.49, peakfind 2.24, everything else unchanged. So the
    # honest figures are stream 5.49 -> 4.42 (1.24x) and peakfind 2.24 -> 1.16 (1.93x). Do NOT compare
    # against the #19 values (5.58 / 2.38 / 0.89), which came from a different protocol.
    #
    # PREDICT RE-MEASURED 2026-08-01 (A100-SXM4-40GB, ana-4.0.59-py3-minipytorch -- the only env on
    # S3DF carrying both torch and cupy, so it is also the env the block above was measured in). Two
    # arms in ONE allocation, 2 rounds each, alternating: `pre` = 5f72fd0 (main just before #68),
    # `post` = 7858457 (main with #68). Same protocol as the block above -- driver built OUTSIDE the
    # timer, driven to the locked post-lock steady state, then 5 warmups + 40 reps per stage.
    #
    # The harness is CALIBRATED against this table, not just self-consistent: four stages #68 cannot
    # touch reproduce their entries here -- peakfind 1.19 (vs 1.16), integrate 0.33 (0.31), h2d 0.184
    # (0.19), peaks_to_q 0.083 (0.08), all within 2-7% and identical across both arms. So a predict
    # number from this harness is like-for-like with the rest of the block.
    #
    #     predict    pre-#68  0.420 / 0.422 min   ->   post-#68  0.157 / 0.156 min   (2.7x)
    #
    # CORRECTION 2026-08-07, to a claim this comment used to make. It said 1.11 "was wrong twice over"
    # and "does not reproduce even on the pre-#68 arm, which measures 0.42", blaming tol=0.006 plus a
    # bench that rebuilt `panels` inside the timed call. That was a BAD INFERENCE FROM A BADLY CHOSEN
    # BASELINE. The arm used as "pre-#68" was 5f72fd0, which already contained #50 -- so it was never
    # a test of 1.11 at all. Re-measured AT ITS OWN COMMIT (0623e34, A100, driver outside the timer,
    # tol=0.002, panels built once) predict reads 1.132 / 1.126 ms, i.e. 1.11 is REPRODUCIBLE and was
    # an honest measurement of the code on 2026-07-21.
    #
    # The real history is three honest numbers, each superseded by a merged optimisation:
    #
    #     predict   1.11  (<=0623e34)  --#50-->  0.42  --#68-->  0.16
    #     wall      4.42  (<=0623e34)  --#50-->  ~3.7  --#68-->  3.64
    #
    # #50 (96b0dda, "collapse + fuse predict's gate, -17% wall") landed the DAY AFTER 4.42/1.11 were
    # recorded and its own message reports the wall at 3.91 -> 3.69 -> 3.23. Nothing carried that into
    # this table. So the lesson is not "someone benchmarked badly" -- it is that a stage table decays
    # silently when the optimisation PRs that move it do not update it.
    #
    # An independent in-situ cross-check (whole-driver instrumented run, 800 frames, same two arms)
    # agrees on the DELTA even though its absolute scale runs ~8% high because it times stages inside
    # the live pipeline rather than in a loop: predict 0.483 -> 0.207, and the end-to-end wall moved
    # 4.12 -> 3.84, i.e. the wall dropped by 0.28 against a predict saving of 0.28.
    "stream_ms":           3.64,    # steady state per frame, B=40   (#50+#68; was 4.42, then 4.16)
    "stream_fps":          275.0,   # = 1000/stream_ms               (#50+#68; was 226, then 240)
    "peakfind_ms":         1.16,    # LARGEST single stage, 7.3x predict          (#41, open; was 2.38)
    "predict_ms":          0.16,    # fused gate+projection kernel      (#68, merged; was 1.11, 0.42)
    "index_b40_ms":        0.54,    # fused index, B=40 (NOT the 0.26 B=120 amortization)
    "integrate_ms":        0.31,    # fused box-integration, per frame
    "peaks_to_q_ms":       0.08,    # geometry, on the host
    "fused_share_ms":      0.85,    # = index_b40_ms + integrate_ms               (re-measured; was 0.89)
    # The host side, measured explicitly rather than left as one "overhead" bucket -- a bucket would
    # have double-counted `accumulate`, which the DRP table already carries as its own row. These
    # components close to stream_ms exactly, which the arithmetic check below enforces.
    "accumulate_ms":       0.51,    # running merge accumulation, on the host
    "h2d_ms":              0.19,    # 2.1 MB upload into the resident ring slot (1024^2 uint16)
    "misc_ms":             0.52,    # peaks D2H (0.02) + python loop and glue
    # Host total = accumulate + h2d + misc = 1.22 (29% of the frame). INVESTIGATED 2026-07-22 and it
    # is NOT a throughput lever: batching the host glue (one readback + one peaks_to_q at flush) is
    # bit-identical but recovers only ~1%. The earlier "~2.3 ms un-attributed machinery" was a
    # benchmark artifact -- attribute_gap.py timed the one-time driver build inside its per-frame loop.
    "host_total_ms":       1.22,
    # THE GAP THE CORRECTED predict_ms OPENED, carried as its own line rather than folded into
    # misc_ms. With predict at its true 0.16 the measured stages sum to 3.47 against a stream_ms of
    # 4.16, so 0.69 ms/frame is un-attributed. That gap is not new work appearing -- it was always
    # there, hidden inside the 0.95 ms of phantom cost the old predict_ms=1.11 was carrying. Naming
    # it keeps `misc_ms` a MEASUREMENT (0.52) instead of quietly turning it into a plug, which is the
    # exact failure mode this file exists to catch.
    #
    # It is an OPEN ITEM, and the arithmetic check warns while it stays this large. The in-situ
    # cross-check run points at stream_ms rather than at a missing stage: instrumenting the whole
    # driver accounts for 99.7% of its own wall with only ~0.2 ms of glue, but measures that wall at
    # 3.66 ms/frame post-#68 -- i.e. ~0.5 ms below stream_ms on the same code, most likely because
    # stream_ms is measured over a single pass of the 40-frame stack (lock-in frames included) while
    # the in-situ run tiles it to 800 frames in the locked steady state. Re-measure stream_ms itself
    # before quoting this decomposition stage-by-stage.
    # RE-MEASURED 2026-08-04, and the suspicion above is CONFIRMED: the gap is in stream_ms, not in a
    # missing stage. Harness: experiments/remeasure_stream_ms.py (A100 sdfampere032/035, main
    # @ f82d0a7, same glint_sim 40x1024 stack, B=40, tol=0.002).
    #
    #     COLD, single pass of 40     13.665 ms/frame   <- setup dominates; NOT a steady state
    #     COLD, tiled x20 = 800        3.579
    #     WARMED, single pass of 40    3.496
    #     WARMED, tiled x20 = 800      3.494            <- steady state (a 2nd run gave 3.662)
    #
    # So the warmed steady-state wall is 3.49-3.66, i.e. 12-16% BELOW the 4.16 recorded here. Against
    # a 3.5-3.66 wall the stage table's 3.47 leaves 0.03-0.19 ms (1-5%) un-attributed instead of 0.69
    # (17%) -- the warning below would not fire. Cold-vs-warm also shows why this is easy to get
    # wrong: one-time cost (graph capture, JIT, grid build) is 3.9x the steady state when amortised
    # over a single 40-frame pass, so any re-measure MUST warm up first.
    #
    # The min-vs-mean protocol difference was ALSO checked and is NOT the explanation: timing the
    # same stages as means inside the live loop rather than as tmin minima costs only 1.18x overall
    # (peakfind 1.05, integrate 1.04, predict 1.35, index 1.45).
    #
    # ACTED ON 2026-08-07, after the two conditions above were met. Both were:
    #
    # PROVENANCE, which turned out to be recoverable after all -- not from the code, from the commit
    # that recorded it. 0623e34's message states the protocol outright: "both arms in the same job on
    # the same A100, interleaved, steady state, min of 15 passes over the 40-frame 1024^2 sim".
    #
    # HARDWARE, confirmed by re-running the SAME harness on both commits in one A100 allocation
    # (A100-SXM4-40GB, 2 rounds, alternating):
    #
    #     0623e34  wall 4.725 / 4.765   predict 1.132 / 1.126     <- reproduces 4.42 / 1.11
    #     21ca4db  wall 3.657 / 3.618   predict 0.155 / 0.157     <- current main
    #
    # The old arm reproducing its own recorded values is what licenses the swap: the hardware is the
    # same, the harness is the same, and the 1.30x difference between the arms is real code. (4.72
    # against a recorded 4.42 is the protocol difference and points the right way -- the original took
    # a MIN of 15 passes, this is a MEAN over 800 frames, and a min is expected to sit below a mean.)
    #
    # So the 0.69 was never un-attributed work. It was #50's ~0.75 ms, measured and merged on
    # 2026-07-22 and never carried into this table. With the wall at 3.64 the stage sum of 3.47
    # leaves 0.17 (5%), which is ordinary protocol slack and no longer trips the warning.
    #
    # These numbers move in GLINT's FAVOUR (stream_fps 240->275, peakfind share 28%->32%, the live gap
    # 15x->13x, the FPGA ceiling 1.39x->1.47x), which is the case for being slow, not for being wrong:
    # every one of them follows from optimisations that were separately measured and merged. What
    # would NOT have been legitimate is re-interpreting the same measurement more kindly.
    "unattributed_ms":     0.17,    # = stream_ms - sum(measured stages)   (protocol slack, 5%)
    # streaming vs offline YIELD -- success fraction, NOT throughput -------------------------------
    # The project's only real-data streaming-vs-offline head-to-head, promoted out of f61a4cf's commit
    # body where it was the sole record. Same 120-frame real cxidb set; the completeness test is the
    # paper's strict research bar (>=25% of spots AND >=10 refl).
    #
    # THE LATTICE TEST IS NOT THE SAME ON BOTH SIDES, and this comment used to claim it was.
    # gap_on_real.py gates the streaming arms with strict_gate(M, q, drv.Mc) (:49, :54) -- against the
    # cell THE DRIVER ITSELF LOCKED -- and the offline arm with strict_gate(M, q, LYSO) (:84), against
    # the reference cell. The frac and >=10-refl parts are identical; only the same_lattice reference
    # differs (what_are_the_failures.py:36-43). So streaming is scored on "consistent with the cell we
    # locked" and offline on "correct". Those coincide only while the lock is right -- it is here, the
    # locked cell matching the batch consensus cell to <0.01 A on every edge -- but a wrong lock would
    # let its own frames pass. The bias runs in STREAMING's favour, so the measured gap is a FLOOR:
    # under drift the true gap can only widen.
    #
    # Denominator is 120 pushed frames in ALL THREE, and the names say so on purpose: `indexing_rate`
    # above is a (strict, loose) PAIR over 120, not a ratio, and reading it as 75/114=66% is the exact
    # misreading these names exist to prevent.
    #
    # INDEX-ONLY. The q-vector dataset cannot exercise the integrate path (test_inlier_frac_gate.py),
    # so these are indexed counts, not integrated-and-merged ones.
    #
    # WHAT CAUSES THE GAP -- measured, and it is NOT the early lock. f61a4cf guessed streaming was
    # losing frames because it commits its cell from ~5-6 warm-up frames while offline votes across
    # all 120, and this comment asserted that guess as fact until #73 measured it: consensus, the lock
    # and cell precision were each checked and cleared. The gap is ONE missing step. Offline runs
    # blind + N-best on every frame before falling back to known-cell; streaming, once locked, runs
    # known-cell ONLY. A blind retry on just the gate-failing frames recovers 10 of the 13-frame gap
    # on real data (78 -> 88 of 120, 77% closed), and the full retry cascade reaches 94 (glint#75).
    # The retry is NOT SHIPPED, so 78 is the shipped number and 88/94 are headroom, not facts.
    #
    # warmup_rescue is FIXED-cost, worth 5/N: 4 points here, 1.2 at 400 frames, negligible at DAQ
    # rates, while the retry gap grows with N. DIALS-60 shows no gap at all (60/60 from warmup_rescue
    # alone), so rich frames do not exercise this failure mode -- do not benchmark it on easy data.
    "stream_rate_of120":     73,    # StreamDriver baseline. Corroborated at HEAD: the min_inlier_frac
                                    # table in stream_driver.py (added by 6bfc6a9, after the gate
                                    # changes) re-measures 73 clearing the bar, 0 good frames lost at
                                    # the shipped 0.15 default                        (f61a4cf, open)
    "stream_rate_rescue_of120": 78, # + warmup_rescue=True.
                                    #
                                    # THE GATE-CHANGE CAVEAT IS DISCHARGED (re-run 2026-08-03, drp-gpu007
                                    # H200 NVL, main @721d5cc, test_streamdriver_vs_offline.py). The worry
                                    # was that 907c057 (fit-gate made fractional) and ae5f53b (ingest gate
                                    # on the single-cell path) had moved acceptance under this arm. They
                                    # did not: reverting ONLY glint/ to f61a4cf on the SAME box -- verified
                                    # by min_inlier_frac count 0 (f61a4cf) vs 7 (HEAD) -- returns the
                                    # IDENTICAL 72 baseline / 77 rescue. The gate changes cost zero frames.
                                    #
                                    # RESOLVED on an A100 (Perlmutter nid001009, A100-SXM4-40GB, same
                                    # commit 721d5cc, `gap_on_real.py`): 91 / 73 / 78 / 78 / 88, gap 13,
                                    # closed 77% -- an EXACT match to these FACTS and to the paper's
                                    # "61--65% (73--78 of 120)" and "closes 77%". It reproduced on a
                                    # completely DIFFERENT software stack (cupy 14.1.1, numpy 2.1.2, torch
                                    # 2.8.0) from the S3DF ana-4.0.58 env where the original was taken, so
                                    # the stack is ruled out too. 73/78 is CORRECT. Do not "fix" it.
                                    #
                                    # The H200's 72/77 is a real ARCHITECTURE sensitivity, and it is ONE
                                    # FRAME: #33. Recovered sets are identical except that H200 adds 33
                                    # (43 gate failures vs A100's 42) -- i.e. on A100 frame 33 clears the
                                    # streaming gate directly, on H200 it lands just under and the blind
                                    # retry takes it. Both architectures converge to 88 after retry, so
                                    # only the PRE-retry arms move. Expect 72/77 when re-running on
                                    # Hopper/Blackwell; that is not a regression.
                                    #
                                    # CORROBORATED UNCHANGED in the same run: offline 91 (exactly), and the
                                    # blind-retry arm 88 -- so the paper's 76% and 73% do not move; only the
                                    # live figure is in question.                (f61a4cf; H200 A/B, open)
    # ⚠ TWO DIFFERENT PIPELINES, and they are one frame apart at n=120 and eleven apart at n=480,
    # which is how they got conflated. sec:streaming's offline reference is hybrid_index(Mc_known=
    # LYSO) -- HANDED the cell, consensus skipped -- while tab:summary's GLINT-(1) row is the BLIND
    # pipeline that derives its own cell. The text used to explain 91-vs-92 as one pipeline scored
    # against the textbook vs the voted cell; at n=480 that reading is refuted outright (357 vs 346).
    "offline_rate_of120":    91,    # offline consensus pipeline (known-hybrid at the same gate);
                                    # independently corroborated by azimuth_validate.py's reconciliation
                                    # block, which records known-hybrid 91 / blind-hybrid 93  (f61a4cf, open)
    # source / sizing ------------------------------------------------------------------------------
    "rep_rate_hz":         35000.0,
    "hit_rate":            0.10,
    "hits_per_s":          3500.0,  # = rep_rate_hz * hit_rate
    "gpus_at_10pct":       1.0,     # = rep_rate_hz * hit_rate * fused_b120_ms/1000
    "xgandalf_blind_ms":   11542.0,
    "xgandalf_speedup":    449.0,   # = xgandalf_blind_ms / blind_ms
    "ffbidx_latency_ms":   4.4,     # per single call -- a LATENCY
    "ffbidx_pipelined_ms": 3.1,     # persistent indexer -- the THROUGHPUT comparator
    "ffbidx_speedup":      12.0,    # = ffbidx_pipelined_ms / fused_b120_ms (throughput vs throughput)
}

DEFAULT_TARGETS = [
    HOME / "git/papers/glint/glint.tex",
    # source of truth for the Confluence "epixUHR 4M -- DRP per-event processing time" comment;
    # the comment is produced by pasting this file through Insert > Markup > Markdown
    HOME / "Desktop/epixuhr_drp_perevent_projections.md",
    HOME / "git/slides/glint/build_glint.py",
    HOME / "git/slides/glint/build_pitch.py",
    HOME / "git/slides/drp/build_drp.py",
    HOME / "git/slides/fftindex/build.py",
    # The repo's own front door. These were NOT guarded until 2026-07-21, which is exactly how
    # README.md kept claiming "~550x the throughput" (11542/21.3, the retired blind figure) long after
    # the paper and the decks had been corrected to ~340x. The guard watched the deliverables we
    # publish and missed the three files a new collaborator opens first.
    HOME / "git/glint/README.md",
    HOME / "git/glint/ROADMAP.md",
    HOME / "git/glint/GLINT_REPORT.md",
]
PDF_TARGETS = [
    HOME / "git/slides/glint/glint_summary.pdf",
    HOME / "git/slides/glint/glint_pitch.pdf",
    HOME / "git/slides/drp/drp_gpu.pdf",
    HOME / "git/slides/fftindex/glint_origin_summary.pdf",
]


@dataclass
class Rule:
    """A pattern that must not appear (or, with `needs`, must appear near a disambiguator)."""
    name: str
    pattern: str
    why: str
    instead: str = ""
    needs: tuple[str, ...] = ()      # if set: a match is OK when one of these is nearby
    exempt: tuple[str, ...] = ()     # a match is OK when one of these is nearby (e.g. "LEGACY")
    window: int = 240                # how far to look for `needs` / `exempt`, in characters
    flags: int = re.I
    _rx: re.Pattern = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rx = re.compile(self.pattern, self.flags)


# ---- 1. retired values -----------------------------------------------------------------------
RETIRED = [
    Rule("blind-15ms", r"(?<![\d.])15\s*ms\b",
         "blind pipeline is 26 ms/frame (re-measured 2026-07-23); 15 ms is a stale pre-fusion value", "26 ms"),
    Rule("blind-21ms", r"(?<![\d.])21(?:\.3)?\s*ms\b",
         "21 ms was a stale end-to-end epoch, superseded by the measured 26 ms", "26 ms"),
    Rule("blind-34ms", r"(?<![\d.])34\s*ms\b",
         "34 ms was the pre-fused-kernel blind pipeline; re-measured 26 ms/frame 2026-07-23 "
         "(slides/README/GLINT_REPORT + PDFs swept the same day)", "26 ms"),
    Rule("fps-47", r"(?<![\d.])47\b(?=[^\n]{0,60}(?:frames?\s*/\s*s|f/s|shots?/s|throughput))",
         "47 f/s is a stale reciprocal; the measured 26 ms gives 39 f/s", "39"),
    # Caught by injecting it back into README after adding README to the targets: the guard read the
    # file and said nothing, because no rule covered it. Adding a target without a matching rule is
    # theatre -- it raises the "checked N files" count and catches nothing.
    Rule("xgandalf-550", r"(?<![\d.])550\s*(?:×|x|\\times)",
         "550x is 11542/21.3 -- the xgandalf ratio taken against a RETIRED blind figure. "
         "Against the current 26 ms it is ~450x. This one outlived its source by weeks in README.md",
         "~450x"),
    Rule("xgandalf-340", r"(?<![\d.])340\s*(?:×|x|\\times)",
         "340x was 11542/34 (the pre-fusion blind figure); against the measured 26 ms it is ~450x", "~450x"),
    Rule("speedup-160", r"(?<![\d.])160\s*(?:×|x|\\times)",
         "the scalar->GPU blind ratio follows 2342/26, not 2342/15", "~90x"),
    # Two traps here, both live for months. (1) HARD-CODED PROSE DRIFTS. This rule named 76%/71%, the
    # values the pair was RETIRED FROM on 2026-08-02 (721d5cc), so had it fired it would have told you
    # to write the number that had just been superseded. It was then hand-corrected to "GLINT-(1) is
    # 77%" while the SAME SENTENCE still called 71% xgandalf's rate -- which FACTS says is 72%. Half a
    # correction, and the half that was missed is exactly the confusion the rule exists to catch. It
    # now interpolates FACTS, so the prose cannot disagree with the table again.
    # (2) `[^\n]` could not span a line break, and _normalize() folds only spaces and tabs -- so in a
    # hard-wrapped .tex every candidate site was saved by where the line happened to break. Use
    # [\s\S] so wrapping is not a hiding place.
    Rule("blind-rate-swap", r"GLINT[\s\S]{0,40}\b71\s*\\?%",
         f"71% is xgandalf's RETIRED blind rate (now {FACTS['xgandalf_blind_rate_pct']}%, "
         f"{FACTS['xgandalf_blind_strict_of120']}/120); GLINT-(1) blind is "
         f"{FACTS['glint_blind_rate_pct']}% ({FACTS['glint1_strict_of120']}/120, paper tab:summary). "
         "Attributing 71% to GLINT understates it and conflates two indexers measured at the same bar",
         f"{FACTS['glint_blind_rate_pct']}%"),
    # The retired pair as the DELIVERABLES actually phrase it -- "76%" headline beside "xgandalf 71%".
    # Scoped to that adjacency on purpose: bare 76% and bare 71% are both still CORRECT elsewhere
    # (91/120 offline, and the 85/120 lattice-bar front end), so an unscoped rule would cry wolf.
    Rule("blind-pair-retired", r"xgandalf\s*(?:\\?geq\s*)?71\s*\\?%",
         f"'xgandalf 71%' is the retired blind pair. Measured at the >=25% bar it is "
         f"{FACTS['xgandalf_blind_rate_pct']}% ({FACTS['xgandalf_blind_strict_of120']}/120) against "
         f"GLINT-(1)'s {FACTS['glint_blind_rate_pct']}% ({FACTS['glint1_strict_of120']}/120)",
         f"xgandalf {FACTS['xgandalf_blind_rate_pct']}%"),
    # The retired HEADLINE, not the retired numbers: 77% and 72% are still correct about the 120-frame
    # subset and appear legitimately three times in sec:comparison, so a rule keyed on either numeral
    # would cry wolf. What is retired is the CLAIM SHAPE -- "matches or exceeds ... blind indexer" -- and
    # quoting the 120 pair as if it were the headline result. At n=480 GLINT is 4 frames BEHIND.
    Rule("blind-lead-retired", r"(?:matches|indexes)\s+(?:more|or\s+exceeds)[\s\S]{0,60}blind\s+indexer",
         f"'exceeds' is not supported at n=480: GLINT-(1) {FACTS['glint1_strict_of480']}/480 against "
         f"xgandalf {FACTS['xgandalf_blind_strict_of480']}/480 leads in DIRECTION (discordant "
         f"{FACTS['mcnemar480_glint_only']} vs {FACTS['mcnemar480_xgandalf_only']}) but only at p=0.18, "
         "so four times the frames still do not license a ranking. The supportable claim is MATCHES",
         f"matches the strongest blind indexer we tested ({FACTS['glint1_strict_of480']} vs "
         f"{FACTS['xgandalf_blind_strict_of480']} of 480 frames, p=0.18)"),
    # The pair-vs-ratio misreading, caught at its one known site. `indexing_rate` = "75/114" is a
    # (strict, loose) PAIR over 120 pushed frames; sec:streaming turned the loose half into a
    # numerator over the 115 post-lock frames and published "114 of 115". Measured, it is 111 of 115
    # (442 of 475 at n=480). Keyed on the exact adjacency because bare 114 and bare 115 are both
    # legitimate elsewhere.
    Rule("postlock-114-of-115", r"114\s*(?:of|/)\s*115",
         f"114 is the LOOSE half of the (strict, loose) pair indexing_rate = {FACTS['indexing_rate']} "
         "over 120 PUSHED frames -- not a numerator over the 115 post-lock frames. The driver's own "
         f"post-lock accept counter gives {FACTS['driver_accept_of115']} of 115 at n=120 and "
         f"{FACTS['driver_accept_of475']} of 475 at n=480",
         f"{FACTS['driver_accept_of475']} of 475 post-lock frames at n=480, "
         f"{FACTS['driver_accept_of115']} of 115 on the 120"),
    Rule("fused-pred-2.4", r"2\.4\s*(?:→|->|-->)\s*0\.45",
         "the fused kernel replaced the 1.46 ms CUDA-graph path, not a 2.4 ms one; "
         "2.4 inflates the gain from 3.1x to an implied 5.3x", "1.46 -> 0.45"),
    # --- the streaming block, superseded 2026-07-21 by the fused peakfind reduction (#41) ---
    Rule("stream-5.58", r"(?<![\d.])5\.58\s*ms",
         "the streaming driver was re-measured at steady state after the fused peakfind reduction; "
         "5.58 ms/frame is the pre-#41 figure", "4.16 ms"),
    Rule("stream-179", r"(?<![\d.])179\b(?=[^\n]{0,60}(?:frames?\s*/\s*s|f/s|fps))",
         "179 f/s is the reciprocal of the retired 5.58 ms", "240"),
    Rule("peakfind-2.38", r"(?<![\d.])2\.38\s*ms",
         "peakfind in the streaming driver is 1.16 ms after #41; 2.38 is the pre-#41 figure. (Do not "
         "restore the 1.58 this rule used to recommend -- that was the cold-warmup artefact below)",
         "1.16 ms"),
    Rule("live-gap-20x", r"[~≈]?\s*20\s*(?:×|x|\\times)(?=[^\n]{0,40}(?:gap|short|hits))",
         "the end-to-end gap to ~3500 hits/s is 3500/275 = ~13x, not ~20x, now that streaming is "
         "3.64 ms/frame", "~13x"),
    # --- the streaming wall, superseded TWICE and for the same reason both times: a merged
    #     optimisation cut the wall and nobody carried it into this table. 4.42 was honest for
    #     2026-07-21; #50 obsoleted it the NEXT DAY. 4.16 was 4.42 minus #68 only, so it still
    #     carried #50's 0.75 ms. Both retire to the measured 3.64.
    Rule("stream-4.42", r"(?<![\d.])4\.42\s*ms",
         "4.42 ms/frame is the 2026-07-21 wall (pre-#50, with predict at 1.11). #50 cut it ~17% the "
         "next day and #68 cut it again; the measured steady-state wall is 3.64 ms/frame", "3.64 ms"),
    Rule("stream-4.16", r"(?<![\d.])4\.16\s*ms",
         "4.16 was 4.42 with ONLY #68 subtracted -- it never had #50's ~0.75 ms taken off, which is "
         "exactly the 0.69 the guard kept reporting as un-attributed. Measured wall is 3.64 ms/frame",
         "3.64 ms"),
    # The exempt is load-bearing, not defensive: glint.tex quotes "4.4 ms / 226 frames/s" for a SINGLE
    # ffbidx call (1000/4.4 = 227). That is a different quantity that happens to round to the same
    # number as the retired streaming figure, and without the exempt this rule sends an editor to
    # "correct" a line that is right.
    Rule("stream-226", r"(?<![\d.])226\b(?=[^\n]{0,60}(?:frames?\s*/\s*s|f/s|fps))",
         "226 f/s is the reciprocal of the retired 4.42 ms", "275",
         exempt=("ffbidx", "single call", "single} call")),
    Rule("stream-240", r"(?<![\d.])240\b(?=[^\n]{0,60}(?:frames?\s*/\s*s|f/s|fps))",
         "240 f/s is the reciprocal of the retired 4.16 ms", "275"),
    Rule("live-gap-16x", r"[~≈]?\s*1[56]\s*(?:×|x|\\times)(?=[^\n]{0,40}(?:gap|short|hits))",
         "the gap to ~3500 hits/s follows the current 275 f/s: 3500/275 = ~13x", "~13x"),
    # --- the COLD-WARMUP stage block, superseded the same day it was written ---
    # These four were published for a few hours between the under-warmed measurement and the warmed
    # re-measure. They are listed because they reached three deliverables, not because they lasted.
    Rule("cold-peakfind-1.58", r"(?<![\d.])1\.58\s*ms",
         "1.58 ms came from a stage benchmark warmed only ONCE; properly warmed peakfind is 1.16 ms",
         "1.16 ms"),
    Rule("cold-predict-1.34", r"(?<![\d.])1\.34\s*ms",
         "1.34 ms was the cold-warmup predict figure, superseded within a day by 1.11 on the same "
         "code. Both are pre-#50. Measured predict is 0.16 ms", "0.16 ms"),
    # --- predict, superseded TWICE by real optimisations: #50 (gate collapse+fusion) then #68
    #     (detector projection fused into the gate). Each value was honest for its own commit.
    # RETRACTION EXEMPT, same shape as peakfind-predict-tie below and added for the same reason: the
    # DRP page explains why the tie framing was wrong, and it cannot do that without naming 1.11 ms.
    # Without this the rule fires on the correction itself, and trap (b) in the notes above says where
    # that ends -- the guard trains you to delete the honest hedging. Markers are explanatory phrases
    # that only appear when the number is being retired, never when it is being asserted.
    Rule("predict-1.11", r"(?<![\d.])1\.11\s*ms",
         "1.11 ms was predict BEFORE #50. It is not an artifact -- re-measured at its own commit "
         "(0623e34) on an A100 it reproduces at 1.13 -- but #50 took it to 0.42 and #68 to 0.16. "
         "Anything derived from it (the peakfind margin, the 'tied' framing, the next-lever ordering) "
         "is stale with it", "0.16 ms",
         exempt=("honest measurement", "briefly claimed", "previously said", "used to say",
                 "reproduces at its own commit")),
    Rule("predict-0.42-stale", r"(?<![\d.])0\.42\s*ms(?=[^\n]{0,60}predict)",
         "0.42 ms is predict between #50 and #68, superseded by #68 (merged), which fuses the "
         "detector projection into the gate kernel", "0.16 ms"),
    # --- the peakfind-vs-predict MARGIN. Wrong in both directions now, so both are retired and the
    #     replacement is a wording change, not a number swap: the two stages are not close.
    Rule("cold-margin-1.18", r"(?<![\d.])1\.18\s*(?:×|x|\\times)",
         "the peakfind-over-predict margin was quoted as 1.18x from cold numbers, then as 1.05x from "
         "warmed ones. Both rest on a predict that never reproduced; measured, peakfind 1.16 vs "
         "predict 0.16 is 7.3x. Peakfind is the largest single stage by a wide margin", "~7x"),
    # Anchored to peakfind/predict on one side or the other. A bare "1.05x" is a perfectly ordinary
    # speedup elsewhere in these files, so an unanchored pattern would be a nag rather than a guard.
    Rule("tie-margin-1.05",
         r"(?:peakfind|predict)[^\n]{0,90}(?<![\d.])1\.0[45]\s*x"
         r"|(?<![\d.])1\.0[45]\s*x(?=[^\n]{0,90}(?:peakfind|predict))",
         "the 1.04x/1.05x peakfind-over-predict 'tie' was an artifact of predict_ms=1.11. Measured, "
         "the margin is 7.3x", "~7x"),
    # The exempt exists because the FIRST thing this rule flagged was the ROADMAP sentence written to
    # retire the tie -- you cannot retract a claim without quoting it. Same shape as legacy-shots
    # below, and deliberately narrow: an explicit retraction marker, not "no longer", which is how the
    # stale Confluence line ("peakfinding is no longer a dominant SFX stage") reads and which must
    # keep firing.
    Rule("peakfind-predict-tie", r"(?:tied?\b[^\n]{0,50}\bpredict|predict[^\n]{0,50}\btied\b)",
         "peakfind and predict are NOT tied and are not within noise of each other: 1.16 vs 0.16 ms, "
         "7.3x. The tie was built on predict_ms=1.11, which never reproduced. This is a FRAMING to "
         "rewrite, not a number to swap -- deliverables that softened the FPGA-front-end argument "
         "because 'there is no single big cost left' need that paragraph revisited",
         "peakfind is the largest single stage, 7.3x predict; the FPGA ceiling is 1.39x",
         exempt=("previously said", "used to say", "never reproduced")),
    Rule("cold-share-0.98", r"(?<![\d.])0\.98\s*ms",
         "index+integrate is 0.85 ms warmed (0.54 + 0.31), not 0.98", "0.85 ms"),
    Rule("legacy-shots", r"\b892\b",
         "892 shots/s is the LEGACY FFT-volume micro-bench, not the current pipeline",
         "mark LEGACY, or use ~29 shots/s", exempt=("legacy",)),
]

# ---- 2. overclaims ---------------------------------------------------------------------------
OVERCLAIM = [
    Rule("live-merge", r"(?:stops? when the data are complete|live merge|real[- ]time merg)",
         "live completeness comes from the running merge accumulator: 179 f/s vs ~3500 needed, "
         "~20x short, and unmerged (#19)",
         "scope the claim to INDEXING, or mark the driver as in review",
         # a sentence that DENIES the live merge is the caveat we want, not an overclaim
         exempt=("not a live merge", "not (yet)", "not yet true", "how close",
                 "20x short", "20× short", "still open", "in review")),
    Rule("steer-run", r"steer a run while the beam is on",
         "asserts a closed loop the measured pipeline does not close", "live hit rate / cell"),
    Rule("mhz-ready", r"ready for MHz[- ]rate",
         "asserts end-to-end readiness from an indexing-kernel projection",
         "one A100 covers the INDEXING tier at 35 kHz x 10% hit"),
    Rule("khz-demonstrated", r"35\s*kHz\s+demonstrated",
         "the 35 kHz figure is a sizing projection for indexing, not an end-to-end run",
         "sizing says ~1 A100 for indexing; end-to-end is ~20x short"),
    Rule("detector-frame-rate", r"index at the detector frame rate",
         "true of the batched 0.26 ms figure, not of the 34 ms it usually sits beside; "
         "state the arithmetic instead",
         "absorbs the indexing load of a 35 kHz source at 10% hit"),
]

# ---- 3. ambiguity ----------------------------------------------------------------------------
AMBIGUOUS = [
    Rule("bare-0.33", r"0\.33",
         "0.33 ms is BOTH fp32 indexing at B=32 and fused box-integration per frame; "
         "a bare one cannot be told apart",
         "name the quantity inline",
         needs=("fp32", "indexing", "integrat", "box-integ", "b=32", "batch 32")),
    Rule("subms-no-batch", r"0\.(?:26|45)\s*ms",
         "a sub-millisecond known-cell figure is throughput amortized over a batch, not a per-frame "
         "latency; without the batch it reads as latency next to ffbidx's 4.4 ms",
         "add /hit and the batch size",
         needs=("/hit", "batch", "b=32", "b=120", "amortiz", "throughput", "steady")),
    # The RATIO evades the rule above. subms-no-batch keys on the literal "0.26 ms", so a sentence
    # that states the same amortized figure as a ratio -- "registration costs ~100x less" -- carried
    # the identical defect straight past the guard and into the abstract (papers/glint 25c8801..b0357c9).
    # 100x IS 26 / 0.26, i.e. a per-image blind latency over a B=120 batched throughput. Unbatched, the
    # per-frame ratio is 25.7 / 16.5 = 1.6x, and tab:summary's own unbatched known-cell row (32 ms) is
    # SLOWER than the blind row (26 ms). So a bare "100x cheaper" overstates the per-frame case ~60x.
    Rule("ratio-100x-no-batch", r"100\s*x\s*(?:less|cheaper|fewer|faster)",
         "the ~100x discovery-vs-registration ratio is 26 ms per-image blind over 0.26 ms/frame "
         "batched at B=120 -- a throughput ratio. Per frame unbatched it is 1.6x (25.7 vs 16.5 ms), "
         "and the unbatched known-cell row is slower than the blind row. Without the batch qualifier "
         "it reads as the cost of registering one frame",
         "say '~100x less cost when batched', as sec:streaming does",
         needs=("batch", "b=32", "b=120", "amortiz", "throughput", "steady"), window=400),
]


# ---- file-level invariants: if the trigger appears, the caveat must appear too ----------------
# Line-level rules cannot express "you may say this only if you also say that". The RTX case is
# exactly that shape: quoting fp32 timings next to a card we have never run on is fine ONLY while
# the page states outright that no number came from one.
REQUIRED = [
    ("RTX", "no number here was measured on an RTX Blackwell",
     "this file argues an RTX Blackwell case from datasheet fp32/$, but every GLINT timing on it is "
     "an A100 (or H100) measurement. Without the blanket disclaimer a reader attributes the fp32 "
     "figures to a card we have never benchmarked -- which is exactly what happened once."),
]


def check_required(path: Path, text: str) -> list[str]:
    out = []
    low = text.lower()
    for trigger, needed, why in REQUIRED:
        if trigger.lower() in low and needed.lower() not in low:
            out.append(f"  {path.name}  [REQUIRED] mentions {trigger!r} without {needed!r}\n"
                       f"      why:  {why}\n")
    return out


def _normalize(text: str) -> str:
    """Fold the notation variants the deliverables use, so one pattern matches all of them."""
    text = unicodedata.normalize("NFKC", text)
    for a, b in (("×", "x"), ("→", "->"), ("–", "-"), ("—", "-"),
                 (" ", " "), (" ", " "), ("≈", "~"), ("∼", "~")):
        text = text.replace(a, b)
    text = text.replace("\\,", " ").replace("\\times", "x").replace("$", "")
    return re.sub(r"[ \t]+", " ", text)


def _read(path: Path) -> str | None:
    if path.suffix == ".pdf":
        try:
            out = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True)
            return out.stdout if out.returncode == 0 else None
        except FileNotFoundError:
            return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _near(hay: str, pos: int, words: tuple[str, ...], window: int) -> bool:
    seg = hay[max(0, pos - window): pos + window].lower()
    return any(w.lower() in seg for w in words)


def scan(path: Path, text: str) -> list[str]:
    """Return a list of human-readable failures for one file."""
    fails: list[str] = []
    norm = _normalize(text)
    # line map: offset -> line number, for reporting
    starts = [0]
    for line in norm.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))

    def lineno(off: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if starts[mid] <= off:
                lo = mid + 1
            else:
                hi = mid
        return lo

    for group, rules in (("RETIRED", RETIRED), ("OVERCLAIM", OVERCLAIM), ("AMBIGUOUS", AMBIGUOUS)):
        for rule in rules:
            for m in rule._rx.finditer(norm):
                if rule.exempt and _near(norm, m.start(), rule.exempt, rule.window):
                    continue
                if rule.needs and _near(norm, m.start(), rule.needs, rule.window):
                    continue
                ln = lineno(m.start())
                quote = norm[starts[ln - 1]:starts[ln]].strip()[:110] if ln else m.group(0)
                fails.append(
                    f"  {path.name}:{ln}  [{group}/{rule.name}] {m.group(0).strip()!r}\n"
                    f"      line: {quote}\n"
                    f"      why:  {rule.why}\n"
                    + (f"      say:  {rule.instead}\n" if rule.instead else "")
                )
    return fails


def check_arithmetic() -> list[str]:
    """The facts table must be internally consistent. A half-applied edit fails here first."""
    F = FACTS
    bad: list[str] = []       # hard failures: the table contradicts itself
    warn: list[str] = []      # advisories: true of the measurement, not fixable by editing a file

    def close(label: str, got: float, want: float, tol: float = 0.03) -> None:
        if want == 0 or abs(got - want) / abs(want) > tol:
            bad.append(f"  FACTS: {label}: table says {got:g}, arithmetic gives {want:g}")

    close("blind_fps = 1000/blind_ms", float(F["blind_fps"]), 1000.0 / float(F["blind_ms"]))
    close("fused_fps = 1000/fused_b120_ms", float(F["fused_fps"]), 1000.0 / float(F["fused_b120_ms"]))
    close("stream_fps = 1000/stream_ms", float(F["stream_fps"]), 1000.0 / float(F["stream_ms"]))
    close("integ_speedup = before/after", float(F["integ_speedup"]),
          float(F["integ_before_ms"]) / float(F["integ_after_ms"]))
    close("xgandalf_speedup = xgandalf_ms/blind_ms", float(F["xgandalf_speedup"]),
          float(F["xgandalf_blind_ms"]) / float(F["blind_ms"]), tol=0.05)
    close("hits_per_s = rep_rate * hit_rate", float(F["hits_per_s"]),
          float(F["rep_rate_hz"]) * float(F["hit_rate"]))
    close("fused_share = index_b40 + integrate", float(F["fused_share_ms"]),
          float(F["index_b40_ms"]) + float(F["integrate_ms"]))
    close("host_total = accumulate + h2d + misc", float(F["host_total_ms"]),
          float(F["accumulate_ms"]) + float(F["h2d_ms"]) + float(F["misc_ms"]))
    # The whole frame must add up. This is the check that would have caught the cold-warmup block:
    # those stage numbers summed to MORE than the end-to-end wall they were supposed to decompose.
    close("stream_ms = sum of all components", float(F["stream_ms"]),
          sum(float(F[k]) for k in ("peakfind_ms", "predict_ms", "index_b40_ms", "integrate_ms",
                                    "peaks_to_q_ms", "accumulate_ms", "h2d_ms", "misc_ms",
                                    "unattributed_ms")))
    close("gpus_at_10pct = rep*hit*t_index", float(F["gpus_at_10pct"]),
          float(F["rep_rate_hz"]) * float(F["hit_rate"]) * float(F["fused_b120_ms"]) / 1000.0, tol=0.12)
    close("ffbidx_speedup = pipelined/fused (throughput:throughput)", float(F["ffbidx_speedup"]),
          float(F["ffbidx_pipelined_ms"]) / float(F["fused_b120_ms"]), tol=0.05)

    # The pool-gate block, EXACT (integer counts -- see the note on close()'s relative tolerance).
    if F["poolgate_bad_locks_gaponly"] - F["poolgate_bad_true_gaponly"] != F["poolgate_bad_wrong_gaponly"]:
        bad.append("  FACTS: poolgate_bad_wrong_gaponly must be locks - true "
                   f'({F["poolgate_bad_locks_gaponly"]} - {F["poolgate_bad_true_gaponly"]})')
    if F["poolgate_bad_locks_gated"] - F["poolgate_bad_true_gated"] != F["poolgate_bad_wrong_gated"]:
        bad.append("  FACTS: poolgate_bad_wrong_gated must be gated locks - gated true locks")
    if (F["poolgate_bad_wrong_gaponly"] - F["poolgate_bad_wrong_gated"]
            != F["poolgate_bad_wrong_removed"]):
        bad.append("  FACTS: poolgate_bad_wrong_removed must be the difference of the two wrong "
                   f'counts ({F["poolgate_bad_wrong_gaponly"]} - {F["poolgate_bad_wrong_gated"]})')
    # The overlap is the load-bearing part of the result: the clean side's LARGEST pool at lock and
    # the bad side's SMALLEST are the same number. That is why no switch removes every wrong lock,
    # and why the honest claim is 192 of 194 (poolgate_bad_wrong_removed), not "all". If a later
    # edit moves one and not the other, the trade-off silently becomes a clean separation that was
    # never measured. This comment said "193 of 194" until review caught it -- 193 was the estimate
    # from the single-pass n_pool profile, which stops at the first ungated lock; 192 is what the
    # real replay gives, because a refused stream keeps running and can lock later. The estimate
    # was written here while the measurement was still going and never re-read against the table
    # it sits beside, which is this file's own failure mode reproduced inside this file.
    # The M3 block's whole claim is "the knob is inert above 4". Two orderings encode it, so an
    # edit that quietly reintroduces a step-count dependence fails here.
    if not (F["m3_blind_steps4_of480"] >= F["m3_blind_steps8_of480"] - 5):
        bad.append("  FACTS: m3_blind_steps4 was MEASURED level with steps8 (284 vs 282); a table "
                   "where 4 is far below 8 has reintroduced the retired 'saturates >= 8' claim")
    if F["m3_blind_steps2_of480"] >= F["m3_blind_steps8_of480"]:
        bad.append("  FACTS: STEPS=2 is the one arm measured WORSE than the default (257 vs 282, "
                   "p = 0.008); the table now says otherwise")
    if F["poolgate_clean_maxpool"] != F["poolgate_bad_minpool"]:
        bad.append("  FACTS: poolgate_clean_maxpool and poolgate_bad_minpool were MEASURED equal "
                   "(54); moving one without the other erases the overlap the claim rests on")
    # ...and the prose beside these keys must not drift from them. The comment above quoted a
    # superseded 193 while the table said 192; a guard whose own commentary can go stale silently
    # is not guarding itself. Read this file back and require the derived count to appear in it.
    _self = open(__file__, encoding="utf-8").read()
    if f'{F["poolgate_bad_wrong_removed"]} of {F["poolgate_bad_locks_gaponly"] - F["poolgate_bad_true_gaponly"]}' not in _self:
        bad.append(f'  FACTS: the pool-gate commentary must state '
                   f'"{F["poolgate_bad_wrong_removed"]} of '
                   f'{F["poolgate_bad_wrong_gaponly"]}" -- it has drifted from the table')

    # The blind pair, derived from its counts. Both percentage keys were DEAD -- defined and read
    # nowhere -- across the whole period the pair drifted 76/71 -> 77/72, so the RETIRED rule below
    # went on naming the superseded values with every run green. Deriving them is what makes that rule
    # text falsifiable: edit a count without its percentage and this fails.
    # EXACT, not close(). close() is 3% RELATIVE, which on an integer percentage is nearly two whole
    # points -- 92 -> 95 frames moves the rate 77 -> 79 and would have slipped through silently. Rates
    # are integers here; compare them as integers.
    for _pct_key, _cnt_key in (("glint_blind_rate_pct", "glint1_strict_of120"),
                               ("xgandalf_blind_rate_pct", "xgandalf_blind_strict_of120")):
        _want = round(100.0 * int(F[_cnt_key]) / 120.0)
        if int(F[_pct_key]) != _want:
            bad.append(f"  FACTS: {_pct_key} = {F[_pct_key]}% but {_cnt_key} = {F[_cnt_key]}/120 rounds to "
                       f"{_want}% -- a count and its percentage were edited apart")
    # Same derivation for the n=480 rows. Note the denominator differs, so this cannot be folded into
    # the loop above -- and folding it would be the exact mistake that makes a percentage stop tracking
    # its count.
    for _pct_key, _cnt_key in (("glint_blind_rate_pct_480", "glint1_strict_of480"),
                               ("xgandalf_blind_rate_pct_480", "xgandalf_blind_strict_of480")):
        _want = round(100.0 * int(F[_cnt_key]) / 480.0)
        if int(F[_pct_key]) != _want:
            bad.append(f"  FACTS: {_pct_key} = {F[_pct_key]}% but {_cnt_key} = {F[_cnt_key]}/480 rounds to "
                       f"{_want}% -- a count and its percentage were edited apart")
    # The n=120 ordering (92 > 86) is still a true fact about that subset, but it is NO LONGER a
    # headline: the synopsis and intro now quote the 480 tie, because the lead did not survive 4x the
    # frames. Keep the subset ordering pinned so a re-measure cannot silently invert the text at
    # sec:comparison that still discusses it.
    if int(F["glint1_strict_of120"]) <= int(F["xgandalf_blind_strict_of120"]):
        bad.append("  FACTS: GLINT-(1) no longer leads xgandalf on the 120-frame subset -- sec:comparison "
                   "discusses that lead and its 8:2 discordant split explicitly. Rewrite that passage, "
                   "do not renumber it")
    # THE claim the paper now leads with: at n=480 the two are INDISTINGUISHABLE. That is a statement
    # about the discordant pairs, not about the rates, so check it where it lives -- recompute the exact
    # two-sided McNemar and fail if it stops supporting "indistinguishable". A rate edit that leaves the
    # discordant counts alone would otherwise sail through.
    _b01, _b10 = int(F["mcnemar480_glint_only"]), int(F["mcnemar480_xgandalf_only"])
    _m = _b01 + _b10
    _p = 1.0 if _m == 0 else min(1.0, 2.0 * sum(comb(_m, _k) for _k in range(0, min(_b01, _b10) + 1)) / 2.0 ** _m)
    if _p <= 0.05:
        bad.append(f"  FACTS: the n=480 blind pair is no longer a tie (exact McNemar p = {_p:.3g} from "
                   f"{_b01} vs {_b10} discordant) -- the synopsis and intro say 'matches ... p=0.7'. "
                   "Rewrite that claim rather than editing these counts")
    if abs((_b01 - _b10) - (int(F["glint1_strict_of480"]) - int(F["xgandalf_blind_strict_of480"]))) != 0:
        bad.append("  FACTS: the n=480 discordant counts and the n=480 totals disagree -- their difference "
                   "must equal the difference of the totals (concordant frames cancel). One of the two "
                   "was re-measured without the other")

    # These three numbers ARE the paper's sec:streaming yield paragraph. sec:streaming used to claim
    # streaming "indexes no fewer frames than the offline pipeline" because the rates were "properties
    # of the consensus computation, which streaming preserves exactly"; the measured yields refuted it,
    # and papers/glint f421e79 rewrote the passage to quote the gap outright -- "61--65% against 76%
    # offline", "what streaming changes is latency and per-frame yield, not the cell". So the paper now
    # DEPENDS on the gap rather than denying it: 73/120 = 61%, 78/120 = 65%, 91/120 = 76%. Closing the
    # gap makes the published sentence wrong in the other direction, so revisit the text -- do not just
    # renumber this table.
    if int(F["stream_rate_rescue_of120"]) >= int(F["offline_rate_of120"]):
        bad.append("  FACTS: streaming yield now meets offline -- sec:streaming's '61--65% against 76% "
                   "offline' and its 'shortfall is completeness, not discovery' framing are now stale. "
                   "Revisit that text rather than just editing these numbers")
    if int(F["stream_rate_of120"]) > int(F["stream_rate_rescue_of120"]):
        bad.append("  FACTS: warmup_rescue now indexes FEWER frames than the baseline it rescues on top "
                   "of -- one of the two was re-measured without the other")
    # sec:streaming prints the DERIVED percentages, not these counts, so a change here can strand the
    # published sentence while both checks above stay green. The concrete case is the blind retry #73
    # located: shipping it takes 78 -> 88, which is still under offline's 91 and still above the
    # baseline, so both stay silent while "61--65%" goes wrong. Pin the counts to what the paper prints.
    # (Added in #74, dropped by 721d5cc's merge, restored here.)
    for _k, _paper_pct in (("stream_rate_of120", 61), ("stream_rate_rescue_of120", 65),
                           ("offline_rate_of120", 76)):
        _got = round(100.0 * int(F[_k]) / 120.0)
        if _got != _paper_pct:
            bad.append(f"  FACTS: {_k} = {F[_k]}/120 is {_got}%, but sec:streaming prints {_paper_pct}% "
                       f"-- fix the paper's '61--65% against 76% offline' sentence, not just this table")
    # sec:streaming's n=480 block. The ORDERING is the paragraph's point, and it inverts between the
    # two run lengths, which is exactly why both are pinned: at n=120 the retry falls SHORT of offline
    # (88 < 91) and the text says it "closes 77%"; at n=480 it lands PAST offline (365 > 357) and the
    # text says it "does more than close it". Lose either relation and one of those two sentences goes
    # wrong -- and they are adjacent, so a single careless re-measure can invert one and not the other.
    if int(F["retry_rate_of480"]) <= int(F["offline_rate_of480"]):
        bad.append("  FACTS: the blind retry no longer beats offline at n=480 -- sec:streaming's 'does "
                   "more than close it: 365 of 480, eight frames past the offline pipeline's 357' is now "
                   "FALSE. Rewrite that passage, do not renumber it")
    if int(F["retry_rate_of120"]) >= int(F["offline_rate_of120"]):
        bad.append("  FACTS: the retry now meets offline at n=120 too -- sec:streaming's 'the crossover "
                   "is a property of run length' rests on it falling SHORT there (88 < 91). Revisit")
    # The narrowing is the other half of the paragraph: 13 points at 120, 6-7 at 480. Derive both from
    # the counts so a re-measure cannot leave the prose's "narrows with run length" unsupported.
    _g120 = 100.0 * (int(F["offline_rate_of120"]) - int(F["stream_rate_rescue_of120"])) / 120.0
    _g480 = 100.0 * (int(F["offline_rate_of480"]) - int(F["stream_rate_rescue_of480"])) / 480.0
    if not _g480 < _g120:
        bad.append(f"  FACTS: the streaming gap no longer narrows with run length ({_g120:.1f} pt at 120, "
                   f"{_g480:.1f} pt at 480) -- sec:streaming's '13 points there, 6-7 here' and the "
                   "withdrawal of 'the gap is a standing tax' both depend on it")
    # The false-lock bound the paper quotes as "a 95% upper bound of 0.75% per ordering" is
    # 1 - 0.05**(1/N) for ZERO events in N trials. It is only that number while the count is zero and
    # the trial count is 400; either moving silently invalidates the printed bound.
    if int(F["seqstop_false_locks"]) != 0:
        bad.append("  FACTS: the sequential stop now has false locks -- sec:streaming's 'zero false "
                   "locks in 400 trials' and the 0.75% upper bound derived from it are both wrong. "
                   "Recompute the bound for a nonzero count, do not just edit the number")
    _ub = 100.0 * (1.0 - 0.05 ** (1.0 / int(F["seqstop_trials"])))
    if abs(_ub - 0.75) > 0.01:
        bad.append(f"  FACTS: {F['seqstop_trials']} trials give a 95% upper bound of {_ub:.2f}%, but "
                   "sec:streaming prints 0.75% -- the bound and the trial count were edited apart")
    # sec:streaming and sec:consensus BOTH now say the handed cell buys nothing at the strict bar --
    # 357 against consensus's 361, a tie on the discordant frames. That is a sign-test claim, so check
    # it as one; comparing totals is what let the last four coincidences pass for identities.
    _cs, _ct = int(F["cons_vs_true_strict_cons"]), int(F["cons_vs_true_strict_true"])
    _m = _cs + _ct
    _p = 1.0 if _m == 0 else min(1.0, 2.0 * sum(comb(_m, _k) for _k in range(min(_cs, _ct) + 1)) / 2.0 ** _m)
    if _p <= 0.05:
        bad.append(f"  FACTS: handing the pipeline the cell is now significant at the strict bar "
                   f"({_cs} vs {_ct}, p={_p:.3g}) -- sec:consensus says 'consensus recovers everything "
                   "knowing the cell is worth' at that bar, and sec:streaming that it 'buys it "
                   "nothing'. Rewrite both")
    # ...and the OTHER half of that sentence: at the >=10-reflection bar the handed cell IS ahead,
    # which is why the claim is now stated per bar instead of flatly. If it inverts, so does the text.
    if int(F["truecell_10refl_of480"]) <= int(F["cons_10refl_of480"]):
        bad.append("  FACTS: the handed cell no longer leads at the >=10-reflection bar "
                   f"({F['truecell_10refl_of480']} vs {F['cons_10refl_of480']}) -- sec:consensus's "
                   "'the true cell does pull ahead' at that bar is now wrong")
    # "aggregate agreement to within the discordant noise" is a claim about a SIGN TEST on the
    # discordant pairs, so check it there. If the split ever becomes significant the two paths are not
    # interchangeable and tab:summary's batched footnote has to say so.
    for _tag, _a, _b in (("120", "bvp_batched_only_120", "bvp_perframe_only_120"),
                         ("480", "bvp_batched_only_480", "bvp_perframe_only_480")):
        _x, _y = int(F[_a]), int(F[_b])
        _m = _x + _y
        _p = 1.0 if _m == 0 else min(1.0, 2.0 * sum(comb(_m, _k) for _k in range(min(_x, _y) + 1)) / 2.0 ** _m)
        if _p <= 0.05:
            bad.append(f"  FACTS: batched vs per-frame is now significant at n={_tag} ({_x} vs {_y}, "
                       f"p={_p:.3g}) -- the paper says the two agree in aggregate to within the "
                       "discordant noise. They are no longer interchangeable; rewrite tab:summary's "
                       "batched footnote and sec:arch, do not renumber")
        if _x + _y != int(F["bvp_disagree_of" + _tag]):
            bad.append(f"  FACTS: bvp_disagree_of{_tag} does not equal its two halves")
    if int(F["stream_fail_nobody480"]) > int(F["stream_fail_of480"]):
        bad.append("  FACTS: more streaming failures are recovered by nobody than exist")
    # the claim that motivates the whole live-merge caveat
    if float(F["stream_fps"]) >= float(F["hits_per_s"]):
        bad.append("  FACTS: stream_fps now meets hits_per_s -- the 'not a live merge' caveat in the "
                   "deliverables is stale and must be revisited, not just this table")
    # the fused kernels being a minority of the streaming budget is why peakfind is the wall
    if float(F["fused_share_ms"]) >= float(F["stream_ms"]) / 2:
        bad.append("  FACTS: fused kernels are no longer a minority of stream_ms -- the 'peakfind is "
                   "the wall' framing needs rechecking")
    # "peakfind is the LARGEST stage" is a framing no arithmetic was watching, and it is the one the
    # deliverables lean on to argue for an FPGA front end.
    #
    # HISTORY, because this margin has now been wrong in both directions. #41 cut peakfind 2.79 ->
    # 1.58 while predict appeared to stay at 1.34, giving a 1.18x margin; the warmed re-measure made
    # it 1.16 vs 1.11, a 1.05x TIE, and three deliverables were rewritten around that tie. The tie was
    # an artifact: predict was never 1.11 (see the FACTS provenance above), and #68 has since taken
    # the corrected 0.42 to 0.16. The real margin is 7.3x, so peakfind is the largest single stage by
    # a wide margin and the FPGA-front-end argument stands on its own numbers again.
    #
    # The tie framing is therefore RETIRED, not just renumbered -- deliverables saying peakfind and
    # predict are "tied", or quoting the 1.05x/1.18x margin, are now positively false. RETIRED rules
    # peakfind-predict-tie, tie-margin-1.05 and cold-margin-1.18 catch that wording in the files.
    m = float(F["peakfind_ms"]) / float(F["predict_ms"])
    if float(F["predict_ms"]) >= float(F["peakfind_ms"]):
        bad.append("  FACTS: predict is now >= peakfind -- 'peakfind is the largest single stage' is "
                   "FALSE, and the FPGA-offload argument built on it must be rewritten, not renumbered")
    elif m < 1.25:
        # A WARNING, deliberately not a failure. No edit to any deliverable can make this condition
        # go away -- it is a property of the measurement -- so failing on it would leave the checker
        # permanently red, and a guard that cries wolf gets weakened or switched off, which is how
        # the drift it exists to catch comes back (see the rule-writing traps above).
        warn.append(f"  FACTS: peakfind {F['peakfind_ms']} ms vs predict {F['predict_ms']} ms is {m:.2f}x -- "
                    + ("a TIE within run-to-run noise. 'Peakfind is the largest stage' is technically "
                       "true and practically meaningless; deliverables must say the two are tied, and "
                       "an FPGA peakfind offload cannot be sold on peakfind's dominance alone"
                       if m < 1.10 else
                       "still the largest, but not comfortably; say the margin out loud"))
    # Peakfind dominating predict is NOT by itself the FPGA case, and the ceiling is what says so:
    # peakfind is 28% of the frame, so deleting it outright bounds at stream_ms/(stream_ms-peakfind).
    # Quote that bound next to the claim, or the reader infers the offload removes the wall.
    ceiling = float(F["stream_ms"]) / (float(F["stream_ms"]) - float(F["peakfind_ms"]))
    if ceiling < 1.5:
        warn.append(f"  FACTS: peakfind is {100*float(F['peakfind_ms'])/float(F['stream_ms']):.0f}% of "
                    f"the frame, so removing it ENTIRELY bounds out at {ceiling:.2f}x "
                    f"({F['stream_ms']} -> {float(F['stream_ms'])-float(F['peakfind_ms']):.2f} ms). "
                    "It is the largest single stage, but state the ceiling wherever the FPGA front end "
                    "is argued -- the case is that pixels stop arriving at all, not that peakfind is "
                    "the wall")
    # The un-attributed remainder. Raised as a warning rather than a failure for the same reason as
    # above: it is a property of the measurement, and the honest response is to re-measure stream_ms,
    # not to edit a deliverable. It exists because the corrected predict_ms stopped hiding it.
    if float(F["unattributed_ms"]) > 0.10 * float(F["stream_ms"]):
        warn.append(f"  FACTS: {F['unattributed_ms']} ms/frame "
                    f"({100*float(F['unattributed_ms'])/float(F['stream_ms']):.0f}% of the frame) is "
                    "UN-ATTRIBUTED -- the measured stages no longer decompose stream_ms. Do not quote "
                    "the stage table as a complete breakdown. RE-MEASURED 2026-08-04 "
                    "(experiments/remeasure_stream_ms.py, A100): the warmed steady-state wall is "
                    "3.49-3.66 ms, not 4.16, which would leave only 1-5% un-attributed. Held at 4.16 "
                    "pending a confirmation run on the original hardware -- see the FACTS comment.")
    # The host bucket exceeds peakfind, but INVESTIGATED 2026-07-22 it is NOT the throughput lever:
    # batching it recovers ~1% and the streaming wall is GPU-compute-bound. No warning is raised for
    # it -- the earlier nag rested on a benchmark artifact (build timed inside the loop).
    return bad, warn


def main(argv: list[str]) -> int:
    if "--facts" in argv:
        width = max(len(k) for k in FACTS)
        print("GLINT authoritative numbers\n" + "-" * 60)
        for k, v in FACTS.items():
            print(f"  {k:<{width}}  {v}")
        return 0

    paths = [Path(a) for a in argv[1:] if not a.startswith("--")]
    if not paths:
        paths = list(DEFAULT_TARGETS)
        if "--pdf" in argv:
            paths += PDF_TARGETS

    fails, advisories = check_arithmetic()
    if fails:
        print("ARITHMETIC (the facts table contradicts itself):")
        print("\n".join(fails) + "\n")
    if advisories:
        print("ADVISORY (not a failure -- a framing the numbers no longer comfortably support):")
        print("\n".join(advisories) + "\n")

    checked = skipped = 0
    for path in paths:
        text = _read(path)
        if text is None:
            print(f"  SKIP {path} (missing, or pdftotext unavailable)")
            skipped += 1
            continue
        checked += 1
        found = scan(path, text) + check_required(path, text)
        if found:
            fails.extend(found)
            print(f"{path}:")
            print("".join(found))

    print(f"{'-' * 60}\nchecked {checked} file(s), skipped {skipped}")
    if fails:
        print(f"FAIL: {len(fails)} problem(s). Fix the deliverable, or -- if a measurement really "
              f"changed -- update FACTS and re-run every target.")
        return 1
    print("OK: no retired values, overclaims, or ambiguous figures found.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
