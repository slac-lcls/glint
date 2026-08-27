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
  3. AMBIGUITY           -- `0.33 ms` means fp64 INDEXING at B=32 (fp32 is 0.31 since #165) *and*
                            fused INTEGRATION per frame; `0.17 ms` means fp64 indexing at B=120 *and*
                            the un-attributed residual of the 3.64 ms driver wall. A bare one is a
                            defect; it must sit near a word that says which.
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
import math
from math import comb
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path.home()
REPO = Path(__file__).resolve().parent.parent   # the checkout this file ships in

# --------------------------------------------------------------------------------------- the facts
# Measured, and the single source of truth. Provenance in the trailing comment: PR number where
# merged, or "open" where it is not yet.
FACTS: dict[str, float | str] = {
    # indexing -------------------------------------------------------------------------------------
    "blind_ms":            25.7,    # blind pipeline GLINT-1 (hybrid_index), per frame, one A100; re-measured 2026-07-23 HEAD be55302, 3 reps; paper displays ~26
    "blind_fps":           39.0,    # = 1000/blind_ms
    "known_perframe_ms":   16.5,    # per-frame known-cell rescue (replica_gpu.index_known_gpu_cell)
    "graph_ms":            1.46,    # batched + CUDA graph                                    (#14)
    # 2026-08-27: obj_fused's candidate axis is now split over blockIdx.y (#165).  Bit-exact --
    # obj_fused's inl/sub are BITWISE identical at K=120/4096/5760, bench_fused max|dM| 1.42e-13 on
    # both trees, same_lattice 120/120, rate (80,115) unchanged -- so these are the SAME pipeline
    # measured faster, not a different answer.  A/B on one exclusive A100 (sdfampere018), both trees
    # in one allocation; the pre-change tree reproduced the retired values exactly (0.261/0.447/
    # 0.158/0.323 vs the 0.26/0.45/0.16/0.33 recorded here since #16).
    # REPLICATED ACROSS FACILITIES the same day: NERSC Perlmutter A100-SXM4-40GB, a wholly different
    # stack (py3.12 / torch 2.6.0 / cuda 12.4 / cupy 14.2 against S3DF's py3.9 / torch 2.1.0), gave
    # 0.266 -> 0.172 fp64 B=120 and 0.162 -> 0.140 fp32 -- every row within 1-3% of S3DF, with
    # max|dM| 1.42e-13 and rate (80,115) identical on both. The B=96 ~ B=64 tie reproduces there
    # too, which is what rules it out as a one-machine artefact. Values below are the S3DF numbers
    # (same device class as the #16 measurement they supersede); Perlmutter is the cross-check.
    "fused_b32_ms":        0.33,    # fused kernels, fp64, batch 32       (was 0.45 pre-#165)
    "fused_b120_ms":       0.17,    # fused kernels, fp64, batch 120      (was 0.26 pre-#165)
    "fused_b32_fp32_ms":   0.31,    # fp32 INDEXING at batch 32           (was 0.33 pre-#165)
    "fused_b120_fp32_ms":  0.14,    # fp32 INDEXING at batch 120          (was 0.16 pre-#165)
    "fused_fps":           5900.0,  # = 1000/fused_b120_ms
    # The VALUE stands but was never a clean saturation: B=120 beat B=64 by 19% before #165 and by
    # 26% after, and B=96 ~ B=64 in both trees because 120 frames at B=96 is a ragged 96+24 while
    # B=120 is one batch -- that curve is batch-count quantisation, not occupancy.  The rationale
    # below survives #165: splitting obj's candidates did NOT remove the need to batch (measured --
    # split at B=16 is 0.579, still slower than pre-#165 at B=120, and the B-spread WIDENS from
    # 2.92x to 3.41x), because anneal/refine still run one block per frame.
    # RENAMED 2026-08-27 (#167) from `saturating_batch`. The old name asserted saturation, and the
    # provenance directly above denies it: B=120 beats B=64 by 19% pre-#165 and 26% post, so a
    # --facts consumer was being handed a claim its own comment refutes. 64 is the streaming
    # driver's DEFAULT batch (stream_driver.py:481), which is what it has always actually been.
    # No genuine saturation point is recorded because none was measured -- throughput improves
    # through B=120, the largest batch the 120-frame benchmark can form.
    "driver_default_batch": 64,      # stream_driver's B default; NOT a measured saturation point
    "saturating_batch":     64,      # deprecated compatibility alias; use driver_default_batch
    # ⚠ 2026-08-27 (#167): this DISAGREES with what the benchmark measures today. bench_fused.py's
    # rate() is the same (strict, loose) pair over the same 120 pushed frames -- gpass() returns
    # (correct-lattice AND >=25% matched, correct-lattice AND >=10 refl) -- and it reads (80, 115),
    # not (75, 114), in FOUR independent runs: origin/main and the #165 branch, at S3DF and at
    # NERSC, fp64 and fp32. So 75/114 is stale, and the drift is NOT from #165: the pre-change tree
    # gives 80/115 too. Which commit moved it was not bisected -- several accuracy-affecting changes
    # have landed since #16 recorded this (dedup radius, binarisation, the alias gate). Recorded as
    # measured rather than left contradicting the bench; if the old pair is wanted for a historical
    # comparison, take it from #16, not from here. Not a published number (it appears in no .tex).
    "indexing_rate":       "80/115",
    # Percentages are ROUNDED, not floored (changed 2026-08-02). tab:summary previously mixed the two:
    # DIALS printed 27% for 32/120 = 26.67 (rounded) while GLINT-(1) printed 76% for 92/120 = 76.67
    # (floored), so three "correct" values for one measurement were in circulation. Counts are now
    # shown inline in the table, which makes the convention checkable instead of inferred.
    # The COUNTS are the measurement; the percentages are DERIVED from them and checked below. Until
    # 2026-08-02 the counts lived only in these trailing comments and both percentage keys were read by
    # nothing, which is precisely how the pair drifted: 76/71 stayed written into the rule text below
    # while the measurement moved to 77/72, and every run stayed green because no code connected them.
    "glint1_strict_of120":         92,  # GLINT-(1), blind + cross-frame consensus, >=25%-of-spots bar
    # ⚑ THIS INTEGER IS DEVICE-SENSITIVE, measured 2026-08-27. The same pipeline on CPU returns 91,
    # bit-deterministically, and it is NOT code drift: 91 comes back identically at HEAD, at 8da091b
    # (pre-#126 binarisation) and at 7ca3b49 -- the very commit that recorded this 92 -- with the
    # same consensus cell and the same support each time. The cause is the gate's marginality, not
    # a defect: TWO of the 120 frames sit within 0.002 of the 0.25 boundary and are ONE PEAK from
    # flipping (frame 118 at 79/318 = 0.2484, frame 89 at 34/137 = 0.2482), so a single peak
    # crossing the 0.15 hkl-residual tolerance -- routine between CUDA and CPU kernels -- moves the
    # published count. azimuth_validate.py's reconciliation block independently records 93 for this
    # same arm at the same gate, which is the same effect in the other direction. So the honest
    # reading of this key is "92 +/- 1, A100": quote it with the device, and do not treat a 91 or a
    # 93 from a re-run as a contradiction. Nothing here is retired; the value stands as measured.
    "glint_blind_rate_pct":        77,  # = round(100 * glint1_strict_of120 / 120)
    "xgandalf_blind_strict_of120": 86,  # xgandalf blind, SAME bar, SAME peak list -- a different indexer
    "xgandalf_blind_rate_pct":     72,  # = round(100 * xgandalf_blind_strict_of120 / 120)
    # The SAME two blind arms at the CORRECT-LATTICE bar -- lattice right, no coverage requirement.
    # Added 2026-08-27 because the strict pair above is device-sensitive at +/-1 (see the note on
    # glint1_strict_of120) while these are not: 115 reproduced identically on CPU at HEAD, at
    # 8da091b and at 7ca3b49, and 115/120 is the same figure the decks were corrected to. Scored by
    # experiments/score_glint_gate.py and experiments/xgandalf/score_xg_gate.py -- one gate
    # function, so the two arms cannot drift apart. The GAP IS WIDER HERE than at the strict bar
    # (115 vs 94, against 92 vs 86), which is worth knowing before quoting only the strict pair.
    "glint1_lattice_of120":       115,  # GLINT-(1) blind, correct reduced cell, no coverage bar
    "glint1_lattice_rate_pct":     96,  # = round(100 * glint1_lattice_of120 / 120)
    "xgandalf_lattice_of120":      94,  # xgandalf blind, same bar, same peak list
    "xgandalf_lattice_rate_pct":   78,  # = round(100 * xgandalf_lattice_of120 / 120)
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
    # is annotated against below: the 114 was the LOOSE half of the then-current (strict, loose) pair
    # 75/114 over 120 PUSHED frames, not a numerator over the 115 post-lock ones. The real value is
    # 111. ⚠ THE COLLISION GOT WORSE, NOT BETTER: `indexing_rate`'s loose half re-measured to 115 on
    # 2026-08-27 (#167), which is now numerically EQUAL to the post-lock frame count. So "115" alone
    # is ambiguous between the two, and any sentence using it must say which. That is why the keys
    # are named `driver_accept_of115` and `indexing_rate` rather than sharing a bare number.
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
    # Binary vs weighted peaks, n=480 (experiments/weight_sweep.py). The family binarises before
    # the transform; this measured whether a soft intensity weight beats it. It does not -- every
    # arm is significantly worse at blind, and the damage across the up-weighting arms is a linear
    # function of the Kish effective sample size (r = +0.997), i.e. reweighting discards peaks
    # rather than extracting information. `inverse` is the falsifier arm and the one point off
    # that line, which is what shows DIRECTION matters on top of variance.
    "pw_blind_binary_of480":     282,  # the control; == the STEPS=8 arm of steps_sweep, exactly
    "pw_hybrid_binary_of480":    361,
    "pw_blind_quarter_of480":    253,  # gentlest weighting, still -29 (p < 0.001)
    "pw_blind_sqrt_of480":       200,
    "pw_blind_linear_of480":     116,  # -166; intensity-proportional is catastrophic
    "pw_blind_inverse_of480":    251,  # falsifier: DOWN-weights strong peaks, -31 at n_eff 0.526
    "pw_neff_sqrt":            0.693,  # ...vs sqrt's -82 at a HIGHER n_eff of 0.693
    "pw_neff_inverse":         0.526,
    "pw_r_neff_upweighting":   0.997,  # Pearson r(n_eff, blind delta), excluding the falsifier
    "pw_qpow_default":           1.0,  # the shipped |q|^-QPOW exponent the "binary" control IS.
                                       # Read from glint_fast.py below, not just described: if the
                                       # default moves, the control is no longer the shipped weight
                                       # and every arm is measured against something nobody runs.
    # DISCORDANT splits, because "significantly worse" is a p-value, not a margin. Same reason the
    # M3 block stores them: totals cannot decide an exact McNemar. Recomputed in check_arithmetic.
    "pw_quarter_gained":          18,  # the gentlest weighting, and still p = 4.2e-4
    "pw_quarter_lost":            47,
    "pw_sqrt_gained":             22,
    "pw_sqrt_lost":              104,
    "pw_linear_gained":           10,
    "pw_linear_lost":            176,
    "pw_inverse_gained":          35,  # the falsifier is significantly worse TOO -- that both
    "pw_inverse_lost":            66,  # directions hurt is half the argument (p = 2.7e-3)
    # M3 ascent steps, re-measured at n=480 (experiments/steps_sweep.py, 13 arms 2->80, strict bar,
    # exact McNemar vs the shipped default). What the block records is that NO arm from 4 to 80
    # shows a detectable difference, and how large an undetected one could still be -- not that the
    # knob is inert, which would assert the null. See the CI bound checked below.
    "m3_steps_default":            8,   # shipped; a throughput choice, not an accuracy one
    "m3_blind_steps8_of480":     282,
    "m3_blind_steps4_of480":     284,   # p = 0.894: no difference DETECTED vs 8. Not
                                        # "saturation is at 4" -- that reads a null as a
                                        # finding, which is the error this block avoids.
    "m3_blind_steps80_of480":    293,   # p = 0.169 vs 8 -- 10x the work, no detectable gain
    "m3_blind_steps2_of480":     257,   # p = 0.008 vs 8 -- the ONLY significant arm, and it is worse
    "m3_hybrid_steps8_of480":    361,
    "m3_hybrid_steps2_of480":    342,   # p = 0.003 vs 8
    "m3_hybrid_steps32_of480":   366,   # the observed max; p = 0.458 vs 8, so not shown better
    "m3_ms_steps8":              6.1,   # ms/frame blind, A100
    "m3_ms_steps80":             9.8,   # = 1.62x, for no rate change anyone can detect
    # Discordant splits live in M3_SPLITS below -- every arm, both channels. Four hand-picked
    # endpoints were not enough: null endpoints say nothing about a non-monotonic interior.
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
    # Jungfrau-4M lysozyme (cxil1015922 r0033) -- tab:realindex's last row and tab:realmerge's.
    #
    # ⚑ RECORD-SOURCED, NOT RE-MEASURED. Unlike every A100 number above, these three were taken
    # from the run record of job 35507050 (Aug rerun, current GLINT + the glint#130 fix), not from a
    # run made while writing this table. Their standing is therefore different: quote them, but if a
    # fresh run disagrees, that is a NEW MEASUREMENT to reconcile, not drift in the paper.
    #
    # THE DENOMINATOR IS THE MEASUREMENT. 1563 = 404 + 373 + 409 + 377, counted directly across the
    # four r0033 `.cxi` files. Naming it is the whole point of this block: a MISSING denominator is
    # what produced the retired "93% (consensus support 54/60)". 86e89d3 (2026-07-06) introduced
    # that as prose with no denominator and no surviving log, and nothing reproduces it -- the July
    # `lyso_glint_int.stream` gives 1476/1563 (94.4%) and the Aug rerun 1506/1563, while the Aug
    # rerun reports the consensus support UNSET, so 54/60 has no source at all. RETIRED rules
    # `jungfrau-93pct` and `jungfrau-support-54-60` catch both in the files.
    "jungfrau_frames_total":  1563,  # MEASURED: 404+373+409+377 over the four r0033 .cxi files
    "jungfrau_blind_of1563":  1506,  # blind indexed                        (job 35507050)
    "jungfrau_blind_rate_pct":  96,  # = round(100 * jungfrau_blind_of1563 / jungfrau_frames_total)
    "jungfrau_final_of1563":  1482,  # the GATED SUBSET of that 1506        (job 35507050)
    "jungfrau_final_rate_pct":  95,  # = round(100 * jungfrau_final_of1563 / jungfrau_frames_total)
    # merge quality --------------------------------------------------------------------------------
    # THE PROTOCOL IS PART OF THE NUMBER, and these keys carry it because an unlabelled CC* is
    # exactly the ambiguity this file exists to prevent. Every value below is a `partialator` merge
    # at the default 10 scaling/post-refinement cycles, which is what tab:realmerge's caption states.
    # That is NOT the protocol of the glint#129 A/B: its clipmean-vs-median comparison runs on THIS
    # SAME r0033 data at UNITY scale, native, and reads R_split 33.0% vs 26.7% with a CC* penalty of
    # -0.0052. Those numbers are not comparable with the 31.6% below, and the #129 A/B is what
    # established that they are different protocols rather than a discrepancy.
    #
    # 1482 is the crystal count tab:realmerge prints, so this row and the indexing row above come
    # from the SAME run -- which is the reason the Jungfrau row was restated from the record at all.
    "jungfrau_ccstar":        0.915,  # partialator, the 1482-crystal set   (tab:realmerge)
    "jungfrau_rsplit_pct":     31.6,  # partialator                         (tab:realmerge)
    "jungfrau_iovers":          7.7,  # <I/sigma>, same merge
    # The CrystFEL/XGANDALF comparator: the SAME frames, merged identically. The paragraph's point is
    # that the two statistics move in OPPOSITE directions -- xgandalf takes CC*, GLINT takes R_split
    # -- so both relations are pinned below. A one-sided "GLINT wins" is the overclaim to prevent.
    "jungfrau_xg_ccstar":     0.930,  # xgandalf is AHEAD here
    "jungfrau_xg_rsplit_pct":  34.9,  # ...and BEHIND here
    # THE headline CC*, and a different dataset from the Jungfrau row. S12 says so out loud ("the
    # CC*=0.90 row is Proteinase K (cxidb-45); the Jungfrau-4M lysozyme row is a different dataset
    # and sits at CC*=0.915") because the two were being read as one number. Keyed apart here for
    # the same reason.
    "pk45_ccstar":             0.90,  # cxidb-45 Proteinase K, 290 frames, partialator
    "pk45_rsplit_pct":         31.0,
    "pk45_iovers":              7.8,
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
    # above is a (strict, loose) PAIR over 120, not a ratio, and reading it as 80/115=70% is the exact
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
    "gpus_at_10pct":       0.6,     # = rep_rate_hz * hit_rate * fused_b120_ms/1000
    "xgandalf_blind_ms":   11542.0,
    "xgandalf_speedup":    449.0,   # = xgandalf_blind_ms / blind_ms
    "ffbidx_latency_ms":   4.4,     # per single call -- a LATENCY
    "ffbidx_pipelined_ms": 3.1,     # persistent indexer -- the THROUGHPUT comparator
    "ffbidx_speedup":      18.0,    # = ffbidx_pipelined_ms / fused_b120_ms (throughput vs throughput)
}

# Every arm-vs-default discordant split from the n=480 M3 sweep, both channels: (arm-only,
# default-only) -- frames this STEPS indexed and 8 did not, and the reverse. Its own table rather
# than 48 flat FACTS keys, and COMPLETE rather than sampled: the claim is about the whole 4..80
# range and the sweep is non-monotonic, so checking endpoints would leave the interior unguarded
# while looking rigorous. check_arithmetic recomputes every p and every interval from these.
M3_SPLITS = {
    (  2, "blind"): ( 29,  54),
    (  3, "blind"): ( 31,  45),
    (  4, "blind"): ( 29,  27),
    (  5, "blind"): ( 18,  24),
    (  6, "blind"): ( 24,  27),
    ( 10, "blind"): ( 18,  20),
    ( 12, "blind"): ( 21,  18),
    ( 16, "blind"): ( 30,  24),
    ( 24, "blind"): ( 27,  19),
    ( 32, "blind"): ( 30,  23),
    ( 48, "blind"): ( 22,  23),
    ( 80, "blind"): ( 32,  21),
    (  2, "hybrid"): ( 10,  29),
    (  3, "hybrid"): ( 16,  23),
    (  4, "hybrid"): ( 16,  16),
    (  5, "hybrid"): ( 14,  15),
    (  6, "hybrid"): ( 20,  18),
    ( 10, "hybrid"): ( 12,  18),
    ( 12, "hybrid"): ( 10,  16),
    ( 16, "hybrid"): ( 14,  17),
    ( 24, "hybrid"): ( 14,  16),
    ( 32, "hybrid"): ( 17,  12),
    ( 48, "hybrid"): ( 16,  12),
    ( 80, "hybrid"): ( 19,  18),
}

DEFAULT_TARGETS = [
    # glint.tex RETIRED from the targets 2026-08-26: it is the SUPERSEDED draft (the submission is
    # glint_rewrite_JAC_refined.tex below), and when the new rules landed it lit up with seven
    # already-fixed-in-the-rewrite values. Fixing a dead document to satisfy the guard is the
    # inverse of the guard's job; the file now carries a SUPERSEDED banner instead.
    # ...and the file that is actually being SUBMITTED, which had never been guarded (2026-08-26).
    # Same shape as the README note below, one step worse: README was an unguarded file, this was an
    # unguarded DELIVERABLE. `glint.tex` is the draft the rewrite superseded, so every correction
    # made in the rewrite landed in a file the guard did not read, and every stale value the
    # 2026-08-26 adversarial review found -- the Jungfrau `93% (54/60)`, a `5.8%` warm-up share, a
    # `346` caption at n=480, a `~100`-peak cap that ships nowhere -- was sitting in the one document
    # going to a journal. The cover letter is here for the same reason: it quotes headline numbers,
    # it goes out with the manuscript, and nothing was checking it.
    # ⚑ Both were VERIFIED GREEN before being added, so wiring them in blocked nothing; the rules
    # added alongside them are what makes the targets mean something (see the note below).
    HOME / "git/papers/glint/glint_rewrite_JAC_refined.tex",
    # ...and its SI, missed when the manuscript was added: the SI carries PARALLEL copies of the
    # dataset tables, and on 2026-08-26 -- the first day it was ever scanned -- its copy of the
    # Jungfrau row still read the retired `93% (support 54/60)` that the main text had already
    # been corrected out of. A guarded manuscript with an unguarded SI just moves the drift one
    # file over (fixed in papers d300440; caught by the jungfrau rules the moment they saw it).
    HOME / "git/papers/glint/glint_SI.tex",
    HOME / "git/papers/glint/cover_letter_JAC.tex",
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
    # ...resolved from THIS CHECKOUT, not from a fixed clone path. These three live in the repo, so
    # keying them to ~/git/glint meant a worktree, a CI runner or anyone else's clone silently
    # checked a different copy or none at all -- which is why the guard had never run in CI on the
    # files it is most able to check.
    REPO / "README.md",
    REPO / "ROADMAP.md",
    REPO / "GLINT_REPORT.md",
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
    needs_all: tuple[tuple[str, ...], ...] = ()  # each any-of group must have a nearby match
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
    # The pre-#165 fused-kernel family. obj_fused's candidate split (bit-exact) moved every one of
    # these on 2026-08-27, and they are exactly the kind that outlive their source -- 0.26 was quoted
    # in the abstract, two docs, three docstrings and four rules in THIS file. Retired together so a
    # half-applied edit cannot leave one behind (cf. xgandalf-550, which outlived its source by weeks).
    # ⚠ THE UNIT IS NOT RELIABLY ADJACENT, AND \s DOES NOT MATCH A LATEX TIE. The manuscript writes
    # "$0.26$~ms" and "$3.8$~kHz", and _normalize() deliberately preserves ASCII "~" (see the
    # pk45-merge-row note). With `\s*ms` these rules reported glint_rewrite_JAC_refined.tex CLEAN
    # while its INTRODUCTION, tab:summary row and streaming section all still said 0.26 -- a guard
    # that could not see the file it was pointed at. Found by an adversarial sweep on #167, after
    # Copilot missed it too. So: [\s~]* for the separator, and the unit itself OPTIONAL, because a
    # table cell carries its unit in the column header ("\textbf{0.26} & \textbf{3800} \\").
    # The (?!\s*\\?%) guard keeps "0.26%" (a percentage in multishot.py) out of it.
    # Dropping the unit entirely over-fires on layout COORDINATES ("x+0.26,cy+0.24" in the deck
    # builders), so the separator is widened instead: [\s~",]{0,3} reaches across a LaTeX tie AND
    # across a Python string boundary ("26 -> 0.26","ms/f ...). The \textbf{} alternative catches the
    # tab:summary cell, where the unit lives in the column header and no separator will ever help.
    Rule("fused-b120-0.26",
         r"(?<![\d.])0\.26\b[\s~\",]{0,3}(?:ms|kHz)|\\textbf\{0\.26\}(?![\s~]*\\?%)",
         "0.26 ms/frame was the pre-#165 B=120 fp64 known-cell figure; the candidate-split kernel "
         "measures 0.17 ms on the same A100, bit-exact", "0.17 ms"),
    Rule("fused-b32-0.45", r"(?<![\d.])0\.45[\s~]*ms",
         "0.45 ms/frame was the pre-#165 B=32 fp64 figure; it is now 0.33 ms", "0.33 ms"),
    # 0.33 is the awkward one: it is LIVE for fp64-at-B=32 and for box-integration, and RETIRED for
    # fp32-at-B=32 (now 0.31).  So it cannot be retired outright, and the bare-0.33 ambiguity rule
    # below does not help -- "fp32" is one of ITS accepted disambiguators, so "fp32 indexing at B=32
    # is 0.33 ms" satisfies both that rule and subms-no-batch while contradicting FACTS.  Found by
    # Copilot on #167.  Fire only on the fp32 PAIRING, either order, and exempt a sentence that also
    # says fp64 (a deliberate fp32-vs-fp64 contrast legitimately puts both near 0.33).
    # ⚠ TWO defects in the first version of this rule, both caught by Copilot on #167 round 2, and
    # both worth stating because they are easy to repeat:
    #   1. It required "ms", so the bare form "0.33 fp32 indexing" walked straight through.
    #   2. It used exempt=("fp64",). `exempt` is a PROXIMITY test, not a scope: _near looks over the
    #      rule's whole +/-240 window, so "fp32 indexing at B=32 is 0.33 ms. fp64 results are
    #      discussed next." was silently exempted by an fp64 in the NEXT SENTENCE. An exemption
    #      cannot express "fp64 owns this number" -- only "fp64 is somewhere nearby".
    # So the exclusion is encoded IN the pattern, scoped to the pairing, and exempt is gone:
    #   forward  fp32 ... 0.33  with NO fp64 between (tempered), so a contrast where fp64 owns the
    #            0.33 cannot match;
    #   reverse  0.33 ... fp32  within 15 chars, tempered the same way, AND with fp32 not followed
    #            by its own decimal -- that last clause is what separates the stale "0.33 fp32
    #            indexing" from the legitimate "fp64 is 0.33 ms, fp32 is 0.31 ms".
    # Pinned by test_fp32_b32_033_rule.py: 5 must-fire, 6 must-not-fire, both Copilot cases included.
    Rule("fp32-b32-0.33",
         # Tempered against fp64 AND against the integration wording: "in fp32, fused
         # box-integration reaches 0.33 ms" is LIVE (integration is the other current meaning of
         # 0.33 and is not precision-tagged), so fp32 alone must not condemn a 0.33.
         r"fp32(?:(?!fp64|integrat|box-integ)[\s\S]){0,40}?(?<![\d.])0\.33\b"
         # Reverse window 60, not 15: "0.33 ms per frame for known-cell indexing in fp32 at B=32"
         # puts 41 characters between the number and its attribution, and 15 missed it. The
         # not-followed-by-its-own-decimal clause is what keeps the contrast sentences safe at this
         # width -- "fp64 is 0.33 ms at B=32, fp32 is 0.31 ms" still passes because that fp32 has a
         # 0.31 of its own within 20 chars.
         r"|(?<![\d.])0\.33\b(?:(?!fp64|integrat|box-integ)[\s\S]){0,60}?fp32(?![^\n]{0,20}\d\.\d)",
         "0.33 was fp32 indexing at B=32 BEFORE #165; it is now 0.31 (0.33 is the fp64 B=32 figure, "
         "and the fused box-integration per frame)", "0.31"),
    # The fp32 B=120 figure has the SAME shape of problem: 0.16 was fp32-at-B=120 (now 0.14), but
    # 0.16 is ALSO predict's live per-frame cost (the "7.3x predict (0.16 ms)" line in the DRP
    # deliverables), so it cannot be retired outright either. Same tempered pairing against fp32.
    # This rule is not hypothetical: adding it immediately caught two stale sites the 0.26/0.45
    # sweep had missed -- slides/drp/build_drp.py and the DRP projections page both still said
    # "0.16 ms at B=120 (0.33 at B=32)" for fp32.
    Rule("fp32-b120-0.16",
         # Tempered against predict in BOTH orders: "fp32 indexing is 0.14 ms at B=120; predict
         # 0.16 ms" is entirely correct and the forward branch used to condemn it. My own negative
         # test covered only the reverse ordering -- testing one direction is testing half a rule.
         r"fp32(?:(?!fp64|predict)[\s\S]){0,60}?(?<![\d.])0\.16\b"
         r"|(?<![\d.])0\.16\b(?:(?!fp64|predict)[\s\S]){0,60}?fp32(?![^\n]{0,20}\d\.\d)",
         "0.16 was fp32 indexing at B=120 BEFORE #165; it is now 0.14 (0.16 is predict's per-frame "
         "cost, which is why this is scoped to the fp32 pairing)", "0.14"),
    Rule("fused-fps-3800",
         r"(?<![\d.])3[,.]?800\b|3\.8[\s~]*kHz",
         "3.8 kHz was 1000/0.26; against the measured 0.17 ms/frame it is ~5.9 kHz", "~5.9 kHz"),
    Rule("ffbidx-12x", r"(?<![\d.])12\s*(?:×|x|\\times)(?=[^\n]{0,80}(?:ffbidx|pipelined))",
         "12x was ffbidx_pipelined_ms/0.26; against 0.17 it is ~18x", "~18x"),
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
    # ⚑ (3) THE 40-CHAR WINDOW IS NARROW, AND DELIBERATELY LEFT THAT WAY. Recorded here because the
    # next person will be tempted to widen it. The manuscript has three legitimate 71% sites -- the
    # 85/120 correct-lattice bar, the "~71% ceiling" framing, and tab:modules' "cluster-FFT seeding
    # (M1, 71%)" -- and the last of these clears this rule ONLY because the nearest "GLINT" sits just
    # past 40 characters. So the pass is incidental: a caption reflow could bring the two together
    # and the rule would fire on a correct row. It is NOT widened, because widening trades a
    # possible miss for a certain false positive on three real sites, and a guard that cries wolf
    # gets switched off (see the ceiling advisory's note on the same trade). If this ever does fire
    # on tab:modules, the fix is to anchor the pattern to the CLAIM ("GLINT ... blind ... 71%"),
    # not to move the number.
    Rule("blind-rate-swap", r"GLINT[\s\S]{0,40}\b71\s*\\?%",
         f"71% is xgandalf's RETIRED blind rate (now {FACTS['xgandalf_blind_rate_pct']}%, "
         f"{FACTS['xgandalf_blind_strict_of120']}/120); GLINT-(1) blind is "
         f"{FACTS['glint_blind_rate_pct']}% ({FACTS['glint1_strict_of120']}/120, paper tab:summary). "
         "Attributing 71% to GLINT understates it and conflates two indexers measured at the same bar",
         f"{FACTS['glint_blind_rate_pct']}%"),
    # The retired pair as the DELIVERABLES actually phrase it -- "76%" headline beside "xgandalf 71%".
    # Scoped to that adjacency on purpose: bare 76% and bare 71% are both still CORRECT elsewhere
    # (91/120 offline, and the 85/120 lattice-bar front end), so an unscoped rule would cry wolf.
    # ...and the pair as the READMEs phrase it, which slipped BOTH rules above for two years'
    # worth of drift: "76% vs 71% at the same gate" carries no `xgandalf` adjacent to the 71 (so the
    # rule below misses) and sits >40 chars from the word GLINT (so blind-rate-swap misses too).
    # Found 2026-08-26 by an audit of the LUTE docs, in README.md -- a GUARDED target that had been
    # running green over a retired pair. Keyed on the two numerals ADJACENT to each other, which is
    # what makes it the pair rather than either legitimate lone number (91/120 offline = 76%, the
    # 85/120 lattice-bar ceiling = 71%).
    # `[^.]`, NOT `[^.\n]`: ordinary Markdown/LaTeX wrapping puts a newline between the two
    # numerals ("76% vs\n71%"), and a class that excludes \n is defeated by reflowing the very
    # paragraph it guards. That is the same defect the stream-band rule had (see :824) -- caught
    # there by the #154 review and reintroduced here, which is why it is spelled out twice.
    Rule("blind-pair-adjacent-retired", r"\b76\s*\\?%[^.]{0,30}?\b71\s*\\?%",
         f"'76% vs 71%' is the RETIRED blind pair -- the counts moved to "
         f"{FACTS['glint1_strict_of120']}/120 and {FACTS['xgandalf_blind_strict_of120']}/120 while "
         "the percentages stayed written down. Quote the counts, and say which n you mean: at "
         f"n=480 the arms are {FACTS['glint1_strict_of480']} vs "
         f"{FACTS['xgandalf_blind_strict_of480']}, a tie at p=0.18",
         f"{FACTS['glint_blind_rate_pct']}% ({FACTS['glint1_strict_of120']}/120) vs "
         f"{FACTS['xgandalf_blind_rate_pct']}% ({FACTS['xgandalf_blind_strict_of120']}/120)",
         # A deck builder's own comment RECORDING that it once shipped the retired pair is not a
         # claim of it. Same historical-mention vocabulary the rules at :743 and :772 already use.
         exempt=("it shipped", "previously said", "used to say", "build behind")),
    # Widened 2026-08-26 to cross table pipes: the report's row is `| xgandalf | blind | 71% |`,
    # where the old `xgandalf\s*` adjacency could not reach past the cell separators, so the same
    # retired pair sat in a second guarded file. A short bounded gap, still same-line, still
    # anchored on the word xgandalf -- so a lone 71% elsewhere stays legitimate.
    Rule("blind-pair-retired", r"xgandalf[^.\n]{0,20}?(?:\\?geq\s*)?\b71\s*\\?%",
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
    # The pair-vs-ratio misreading, caught at its one known site. `indexing_rate` = "80/115" is a
    # (strict, loose) PAIR over 120 pushed frames; sec:streaming turned the loose half into a
    # numerator over the 115 post-lock frames and published "114 of 115". Measured, it is 111 of 115
    # (442 of 475 at n=480). Keyed on the exact adjacency because bare 114 and bare 115 are both
    # legitimate elsewhere.
    # ⚠ Do NOT interpolate FACTS['indexing_rate'] into this message. Interpolation is right for a
    # LIVE value (that is why blind-rate-swap does it) and wrong for a HISTORICAL one: 114 was the
    # loose half of the pair AS IT STOOD when the error was published (75/114). The pair re-measured
    # to 80/115 on 2026-08-27, and interpolating turned this message into the false claim that 114
    # is the loose half of 80/115. Same trap as the hard-coded 76%/71% two rules down, approached
    # from the opposite side -- the fix is not "always interpolate", it is "interpolate what is live".
    Rule("postlock-114-of-115", r"114\s*(?:of|/)\s*115",
         "114 was the LOOSE half of the (strict, loose) indexing_rate pair AS PUBLISHED (75/114; the "
         "pair now measures 80/115) over 120 PUSHED frames -- not a numerator over the 115 post-lock "
         "frames. The driver's own "
         f"post-lock accept counter gives {FACTS['driver_accept_of115']} of 115 at n=120 and "
         f"{FACTS['driver_accept_of475']} of 475 at n=480",
         f"{FACTS['driver_accept_of475']} of 475 post-lock frames at n=480, "
         f"{FACTS['driver_accept_of115']} of 115 on the 120"),
    Rule("fused-pred-2.4", r"2\.4\s*(?:→|->|-->)\s*0\.45",
         "the fused kernel replaced the 1.46 ms CUDA-graph path, not a 2.4 ms one; "
         "2.4 inflates the gain from 3.1x to an implied 5.3x", "1.46 -> 0.45"),
    # --- the streaming block, superseded 2026-07-21 by the fused peakfind reduction (#41) ---
    # ⚑ These two `instead` values were themselves STALE, and interpolate FACTS now for the same
    # reason blind-rate-swap does. stream-5.58 recommended "4.16 ms" -- retired by stream-4.16 two
    # rules below -- and stream-179 recommended "240" -- retired by stream-240. Both were correct
    # when written and neither moved when the wall did, so the guard was holding two pieces of
    # advice that its own rules refuse. Caught by test_guard_advice_is_not_itself_retired in
    # experiments/test_check_numbers_ci.py, which exists to make this class of drift impossible to
    # reintroduce; the same defect in legacy-shots ("~29 shots/s" = 1000/34) was found the same way.
    Rule("stream-5.58", r"(?<![\d.])5\.58\s*ms",
         "the streaming driver was re-measured at steady state after the fused peakfind reduction; "
         "5.58 ms/frame is the pre-#41 figure", f"{FACTS['stream_ms']} ms"),
    Rule("stream-179", r"(?<![\d.])179\b(?=[^\n]{0,60}(?:frames?\s*/\s*s|f/s|fps))",
         "179 f/s is the reciprocal of the retired 5.58 ms", f"{FACTS['stream_fps']:.0f}"),
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
    # WIDENED to 400 (2026-08-26). The exempt was clearing at the default 240 only because the word
    # "ffbidx" happened to fall inside it, and in a LaTeX table that is a property of the line
    # breaks, not of the text: tab:summary's ffbidx row and the "\S" footnote that says the 4.4 ms /
    # 226 f/s pair is a SINGLE CALL sit in different parts of the float, so one reflow of the table
    # strands the marker and the guard starts demanding a "correction" to a correct line. The
    # exempt's job is to recognise a different quantity that rounds to the same number, and it should
    # not depend on where a row wrapped.
    Rule("stream-226", r"(?<![\d.])226\b(?=[^\n]{0,60}(?:frames?\s*/\s*s|f/s|fps))",
         "226 f/s is the reciprocal of the retired 4.42 ms", "275",
         exempt=("ffbidx", "single call", "single} call"), window=400),
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
    # --- the values the 2026-08-26 adversarial review found in the MANUSCRIPT, retired here at the
    #     same time the manuscript's two files were added to DEFAULT_TARGETS above.
    #
    # This group is the reason those targets are worth adding at all. The file's own lesson, learned
    # when README.md was wired in and a retired 550x survived in it silently: adding a target without
    # a matching rule is theatre -- it raises the "checked N files" count and catches nothing. Each
    # rule below names a value the review actually found, so the target list and the rule list were
    # grown together rather than one without the other.
    #
    # ⚑ WINDOWS ARE MEASURED, NOT GUESSED. Every proximity window here was tuned against the real
    # files before being written down; the notes on the individual rules record what the tuning
    # found, because "200 looked safe" is how a rule acquires a false positive.
    Rule("jungfrau-93pct", r"jungfrau[\s\S]{0,120}?(?<![\d.])93\s*\\?%",
         f"the Jungfrau-4M 93% is NOT REPRODUCIBLE from any surviving artifact. It entered as prose "
         f"in 86e89d3 (2026-07-06) with no denominator and no log; against the MEASURED denominator "
         f"of {FACTS['jungfrau_frames_total']} frames the July stream gives 1476 (94.4%) and the Aug "
         f"rerun (job 35507050) {FACTS['jungfrau_blind_of1563']} "
         f"({FACTS['jungfrau_blind_rate_pct']}%). A bare 93% here also collides with cxidb-45's "
         "genuine 93% (842/907), which is a different dataset",
         f"{FACTS['jungfrau_blind_rate_pct']}% blind "
         f"({FACTS['jungfrau_blind_of1563']}/{FACTS['jungfrau_frames_total']}); "
         f"{FACTS['jungfrau_final_rate_pct']}% final "
         f"({FACTS['jungfrau_final_of1563']}/{FACTS['jungfrau_frames_total']})"),
    # 120 chars, and the width is load-bearing rather than arbitrary: at 200 this rule FIRES ON THE
    # CORRECTED ROW. tab:realindex's Jungfrau line now reads "96% blind (1506/1563); 95% final
    # (1482/1563)", and 200 characters is enough to reach past \bottomrule and \end{tabular} into the
    # footnote's legitimate "93% blind rate" for cxidb-45 -- so the wider window would send an editor
    # to correct the one row that had just been fixed. Measured on the file, not estimated.
    # The prose site is caught too: glint.tex's "(Jungfrau-4M) it recovers the cell blindly /
    # (consensus support 54/60) and indexes 93%" spans a line break, which is why this uses [\s\S]
    # rather than [^\n] -- see the blind-rate-swap note above for how wrapping hides a match.
    Rule("jungfrau-support-54-60", r"(?<![\d.])54\s*/\s*60(?![\d.])",
         "the '54/60 consensus support' for the Jungfrau row has NO source: the Aug rerun (job "
         "35507050) reports the consensus support unset, and no earlier log survives. It is the "
         "other half of the unreproducible 93% and travels with it",
         f"drop the support figure, or state one from the run record: "
         f"{FACTS['jungfrau_blind_of1563']}/{FACTS['jungfrau_frames_total']} indexed blind"),
    # Unanchored on purpose, and safe because it can be: '54/60' is a two-number adjacency that
    # occurs nowhere else in any deliverable. It is also the rule that carries the PROSE site on its
    # own if the Jungfrau window above ever falls short, so the pair does not share a single point of
    # failure.
    Rule("compare3-346-at-480",
         r"(?<![\d.])480(?![\d.])[\s\S]{0,120}?(?<![\d.])346(?![\d.])"
         r"|(?<![\d.])346(?![\d.])[\s\S]{0,120}?(?<![\d.])480(?![\d.])",
         f"346 of 480 is compare3.py's blind-top-1 + consensus + rescue arm, NOT GLINT-(1). "
         f"GLINT-(1) IS hybrid_index(Mc_known=None) and indexes "
         f"{FACTS['glint1_strict_of480']}/480. The two are indistinguishable on the 120 subset "
         "(both 92) and 15 frames apart at n=480, so quoting 346 as GLINT-(1) was invisible until "
         "the set grew -- and it changes the comparison's p-value as well as its count",
         f"{FACTS['glint1_strict_of480']} (and name the arm in the caption)"),
    # Anchored to 480 because a bare 346 is ordinary: the DRP page carries "346-1,136 us" as a
    # microsecond range, and an unanchored rule would nag about it forever. Verified against that
    # file -- at a 120-char window it does not reach any 480.
    Rule("warmup-5.8pct",
         r"warm[\s\S]{0,120}?(?<![\d.])5\.8\s*\\?%|(?<![\d.])5\.8\s*\\?%[\s\S]{0,120}?warm",
         "5.8% of 120 frames is 7, and the warm-up is neither 7 nor a fraction: it is a FIXED 5 "
         "frames (warmup_rescue, worth 5/N -- 4.2% at 120, 1.2% at 400, negligible at DAQ rates). "
         "The 5.8% conflates it with the median-6-frames time-to-lock, which is a different "
         "quantity that happens to sit beside it",
         "4.2% (a fixed 5 frames of 120), or give the fixed count and let the reader divide"),
    # (?<![\d.]) is doing real work here: without it this fires inside "105.8/105.8/75.5 A", the
    # cxidb-62 unit cell: ACG (Agrocybe cylindracea galectin), which the paper's SI S1 gives as
    # hexagonal P6 at 105.8/105.8/75.5 A, and which cxidb.org entry 62 identifies. Entry 83 is
    # beta-lactamase (EuXFEL/AGIPD) and is a different cell. (An earlier version of this comment
    # offered a=b as corroboration; it is not diagnostic -- tetragonal and cubic satisfy it too,
    # and what makes P6 is gamma=120. The identification rests on the deposition and the stated
    # space group, not on the axis lengths.) Trap (a) in the notes above, reproduced
    # exactly. The dataset label was wrong here, and in this rule's fixture in
    # experiments/test_check_numbers_ci.py, until 2026-08-27. The REGEX was always right, so
    # nothing the guard did was affected -- but a provenance comment naming the wrong deposition
    # is exactly the kind of thing this file exists to prevent.
    Rule("integ-fused-6-32x", r"(?<![\d.])6\s*-{1,2}\s*32\s*x",
         f"6--32x is the fused box-integration measured against NUMPY, i.e. against a baseline two "
         f"optimisations back. Its own before/after is {FACTS['integ_after_ms']} -> "
         f"{FACTS['integ_fused_ms']} ms, which is 23x; quoting the numpy range beside the 7.6/0.33 "
         f"row makes the same lever look like two different results",
         f"23x (from {FACTS['integ_after_ms']} to {FACTS['integ_fused_ms']} ms)"),
    Rule("stream-band-120-at-480",
         r"61\s*-{1,2}\s*65\s*\\?%[^.]{0,80}?(?<![\d.])(?:480|323|331|357)(?![\d.])"
         r"|(?<![\d.])(?:480|323|331|357)(?![\d.])[^.]{0,80}?61\s*-{1,2}\s*65\s*\\?%",
         f"61--65% is the n=120 streaming band ({FACTS['stream_rate_of120']}--"
         f"{FACTS['stream_rate_rescue_of120']} of 120). At n=480 the same two arms give "
         f"{FACTS['stream_rate_of480']}--{FACTS['stream_rate_rescue_of480']} of 480, which is "
         f"67--69%. Quoting the 120-frame band beside a 480-frame count states a rate that was "
         "never measured on that set",
         f"67--69% ({FACTS['stream_rate_of480']}--{FACTS['stream_rate_rescue_of480']} of 480), or "
         f"keep 61--65% and quote it against 120"),
    # ⚑ This one is here because it is MY OWN error, not a found one: the Fig 9 caption paired the
    # 120-frame band with 480-frame counts and the review caught it before this rule existed. It
    # fires on nothing today -- both bands are correctly labelled everywhere, including in glint.tex,
    # which states 61--65% against 120 counts on one line and 67--69% against 480 counts on another.
    # A rule that fires on nothing is still worth its lines when the defect it names has already
    # happened once.
    # The window is [^.]{0,80} -- SAME SENTENCE -- and that is the second thing testing changed.
    # A 160-char [\s\S] window fired on a paragraph that quotes BOTH bands correctly, each against
    # its own denominator ("61--65% (73--78 of 120) ... At n=480 the same arms read 67--69%
    # (323--331 of 480)"), which is exactly how glint.tex already writes it and how anyone would
    # write the comparison. The defect is 61--65% presented AS the 480 set's rate, so the proximity
    # that matters is within one clause, not within 160 characters.
    # ⚑ THE SENTENCE, NOT THE LINE. This was [^.\n] until review of #154, and the newline was doing
    # damage the period was not: _normalize() folds spaces and tabs but deliberately keeps line
    # breaks (scan() maps offsets to line numbers off the normalized text), so in a hard-wrapped
    # .tex an ORDINARY WRAP between "61--65%" and the count defeated the rule outright. The exact
    # defect this rule exists for -- the Fig 9 caption pairing the 120-frame band with 480-frame
    # counts -- is a caption, i.e. the text most likely to be wrapped by the editor rather than by
    # the author. Same lesson as blind-rate-swap's note (2): where a line happens to break must
    # never be a hiding place. A period still ends the window, so the two-bands-stated-correctly
    # paragraph in the negative injection stays green whether it is wrapped or not.
    #
    # ⚑ `needs`, not a bare refusal, and this rule was WRONG without it (found in review of #154):
    # its own `instead` asks the writer to "name the bar and the arm, e.g. '>=10-reflection gate,
    # offline'", and the rule then fired on exactly that sentence. A guard whose advice its own
    # pattern rejects has no correct output -- the only way to satisfy it was to delete a true,
    # correctly qualified historical citation, which is trap (b) in the notes above (the guard
    # trains you to delete the honest hedging) reproduced in a rule written to prevent it.
    # The disambiguators are the words that make 117/120 mean what it is: the BAR (the paper writes
    # it as `$\geq$10-reflection gate`, which _normalize() renders `\geq10-reflection gate`, so
    # keying on "10-reflection" covers the LaTeX, the ASCII ">=10-reflection" and the prose "loose
    # bar" spelling) and the ARM ("offline"). Window left at the default 240: the labelled forms in
    # the manuscript put the qualifier within ~30 characters of the count, and 240 was checked
    # against the real targets rather than assumed -- none of them carries a 117/120 at all today,
    # so the width buys tolerance for a table cell without any measured false positive to trade.
    Rule("consensus-117-of-120", r"(?<![\d.])117\s*(?:of|/)\s*120(?![\d.])",
         "117/120 is the OFFLINE hybrid at the LOOSE (>=10-reflection) bar, and it is quoted as if "
         "it were the streaming or single-frame result. The bar is the whole difference: at the "
         f"strict bar the same pipeline gives {FACTS['glint1_strict_of120']}/120, and the driver's "
         f"own post-lock counter gives {FACTS['driver_accept_of115']} of 115. An unlabelled "
         "117/120 reads as a headline rate for a pipeline that was never measured at 97.5%",
         "name the bar and the arm, e.g. '>=10-reflection gate, offline' -- do not renumber",
         needs=("10-reflection", "loose bar", "loose gate", "offline")),
    Rule("legacy-shots", r"(?<![\d.])892(?![\d.])",
         "892 shots/s is the LEGACY FFT-volume micro-bench, not the current pipeline",
         # Interpolated, not hard-coded: this `instead` used to read "~29 shots/s", which is
         # 1000/34 -- the reciprocal of a blind figure retired by rule blind-34ms above. So the
         # guard was recommending a number another of its own rules retires, and had it ever fired
         # it would have walked an editor into a fresh violation. Exactly the drift the
         # blind-rate-swap note describes, found in this file rather than in a deliverable.
         f"mark LEGACY, or use ~{FACTS['blind_fps']:.0f} shots/s", exempt=("legacy",)),
    # `\b892\b` was the pattern for months and passed only by luck of the neighbouring digits: the
    # run ID mfxl1038923 contains "892" with a digit on each side, so \b refused it. It would NOT
    # have refused "892.5" or a trailing "892." -- \b treats the decimal point as a boundary, which
    # is trap (a) in the notes above. (?<![\d.])892(?![\d.]) keeps the run ID safe for the stated
    # reason instead of the accidental one, and closes the decimal case at the same time.
    Rule("fps-29", r"(?<![\d.\-])29\b(?=[^\n]{0,60}(?:frames?\s*/\s*s|f/s|shots?/s|fps))",
         f"29 f/s is 1000/34 -- the reciprocal of the blind figure retired by blind-34ms. The "
         f"measured {FACTS['blind_ms']} ms gives {FACTS['blind_fps']:.0f} f/s",
         f"{FACTS['blind_fps']:.0f}"),
    # (?<!\-) is load-bearing and was found by testing, not by reading: without it this rule fires on
    # the ISO date in GLINT_REPORT.md's banner, "Snapshot: 2026-06-29. The throughput figures here
    # are superseded" -- a "29" followed inside 60 characters by the word "throughput". The fix is
    # both halves: refuse a hyphen-prefixed 29, and drop "throughput" from the unit list that
    # fps-47 carries, since for this numeral it is the word that makes dates collide.
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
    # #165 did not remove this collision, it MOVED it: fp32 at B=32 went 0.33 -> 0.31, and fp64 at
    # B=32 went 0.45 -> 0.33. So a bare 0.33 is still ambiguous, but now between fp64 indexing and
    # box-integration rather than fp32 indexing and box-integration. Same rule, different pair.
    Rule("bare-0.33", r"0\.33",
         "0.33 ms is BOTH fp64 indexing at B=32 (post-#165) and fused box-integration per frame; "
         "a bare one cannot be told apart",
         "name the quantity inline",
         # "fp32" is NOT in this list: neither current meaning of 0.33 is fp32 (fp32 at B=32 is
         # 0.31 since #165), so naming fp32 near a 0.33 does not disambiguate it -- it misattributes
         # it. That pairing is caught by the fp32-b32-0.33 RETIRED rule above instead.
         needs=("fp64", "indexing", "integrat", "box-integ", "b=32", "batch 32")),
    # After #165 the known-cell pair is 0.17/0.33, and 0.33 now COLLIDES with fused box-integration
    # per frame -- which is a genuine per-frame latency, so this rule's premise does not apply to it.
    # "integrat"/"box-integ" are accepted for the same reason bare-0.33 accepts them: they name the
    # other quantity. Widening the escape hatch, not the rule.
    # [\s~]* here too: fixing the tie blind spot only in the RETIRED rules would have left the
    # identical hole one screen further down. Fix a class, not an instance.
    Rule("subms-no-batch", r"0\.(?:17|33)[\s~]*ms",
         "a sub-millisecond known-cell figure is throughput amortized over a batch, not a per-frame "
         "latency; without the batch it reads as latency next to ffbidx's 4.4 ms",
         "add /hit and the batch size (or name the other quantity, e.g. integration)",
         needs=("integrat", "box-integ", "un-attributed", "unattributed"),
         needs_all=(("/hit", "amortiz", "throughput", "steady"),
                    ("b=32", "batch 32", "b=120", "batch 120"))),
    # The RATIO evades the rule above. 100x was 26 / 0.26, so it is retired with the old B=120 timing;
    # the current 25.7 / 0.17 is ~151x. A batch qualifier no longer makes the old derivation current.
    Rule("ratio-100x", r"100\s*x\s*(?:less|cheaper|fewer|faster)",
         "the ~100x discovery-vs-registration ratio used the retired 0.26 ms B=120 figure; "
         "25.7 / 0.17 is ~151x",
         "say '~151x less cost when batched at B=120'", window=400),
]


# ---- file-level invariants: if the trigger appears, the caveat must appear too ----------------
# Line-level rules cannot express "you may say this only if you also say that". The RTX case is
# exactly that shape: quoting fp32 timings next to a card we have never run on is fine ONLY while
# the page states outright that no number came from one.
#
# ENFORCEMENT COMES IN TWO WIDTHS, and BOTH are needed -- measured, not assumed. `window=0` is the
# original whole-file form: the trigger scopes which files the invariant applies to, and each
# `needed` pattern then has to appear SOMEWHERE in that file. That form catches a value LEAVING the
# file (a renumber, a deleted row, a table dropped in an edit) and nothing finer, which was not
# enough: 0.915 is written at THREE sites in the manuscript (the sec:realdata prose, the S12 note
# and the tab:realmerge row), so editing the row's CC* cell alone left two copies behind and the
# whole-file check green. Injection-tested exactly that way before this note was written.
#
# `window > 0` is the per-occurrence form and closes it: every occurrence of the trigger must carry
# all of `needed` within that many characters. It is used with a trigger that matches the merge ROW
# and only the row -- "Jungfrau-4M lysozyme &" is the tab:realmerge label; tab:realindex writes
# "Jungfrau-4M & lysozyme (Jungfrau-4M) &" and the prose has no "&" at all -- so a single edited
# cell in that row now fires while the three sentences that merely name the dataset stay silent.
# Anchoring on a table cell is a structural dependency and the failure mode is stated rather than
# hidden: reflow the row so the label no longer abuts an "&" and the windowed entry goes QUIET
# (fail-open), which is why the whole-file entry is kept alongside it rather than replaced by it.
#
# The needles are REGEXES over _normalize()d text, not substrings, because the same number is
# written differently in a .tex and in a deck builder -- `$31\%$` against `31%` -- and a substring
# test would force one of the two to be excluded. They interpolate FACTS for the reason
# blind-rate-swap's `instead` does: a hard-coded number here is a second copy of the measurement,
# and second copies drift.
def _lit(v: str) -> str:
    """A FACTS value as a regex matching the number AS WRITTEN, not as a digit-run inside another.

    Same trap (a) the RETIRED patterns document: \\b treats a decimal point as a boundary, so
    `\\b1506\\b` happily matches inside `1506.4`, and a bare `31` matches the tail of `531`.
    """
    return rf"(?<![\d.]){re.escape(v)}(?![\d])"


@dataclass
class Required:
    """If `trigger` appears in a file, every pattern in `needed` must appear in it too."""
    name: str
    trigger: str                       # regex; scopes the invariant to the files it is about
    needed: tuple[str, ...]            # regexes, ALL of which must appear
    why: str
    window: int = 0                    # 0: anywhere in the file. >0: within this many chars of
                                       # EVERY occurrence of the trigger
    flags: int = re.I
    _trx: re.Pattern = field(init=False, repr=False)
    _needles: tuple[re.Pattern, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._trx = re.compile(self.trigger, self.flags)
        self._needles = tuple(re.compile(n, self.flags) for n in self.needed)


REQUIRED = [
    Required("rtx-disclaimer", r"RTX",
             ("no number here was measured on an RTX Blackwell",),
             "this file argues an RTX Blackwell case from datasheet fp32/$, but every GLINT timing "
             "on it is an A100 (or H100) measurement. Without the blanket disclaimer a reader "
             "attributes the fp32 figures to a card we have never benchmarked -- which is exactly "
             "what happened once."),
    # ⚑ The merge-quality FACTS were DECORATIVE until this entry existed (found in review of #154).
    # jungfrau_ccstar / jungfrau_rsplit_pct / jungfrau_iovers were added to the table, commented at
    # length, and read by check_arithmetic's ordering guards -- and by nothing that looks at a
    # deliverable. So the arithmetic knew 0.915 had to sit below xgandalf's 0.930, and no code
    # anywhere connected either number to the row tab:realmerge actually prints. Editing that row
    # to any value at all left every run green. That is the same defect this file documents twice
    # over in FACTS ("a fact nothing reads is a comment"), reproduced in the guard itself.
    #
    # THE TRIGGER IS THE DATASET, NOT THE DETECTOR, and the distinction is measured rather than
    # stylistic: `Jungfrau` alone appears in the DRP projections page (the calib bit-exactness note,
    # "Jungfrau 1M/4M") and four times in build_glint.py (mfx r199, "jungfrau-16M") -- files that
    # have no business carrying a merge table and would be told to grow one. "Jungfrau-4M lysozyme"
    # names the cxil1015922 r0033 dataset and occurs in the manuscript and nowhere else.
    Required("jungfrau-merge-facts", r"Jungfrau-4M\s+lysozyme",
             (_lit(f"{FACTS['jungfrau_ccstar']:.3f}"),
              _lit(f"{FACTS['jungfrau_rsplit_pct']:g}") + r"\s*\\?%",
              _lit(f"{FACTS['jungfrau_iovers']:g}"),
              _lit(f"{FACTS['jungfrau_blind_of1563']:d}"),
              _lit(f"{FACTS['jungfrau_final_of1563']:d}"),
              _lit(f"{FACTS['jungfrau_frames_total']:d}")),
             f"a file that reports the Jungfrau-4M lysozyme dataset must state the RECORD-SOURCED "
             f"numbers it was measured at (job 35507050): CC* {FACTS['jungfrau_ccstar']}, R_split "
             f"{FACTS['jungfrau_rsplit_pct']}%, <I/sigma> {FACTS['jungfrau_iovers']} on the "
             f"{FACTS['jungfrau_final_of1563']}-crystal merge, out of "
             f"{FACTS['jungfrau_blind_of1563']} blind of {FACTS['jungfrau_frames_total']} frames. "
             "If a value here really moved, edit FACTS and re-run every target -- do not edit the "
             "row. THE DENOMINATOR IS THE MEASUREMENT: the retired '93% (support 54/60)' is what a "
             "merge row with no count behind it becomes"),
    # ...and the SAME numbers again at row width, which is the half that catches a single edited
    # cell. See the two-widths note above for why both entries exist and which one fails open.
    Required("jungfrau-merge-row", r"Jungfrau-4M\s+lysozyme\s*&",
             (_lit(f"{FACTS['jungfrau_final_of1563']:d}"),
              _lit(f"{FACTS['jungfrau_ccstar']:.3f}"),
              _lit(f"{FACTS['jungfrau_rsplit_pct']:g}") + r"\s*\\?%",
              _lit(f"{FACTS['jungfrau_iovers']:g}")),
             f"tab:realmerge's Jungfrau-4M row must read "
             f"{FACTS['jungfrau_final_of1563']} crystals / CC* {FACTS['jungfrau_ccstar']} / "
             f"R_split {FACTS['jungfrau_rsplit_pct']}% / <I/sigma> {FACTS['jungfrau_iovers']} -- "
             "the partialator merge of job 35507050's gated subset, at the default 10 "
             "scaling/post-refinement cycles the caption states. It is NOT the glint#129 A/B "
             "protocol (unity scale, native), which reads R_split 33.0 vs 26.7 on this same r0033 "
             "data and is not comparable with this row",
             window=120),
    # cxidb-45, and NOT "Proteinase K": build_pitch.py discusses Proteinase K indexing (the DIALS
    # head-to-head) without ever merging it, so keying on the protein name would demand merge
    # statistics from a deck that correctly does not quote any. `cxidb-45` names the serial set and
    # appears in exactly the two files that do state them -- the manuscript and build_glint.py's
    # "CC* Proteinase K / 0.90" panel. The cover letter quotes CC*=0.90 without naming the dataset,
    # so it is out of scope here by construction; S12's "the CC*=0.90 row is Proteinase K" is the
    # sentence that keeps the two headline CC* values apart, and check_arithmetic's collapse guard
    # is what keeps them from being edited equal.
    Required("pk45-merge-facts", r"cxidb-45",
             (_lit(f"{FACTS['pk45_ccstar']:.2f}"),
              _lit(f"{FACTS['pk45_rsplit_pct']:g}") + r"\s*\\?%",
              _lit(f"{FACTS['pk45_iovers']:g}")),
             f"a file that reports the cxidb-45 Proteinase K merge must state the values it was "
             f"measured at: CC* {FACTS['pk45_ccstar']}, R_split {FACTS['pk45_rsplit_pct']:g}%, "
             f"<I/sigma> {FACTS['pk45_iovers']}. This is the HEADLINE CC* -- the abstract and the "
             "cover letter both lead with it -- and it is a DIFFERENT DATASET from the Jungfrau-4M "
             f"row's {FACTS['jungfrau_ccstar']}, which is why S12 says so out loud"),
    # `Proteinase~K` with the LaTeX tie, so `~?\s*` rather than a plain space: _normalize() leaves
    # an ASCII "~" alone (it only folds the UNICODE approximation signs), and the row is written
    # with the tie in the manuscript and without one anywhere a deck might grow such a table.
    Required("pk45-merge-row", r"cxidb-45\s+Proteinase~?\s*K\s*&",
             (_lit(f"{FACTS['pk45_ccstar']:.2f}"),
              _lit(f"{FACTS['pk45_rsplit_pct']:g}") + r"\s*\\?%",
              _lit(f"{FACTS['pk45_iovers']:g}")),
             f"tab:realmerge's cxidb-45 row must read CC* {FACTS['pk45_ccstar']} / R_split "
             f"{FACTS['pk45_rsplit_pct']:g}% / <I/sigma> {FACTS['pk45_iovers']} -- the 290-frame "
             "partialator merge. The abstract, sec:realdata, the S12 note and the cover letter all "
             "quote this CC*, so a cell edited here silently disagrees with four other sites",
             window=120),
]


def check_required(path: Path, text: str) -> list[str]:
    out = []
    norm = _normalize(text)
    for req in REQUIRED:
        for m in req._trx.finditer(norm):
            seg = (norm if not req.window
                   else norm[max(0, m.start() - req.window): m.end() + req.window])
            missing = [pat for pat, rx in zip(req.needed, req._needles) if not rx.search(seg)]
            if missing:
                where = "" if not req.window else f" within {req.window} chars of it"
                out.append(f"  {path.name}  [REQUIRED/{req.name}] mentions {m.group(0).strip()!r} "
                           f"but does not state {', '.join(repr(x) for x in missing)}{where}\n"
                           f"      why:  {req.why}\n")
            if not req.window:
                break            # whole-file: one report per file, not one per mention
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


def _near_sentence(hay: str, pos: int, words: tuple[str, ...]) -> bool:
    """Whether one of `words` occurs in the sentence or clause containing `pos`."""
    before = hay[:pos]
    start = max(before.rfind("\n\n") + 2,
                max((m.end() for m in re.finditer(r"[.!?;](?:\s|$)", before)), default=0))
    after = re.search(r"[.!?;](?:\s|$)", hay[pos:])
    end = pos + after.start() + 1 if after else len(hay)
    seg = hay[start:end].lower()
    return any(word.lower() in seg for word in words)


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
                if rule.needs_all and all(_near_sentence(norm, m.start(), group)
                                          for group in rule.needs_all):
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
    if F["saturating_batch"] != F["driver_default_batch"]:
        bad.append("  FACTS: deprecated saturating_batch alias must equal driver_default_batch")
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
    # The "binary" control is the SHIPPED weight, w = |q|^-QPOW at QPOW=1.0 -- so tie the fact to
    # the source rather than to a comment about the source. Same defect the M3 block had: a fact
    # nothing reads is a comment, and this one silently defines what "binary" even means here.
    _gf_p = Path(__file__).resolve().parent.parent / "glint" / "glint_fast.py"
    _gf_s = _gf_p.read_text(encoding="utf-8") if _gf_p.exists() else ""
    _qm = re.search(r'QPOW = float\(os\.environ\.get\("QPOW", "([\d.]+)"\)\)', _gf_s)
    if not _gf_s:
        bad.append(f"  FACTS: {_gf_p} unreadable, so pw_qpow_default is unchecked and the peak-"
                   "weight control is defined by nothing")
    elif _qm is None:
        bad.append("  FACTS: the QPOW default could not be located in glint/glint_fast.py -- the "
                   "pw_qpow_default check is dead; fix the pattern rather than dropping it")
    elif float(_qm.group(1)) != float(F["pw_qpow_default"]):
        bad.append(f"  FACTS: pw_qpow_default is {F['pw_qpow_default']} but glint_fast ships "
                   f"{_qm.group(1)}; the 'binary' arm is no longer the shipped weight, so every "
                   "weighting comparison is against a baseline nobody runs")

    # The peak-weight block's headline is "every soft weighting is SIGNIFICANTLY worse", which is a
    # claim about p-values; check them as such, from the stored splits, and require the reconciling
    # margin so the splits and the totals cannot drift apart.
    for _tag, _g, _l, _arm in (("quarter", "pw_quarter_gained", "pw_quarter_lost",
                                "pw_blind_quarter_of480"),
                               ("sqrt", "pw_sqrt_gained", "pw_sqrt_lost", "pw_blind_sqrt_of480"),
                               ("linear", "pw_linear_gained", "pw_linear_lost",
                                "pw_blind_linear_of480"),
                               ("inverse", "pw_inverse_gained", "pw_inverse_lost",
                                "pw_blind_inverse_of480")):
        _m = int(F[_g]) + int(F[_l])
        _p = 1.0 if _m == 0 else min(1.0, 2.0 * sum(comb(_m, _k)
                                                    for _k in range(min(int(F[_g]), int(F[_l])) + 1))
                                     / 2.0 ** _m)
        if _p > 0.05:
            bad.append(f"  FACTS: blind {_tag} vs binary was MEASURED significantly worse; the "
                       f"stored split {F[_g]}/{F[_l]} now gives p = {_p:.3g}, so glint_fast's "
                       "'EVERY soft weighting is significantly worse' no longer holds")
        if int(F[_g]) - int(F[_l]) != int(F[_arm]) - int(F["pw_blind_binary_of480"]):
            bad.append(f"  FACTS: blind {_tag}: discordant split {F[_g]}-{F[_l]} does not reconcile "
                       f"with the totals {F[_arm]}-{F['pw_blind_binary_of480']}")

    # The peak-weight block. Two claims carry it, and both are orderings rather than values, so a
    # later edit cannot flip the conclusion while leaving the table looking plausible.
    if F["pw_blind_binary_of480"] <= max(F["pw_blind_quarter_of480"], F["pw_blind_sqrt_of480"],
                                         F["pw_blind_linear_of480"], F["pw_blind_inverse_of480"]):
        bad.append("  FACTS: binary was MEASURED the best blind arm (282); a table where some "
                   "weighting beats it has inverted the result this block exists to record")
    # The falsifier: LOWER n_eff than sqrt, yet a BETTER rate. That inequality is the whole
    # evidence that direction matters and not just variance.
    if not (F["pw_neff_inverse"] < F["pw_neff_sqrt"]
            and F["pw_blind_inverse_of480"] > F["pw_blind_sqrt_of480"]):
        bad.append("  FACTS: the inverse arm must sit BELOW sqrt in n_eff and ABOVE it in rate "
                   "(0.526 < 0.693, 251 > 200); without that the variance story is unfalsified")
    # The M3 block. It does NOT claim equivalence -- failing to reject is not evidence of no
    # difference -- so what is guarded is what was measured: no arm from 4 to 80 reaches
    # significance in either channel, STEPS=2 does, and the interval bounding any true effect stays
    # where glint_fast quotes it. EVERY arm is checked, both channels: the sweep is non-monotonic,
    # so endpoint checks would leave the interior unguarded while looking thorough.
    def _mcnemar_p(a: int, b: int) -> float:
        m = a + b
        return 1.0 if m == 0 else min(1.0, 2.0 * sum(comb(m, k) for k in range(min(a, b) + 1)) / 2.0 ** m)

    # Tango (1998) score interval for the paired rate difference d = p01 - p10.
    #
    # The first version of this was Clopper-Pearson on the conditional share n01/(n01+n10),
    # rescaled by the OBSERVED discordance rate m/n. That conditions on m and then reports the
    # result as an interval for the marginal difference, which ignores the randomness in m.
    # Measured coverage: 93.6% at the (10,29)-matching probabilities, and 59.3% at (2,0) -- and
    # (0,0) came out as the zero-width [0, 0], i.e. certainty from no information. It read as an
    # exact interval and was not one.
    #
    # Tango's score interval inverts the score test for d, using the constrained MLE of p10 under
    # p01 - p10 = d. Measured coverage on the same configurations: 95.0%, 98.3%, and (0,0) gives
    # [-0.79%, +0.79%]. experiments/test_check_numbers_ci.py pins both the closed-form MLE and the
    # coverage.
    # --- BEGIN paired-difference interval (lifted verbatim by experiments/test_check_numbers_ci.py) ---
    def _zq(p: float) -> float:
        """Phi^-1 by bisection on math.erf -- no third-party import, and the only quantile needed."""
        lo, hi = 0.0, 10.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p else (lo, mid)
        return (lo + hi) / 2

    _Z975 = 1.959963984540054

    def _p10_mle(n01: int, n10: int, n: int, d: float) -> float:
        """Constrained MLE of p10 given p01 - p10 = d. Maximising
             L(p) = n01*ln(p+d) + n10*ln(p) + r*ln(1-2p-d),  r = n - n01 - n10
        and clearing denominators gives 2n*p^2 - [n01+n10 - d*(n01+3*n10+2r)]*p - n10*d*(1-d) = 0;
        the positive root is the MLE. (Checked against brute-force maximisation, agrees to 9e-7.)"""
        r = n - n01 - n10
        B = n01 + n10 - d * (n01 + 3 * n10 + 2 * r)
        C = -n10 * d * (1 - d)
        p = (B + max(B * B - 8 * n * C, 0.0) ** 0.5) / (4 * n)
        return min(max(p, max(0.0, -d)), (1.0 - d) / 2)

    def _score(n01: int, n10: int, n: int, d: float) -> float:
        p10 = _p10_mle(n01, n10, n, d)
        var = n * (2 * p10 + d * (1 - d))
        if var <= 0:
            k = n01 - n10 - n * d
            return 0.0 if k == 0 else (1e18 if k > 0 else -1e18)
        return (n01 - n10 - n * d) / var ** 0.5

    def _ci_diff(n01: int, n10: int, n: int, z: float = _Z975):
        """95% CI on the RATE difference (arm - default) = {d : |Z(d)| <= z}.

        This is what turns "not significant" into a statement with a SIZE attached, which is the
        point: the block quotes an interval instead of claiming a parameter is inert. Z is
        decreasing in d, so each end is a single bisection.
        """
        point = (n01 - n10) / n
        a, b = -1.0 + 1e-12, point
        for _ in range(60):
            mid = (a + b) / 2
            a, b = (mid, b) if _score(n01, n10, n, mid) > z else (a, mid)
        low = (a + b) / 2
        a, b = point, 1.0 - 1e-12
        for _ in range(60):
            mid = (a + b) / 2
            a, b = (a, mid) if _score(n01, n10, n, mid) < -z else (mid, b)
        return low, (a + b) / 2

    # --- END paired-difference interval ---

    # The table must be COMPLETE before anything is concluded from it: deleting an entry would
    # otherwise quietly narrow "every arm, both channels" to "the arms that are left".
    _M3_ARMS = (2, 3, 4, 5, 6, 10, 12, 16, 24, 32, 48, 80)
    _want_keys = {(a, c) for a in _M3_ARMS for c in ("blind", "hybrid")}
    if set(M3_SPLITS) != _want_keys:
        _missing = sorted(_want_keys - set(M3_SPLITS)); _extra = sorted(set(M3_SPLITS) - _want_keys)
        bad.append(f"  FACTS: M3_SPLITS is not the full sweep -- missing {_missing}, unexpected "
                   f"{_extra}. The block claims every arm in both channels; it can only claim what "
                   "is in the table")

    _N480 = 480
    _bounds = {"blind": [0.0, 0.0], "hybrid": [0.0, 0.0]}     # pointwise, per arm
    _sim = {"blind": [0.0, 0.0], "hybrid": [0.0, 0.0]}        # simultaneous over all 24
    _Z_SIM = _zq(1 - 0.05 / (2 * 24))
    for (_arm, _ch), (_g, _l) in sorted(M3_SPLITS.items()):
        _p = _mcnemar_p(_g, _l)
        if _arm == 2 and _p > 0.05:
            bad.append(f"  FACTS: M3 STEPS=2/{_ch} was MEASURED significantly worse; the stored "
                       f"split {_g}/{_l} now gives p = {_p:.3g}, so glint_fast's 'the only arm that "
                       "differs is the shortest, and it is worse' no longer holds")
        # ...and WORSE, which the p-value alone cannot say: McNemar is symmetric, so swapping the
        # split to (54, 29) leaves p = 0.008 untouched while turning STEPS=2 into significantly
        # BETTER than the default. The direction is half the claim, so check it explicitly.
        if _arm == 2 and _g >= _l:
            bad.append(f"  FACTS: M3 STEPS=2/{_ch} split {_g}/{_l} says the SHORT arm gained at "
                       "least as many frames as it lost, i.e. fewer steps are as good or better. "
                       "That inverts the block, and p cannot catch it -- McNemar is symmetric")
        # EVERY arm but 2, not just >= 4. Arm 3 fell through both branches and was unguarded, so
        # the "only significant arm is 2" invariant did not actually cover the arm most likely to
        # move next -- it is the one adjacent to the significant one.
        if _arm != 2 and _p <= 0.05:
            bad.append(f"  FACTS: M3 STEPS={_arm}/{_ch} now reaches significance (split {_g}/{_l}, "
                       f"p = {_p:.3g}). The block says STEPS=2 is the only arm that differs -- that "
                       "IS the result, so rewrite the claim rather than the table")
        if _arm >= 4:
            _lo, _hi = _ci_diff(_g, _l, _N480)
            _bounds[_ch][0] = min(_bounds[_ch][0], _lo)
            _bounds[_ch][1] = max(_bounds[_ch][1], _hi)
            # ...and the SIMULTANEOUS version. glint_fast's sentence is about ANY arm, which is a
            # familywise claim; the min/max of pointwise 95% intervals is not one, however natural
            # it looks. Bonferroni over the 24 arm x channel comparisons the block presents.
            _slo, _shi = _ci_diff(_g, _l, _N480, _Z_SIM)
            _sim[_ch][0] = min(_sim[_ch][0], _slo)
            _sim[_ch][1] = max(_sim[_ch][1], _shi)
    # ...and the SIZE the data still admit, which is the honest version of "no difference".
    # glint_fast quotes this interval verbatim; if it widens, that sentence is wrong.
    # ...and the SIZE the data still admit, PER CHANNEL and at the bound glint_fast actually
    # prints. A single merged extremum let the hybrid claim drift up to the blind allowance, and a
    # slack threshold (-4.5/+6.0) permitted numbers the prose does not support. These are the
    # displayed values with one rounding step of slack, no more.
    # The headline totals are DERIVED, not independent: arm_total = default_total + gained - lost.
    # They were added as flat facts and then orphaned when the splits moved into M3_SPLITS, which
    # is the failure this file documents twice over -- a fact nothing reads is a comment, and here
    # the comment was a number the prose quotes. Tie each one to the split it came from.
    for _ch, _arm, _key in (("blind", 2, "m3_blind_steps2_of480"),
                            ("blind", 4, "m3_blind_steps4_of480"),
                            ("blind", 80, "m3_blind_steps80_of480"),
                            ("hybrid", 2, "m3_hybrid_steps2_of480"),
                            ("hybrid", 32, "m3_hybrid_steps32_of480")):
        _base = F["m3_blind_steps8_of480"] if _ch == "blind" else F["m3_hybrid_steps8_of480"]
        _g, _l = M3_SPLITS[(_arm, _ch)]
        if int(F[_key]) != int(_base) + _g - _l:
            bad.append(f"  FACTS: {_key} = {F[_key]} but the {_ch} STEPS={_arm} split {_g}/{_l} "
                       f"against a default of {_base} gives {int(_base) + _g - _l}. One of the two "
                       "was edited without the other")

    for _ch, _tab, _kind, (_want_lo, _want_hi) in (
            ("blind",  _bounds, "per-arm 95%",      (-0.040, 0.054)),
            ("hybrid", _bounds, "per-arm 95%",      (-0.037, 0.034)),
            ("blind",  _sim,    "simultaneous 95%", (-0.058, 0.074)),
            ("hybrid", _sim,    "simultaneous 95%", (-0.053, 0.050))):
        _lo, _hi = _tab[_ch]
        if _lo < _want_lo - 0.0005 or _hi > _want_hi + 0.0005:
            bad.append(f"  FACTS: across 4..80 the {_ch} {_kind} interval on the rate difference "
                       f"now spans [{100*_lo:+.2f}%, {100*_hi:+.2f}%], outside the "
                       f"[{100*_want_lo:+.1f}%, {100*_want_hi:+.1f}%] glint_fast states")

    # m3_steps_default must BE the shipped default, not a description of it -- a fact nothing reads
    # is a comment (cf. the glint_blind_rate_pct drift above). Read it out of the source.
    _gf_path = Path(__file__).resolve().parent.parent / "glint" / "glint_fast.py"
    _gf = _gf_path.read_text(encoding="utf-8") if _gf_path.exists() else ""
    _m = re.search(r'STEPS = int\(os\.environ\.get\("STEPS", "(\d+)"\)\)', _gf)
    if not _gf:
        bad.append(f"  FACTS: {_gf_path} unreadable, so m3_steps_default is unchecked and the "
                   "whole M3 block is stated against a baseline nothing verifies")
    elif _m is None:
        bad.append("  FACTS: the STEPS default could not be located in glint/glint_fast.py, so "
                   "the m3_steps_default check is dead -- fix the pattern, do not drop the check")
    elif int(_m.group(1)) != int(F["m3_steps_default"]):
        bad.append(f"  FACTS: m3_steps_default is {F['m3_steps_default']} but glint_fast ships "
                   f"{_m.group(1)}; every M3 comparison is against a baseline that is no longer "
                   "the default, so the block describes a setting nobody runs")
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
                               ("xgandalf_blind_rate_pct", "xgandalf_blind_strict_of120"),
                               # the correct-lattice pair, same denominator, same rule
                               ("glint1_lattice_rate_pct", "glint1_lattice_of120"),
                               ("xgandalf_lattice_rate_pct", "xgandalf_lattice_of120")):
        _want = round(100.0 * int(F[_cnt_key]) / 120.0)
        if int(F[_pct_key]) != _want:
            bad.append(f"  FACTS: {_pct_key} = {F[_pct_key]}% but {_cnt_key} = {F[_cnt_key]}/120 rounds to "
                       f"{_want}% -- a count and its percentage were edited apart")
    # The correct-lattice bar is strictly LOOSER than the strict bar (it drops the coverage
    # requirement), so a count below its own strict count is an edit that crossed two rows.
    for _lat, _strict, _who in (("glint1_lattice_of120", "glint1_strict_of120", "GLINT-(1)"),
                                ("xgandalf_lattice_of120", "xgandalf_blind_strict_of120", "xgandalf")):
        if int(F[_lat]) < int(F[_strict]):
            bad.append(f"  FACTS: {_who} correct-lattice {F[_lat]}/120 is BELOW its strict "
                       f"{F[_strict]}/120 -- the looser bar cannot pass fewer frames")
    # Same derivation for the n=480 rows. Note the denominator differs, so this cannot be folded into
    # the loop above -- and folding it would be the exact mistake that makes a percentage stop tracking
    # its count.
    for _pct_key, _cnt_key in (("glint_blind_rate_pct_480", "glint1_strict_of480"),
                               ("xgandalf_blind_rate_pct_480", "xgandalf_blind_strict_of480")):
        _want = round(100.0 * int(F[_cnt_key]) / 480.0)
        if int(F[_pct_key]) != _want:
            bad.append(f"  FACTS: {_pct_key} = {F[_pct_key]}% but {_cnt_key} = {F[_cnt_key]}/480 rounds to "
                       f"{_want}% -- a count and its percentage were edited apart")
    # The Jungfrau-4M block. RECORD-SOURCED (job 35507050), so nothing here re-derives a measurement
    # -- what it enforces is that the three counts keep telling ONE story and that the paragraph's
    # mixed-direction comparison cannot be flattened into a win.
    #
    # (1) The nesting that makes 1482 "the gated subset of 1506". If a later edit lifts the final
    # count above the blind one, tab:realindex's "96% blind; 95% final" stops describing a subset and
    # starts describing two unrelated runs -- which is how the retired 93% got its missing
    # denominator in the first place.
    if not (int(F["jungfrau_final_of1563"]) <= int(F["jungfrau_blind_of1563"])
            <= int(F["jungfrau_frames_total"])):
        bad.append(f"  FACTS: the Jungfrau counts must nest -- final "
                   f"{F['jungfrau_final_of1563']} <= blind {F['jungfrau_blind_of1563']} <= frames "
                   f"{F['jungfrau_frames_total']}. tab:realindex prints the final row as the GATED "
                   "SUBSET of the blind one; if that stops holding, rewrite the row rather than "
                   "renumbering it")
    # (2) The percentages tab:realindex actually prints, re-derived from the counts. Same reason the
    # 120 and 480 pairs are derived above: a percentage that nothing computes is a comment, and the
    # retired 93% is precisely a percentage nobody could tie back to a count.
    for _pct_key, _cnt_key in (("jungfrau_blind_rate_pct", "jungfrau_blind_of1563"),
                               ("jungfrau_final_rate_pct", "jungfrau_final_of1563")):
        _want = round(100.0 * int(F[_cnt_key]) / int(F["jungfrau_frames_total"]))
        if int(F[_pct_key]) != _want:
            bad.append(f"  FACTS: {_pct_key} = {F[_pct_key]}% but {_cnt_key} = {F[_cnt_key]}/"
                       f"{F['jungfrau_frames_total']} rounds to {_want}% -- a count and its "
                       "percentage were edited apart")
    # (3) THE MIXED DIRECTION, pinned on BOTH sides. sec:realdata's Jungfrau paragraph ends on
    # "the two summary statistics show small differences in opposite directions: XGANDALF gives the
    # higher CC*, whereas GLINT gives the lower R_split". Each half is a separate claim and each can
    # invert on its own, so neither is left to the other's check. Same shape as the streaming
    # crossover guards below, and for the same reason: a single careless re-measure can flip one
    # relation and not its neighbour, leaving a sentence that reads as if both still held.
    if not float(F["jungfrau_ccstar"]) < float(F["jungfrau_xg_ccstar"]):
        bad.append(f"  FACTS: GLINT's Jungfrau CC* ({F['jungfrau_ccstar']}) no longer sits BELOW "
                   f"xgandalf's ({F['jungfrau_xg_ccstar']}) -- sec:realdata says outright that "
                   "XGANDALF gives the higher CC*. Rewrite that passage, do not renumber it")
    if not float(F["jungfrau_rsplit_pct"]) < float(F["jungfrau_xg_rsplit_pct"]):
        bad.append(f"  FACTS: GLINT's Jungfrau R_split ({F['jungfrau_rsplit_pct']}%) no longer sits "
                   f"BELOW xgandalf's ({F['jungfrau_xg_rsplit_pct']}%) -- with the CC* relation "
                   "above, that is what makes the comparison MIXED rather than a win. Rewrite the "
                   "passage, do not renumber it")
    # (4) The two CC* values the paper prints are DIFFERENT DATASETS, and S12 exists to say so. If
    # they are ever edited equal, that sentence becomes unreadable and the headline 0.90 silently
    # acquires the Jungfrau row's provenance -- the exact conflation S12 was written to stop.
    if float(F["pk45_ccstar"]) == float(F["jungfrau_ccstar"]):
        bad.append(f"  FACTS: pk45_ccstar and jungfrau_ccstar are both {F['pk45_ccstar']} -- S12 "
                   "distinguishes them by value ('the CC*=0.90 row is Proteinase K; the Jungfrau-4M "
                   "row ... sits at CC*=0.915'). Two datasets have been collapsed into one number")
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
