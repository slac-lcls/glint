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
    "glint_blind_rate_pct":    77,  # GLINT-(1) blind, 92/120, paper tab:summary
    "xgandalf_blind_rate_pct": 72,  # xgandalf blind, 86/120, same table and same gate -- DIFFERENT indexers
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
    # 1.11 WAS WRONG TWICE OVER, and the first way has nothing to do with #68: it does not reproduce
    # even on the pre-#68 arm, which measures 0.42. Per the profiling in the glint-predict-stage-cost
    # note, 1.11 came from two stacked benchmark artifacts -- tol=0.006 (the predict() SIGNATURE
    # default) instead of the driver's tol=0.002, plus a bench that rebuilt the `panels` dict INSIDE
    # the timed call (+0.448 ms). Production rebuilds neither: StreamDriver.__init__ builds panels
    # once (stream_driver.py:350) and passes the same object every frame. #68 (merged) then fused the
    # detector projection into the gate kernel and took the corrected 0.42 down to 0.16.
    #
    # An independent in-situ cross-check (whole-driver instrumented run, 800 frames, same two arms)
    # agrees on the DELTA even though its absolute scale runs ~8% high because it times stages inside
    # the live pipeline rather than in a loop: predict 0.483 -> 0.207, and the end-to-end wall moved
    # 4.12 -> 3.84, i.e. the wall dropped by 0.28 against a predict saving of 0.28.
    "stream_ms":           4.16,    # steady state per frame, B=40      (#68; was 5.58, then 4.42)
    "stream_fps":          240.0,   # = 1000/stream_ms                  (#68; was 179, then 226)
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
    # DELIBERATELY NOT ACTED ON. stream_ms stays 4.16 and unattributed_ms stays 0.69. Correcting the
    # wall moves several numbers in GLINT's FAVOUR -- stream_fps 240->286, peakfind share 28%->33%,
    # the gap to ~3500 hits/s 15x->12x, the FPGA ceiling 1.39x->1.50x -- which is exactly when to be
    # slowest, and the provenance of 4.16 (which GPU, which protocol) is not recoverable from the
    # code. Confirm on the hardware the original used, then swap both values together.
    "unattributed_ms":     0.69,    # = stream_ms - sum(measured stages)          (open, 2026-08-01)
    # streaming vs offline YIELD -- success fraction, NOT throughput -------------------------------
    # The project's only real-data streaming-vs-offline head-to-head, promoted out of f61a4cf's commit
    # body where it was the sole record. Same 120-frame real cxidb set, same strict research gate
    # (same_lattice AND >=25% of spots AND >=10 refl) as the paper's tab:summary.
    #
    # Denominator is 120 pushed frames in ALL THREE, and the names say so on purpose: `indexing_rate`
    # above is a (strict, loose) PAIR over 120, not a ratio, and reading it as 75/114=66% is the exact
    # misreading these names exist to prevent.
    #
    # INDEX-ONLY. The q-vector dataset cannot exercise the integrate path (test_inlier_frac_gate.py),
    # so these are indexed counts, not integrated-and-merged ones.
    #
    # The gap is structural, not noise: streaming commits its cell from ~5-6 warm-up frames, while the
    # offline reference votes across all 120. warmup_rescue recovers the warm-up frames themselves
    # (5/5) but not the consequences of the early lock.
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
    Rule("blind-rate-swap", r"GLINT[^\n]{0,40}\b71\s*\\?%",
         "71% is XGANDALF's blind rate; GLINT-(1) blind is 77% (92/120, paper tab:summary). Attributing "
         "71% to GLINT understates it and confuses two indexers measured at the same gate", "77%"),
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
         "the end-to-end gap to ~3500 hits/s is 3500/240 = ~15x, not ~20x, now that streaming is "
         "4.16 ms/frame", "~15x"),
    # --- the streaming wall again, superseded 2026-08-01 by the corrected predict (#68) ---
    # 4.42 and 226 were correct for their own measurement; what moved is predict, by 0.26 ms/frame.
    Rule("stream-4.42", r"(?<![\d.])4\.42\s*ms",
         "4.42 ms/frame carried predict at 1.11 ms, which never reproduced (the pre-#68 arm measures "
         "0.42 and #68 took it to 0.16). The wall is 4.16 ms/frame", "4.16 ms"),
    # The exempt is load-bearing, not defensive: glint.tex quotes "4.4 ms / 226 frames/s" for a SINGLE
    # ffbidx call (1000/4.4 = 227). That is a different quantity that happens to round to the same
    # number as the retired streaming figure, and without the exempt this rule sends an editor to
    # "correct" a line that is right.
    Rule("stream-226", r"(?<![\d.])226\b(?=[^\n]{0,60}(?:frames?\s*/\s*s|f/s|fps))",
         "226 f/s is the reciprocal of the retired 4.42 ms", "240",
         exempt=("ffbidx", "single call", "single} call")),
    Rule("live-gap-16x", r"[~≈]?\s*16\s*(?:×|x|\\times)(?=[^\n]{0,40}(?:gap|short|hits))",
         "the gap to ~3500 hits/s follows the current 240 f/s: 3500/240 = ~15x", "~15x"),
    # --- the COLD-WARMUP stage block, superseded the same day it was written ---
    # These four were published for a few hours between the under-warmed measurement and the warmed
    # re-measure. They are listed because they reached three deliverables, not because they lasted.
    Rule("cold-peakfind-1.58", r"(?<![\d.])1\.58\s*ms",
         "1.58 ms came from a stage benchmark warmed only ONCE; properly warmed peakfind is 1.16 ms",
         "1.16 ms"),
    Rule("cold-predict-1.34", r"(?<![\d.])1\.34\s*ms",
         "1.34 ms was the cold-warmup predict figure. Do NOT replace it with the 1.11 this rule used "
         "to recommend -- that was wrong too (tol=0.006 plus a bench rebuilding `panels` inside the "
         "timed call). Measured predict is 0.16 ms after #68, 0.42 before it", "0.16 ms"),
    # --- predict itself, retired 2026-08-01. Two independent errors, so two things to check when
    #     this fires: the VALUE is wrong, and anything derived from it (the margin, the tie, the
    #     "next lever" ordering) is wrong with it.
    Rule("predict-1.11", r"(?<![\d.])1\.11\s*ms",
         "1.11 ms for predict never reproduced on ANY arm: the pre-#68 code measures 0.42 in the same "
         "harness that reproduces peakfind/integrate/h2d to within 7%. It came from tol=0.006 (the "
         "predict() signature default, not the driver's 0.002) stacked on a benchmark that rebuilt "
         "the `panels` dict inside the timed call. #68 then took the corrected 0.42 to 0.16",
         "0.16 ms"),
    Rule("predict-0.42-stale", r"(?<![\d.])0\.42\s*ms(?=[^\n]{0,60}predict)",
         "0.42 ms is the CORRECTED pre-#68 predict, superseded by #68 (merged), which fuses the "
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
