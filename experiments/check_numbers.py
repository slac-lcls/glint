#!/usr/bin/env python3
"""Guard the GLINT deliverables against number drift.

Every problem the 2026-07-19 audit found was drift BETWEEN deliverables that each looked fine on its
own: `34 ms` was corrected by hand in the paper and the derived quantities (frames/s, speedup ratios)
were left stranded at the old value in four other files. This turns that class of error into a failing
script.

It checks four things, in rough order of how much damage each does:

  1. RETIRED values      -- a number we have superseded, appearing anywhere. The 15 ms / 21 ms /
                            47 f/s / 160x / 76% family. Each carries what it should say instead.
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
    "blind_ms":            34.0,    # blind, per frame, one A100
    "blind_fps":           29.0,    # = 1000/blind_ms
    "known_perframe_ms":   16.5,    # per-frame known-cell rescue (replica_gpu.index_known_gpu_cell)
    "graph_ms":            1.46,    # batched + CUDA graph                                    (#14)
    "fused_b32_ms":        0.45,    # fused kernels, fp64, batch 32                           (#16)
    "fused_b120_ms":       0.26,    # fused kernels, fp64, batch 120                          (#16)
    "fused_b32_fp32_ms":   0.33,    # fp32 INDEXING at batch 32                               (#15/#16)
    "fused_b120_fp32_ms":  0.16,    # fp32 INDEXING at batch 120                              (#15/#16)
    "fused_fps":           3800.0,  # = 1000/fused_b120_ms
    "saturating_batch":    64,      # one block per frame; 108 SMs on an A100
    "indexing_rate":       "75/114",
    "blind_rate_pct":      71,      # NOT 76 -- the spurious-limited ceiling
    # integration ----------------------------------------------------------------------------------
    "integ_before_ms":     585.0,   # 16 Mpix / 800 reflections, whole-frame float64 upcast
    "integ_after_ms":      7.6,     # upcast removed, bit-identical                           (#17)
    "integ_speedup":       76.8,    # = integ_before_ms / integ_after_ms
    "integ_fused_ms":      0.33,    # fused GPU box-integration, per frame                    (#18)
    "h2d_16mpix_ms":       4.0,     # ~3-5 ms; why the fused kernel only pays device-resident
    # streaming (NOT merged) -----------------------------------------------------------------------
    "stream_ms":           5.58,    # end-to-end per frame, B=40                       (#19, open)
    "stream_fps":          179.0,   # = 1000/stream_ms                                 (#19, open)
    "peakfind_ms":         2.38,    # largest single stage, already on GPU             (#19, open)
    "fused_share_ms":      0.89,    # index+integrate share of the 5.58
    # source / sizing ------------------------------------------------------------------------------
    "rep_rate_hz":         35000.0,
    "hit_rate":            0.10,
    "hits_per_s":          3500.0,  # = rep_rate_hz * hit_rate
    "gpus_at_10pct":       1.0,     # = rep_rate_hz * hit_rate * fused_b120_ms/1000
    "xgandalf_blind_ms":   11542.0,
    "xgandalf_speedup":    340.0,   # = xgandalf_blind_ms / blind_ms
    "ffbidx_latency_ms":   4.4,     # per single call -- a LATENCY
    "ffbidx_pipelined_ms": 3.1,     # persistent indexer -- the THROUGHPUT comparator
    "ffbidx_speedup":      12.0,    # = ffbidx_pipelined_ms / fused_b120_ms (throughput vs throughput)
}

DEFAULT_TARGETS = [
    HOME / "git/papers/glint/glint.tex",
    HOME / "git/slides/glint/build_glint.py",
    HOME / "git/slides/glint/build_pitch.py",
    HOME / "git/slides/drp/build_drp.py",
    HOME / "git/slides/fftindex/build.py",
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
    Rule("blind-15ms", r"\b15\s*ms\b",
         "blind was corrected to 34 ms/frame; 15 ms is the pre-sweep value", "34 ms"),
    Rule("blind-21ms", r"\b21(?:\.3)?\s*ms\b",
         "21.3 ms was the M2/M4 milestone, superseded by 34 ms end-to-end", "34 ms"),
    Rule("fps-47", r"\b47\b(?=[^\n]{0,60}(?:frames?\s*/\s*s|f/s|shots?/s|throughput))",
         "47 f/s is the reciprocal of the old 21.3 ms; 34 ms gives 29 f/s", "29"),
    Rule("speedup-160", r"[~≈]?\s*160\s*(?:×|x|\\times)",
         "the scalar->GPU blind ratio follows 2342/34, not 2342/15", "~69x"),
    Rule("blind-rate-76", r"\b76\s*%",
         "the blind ceiling is 71% (spurious-limited); nothing supports 76%", "71%"),
    Rule("fused-pred-2.4", r"2\.4\s*(?:→|->|-->)\s*0\.45",
         "the fused kernel replaced the 1.46 ms CUDA-graph path, not a 2.4 ms one; "
         "2.4 inflates the gain from 3.1x to an implied 5.3x", "1.46 -> 0.45"),
    Rule("legacy-shots", r"\b892\b",
         "892 shots/s is the LEGACY FFT-volume micro-bench, not the current pipeline",
         "mark LEGACY, or use ~29 shots/s", exempt=("legacy",)),
]

# ---- 2. overclaims ---------------------------------------------------------------------------
OVERCLAIM = [
    Rule("live-merge", r"(?:stops? when the data are complete|live merge|real[- ]time merg)",
         "live completeness comes from the running merge accumulator: 179 f/s vs ~3500 needed, "
         "~20x short, and unmerged (#19)",
         "scope the claim to INDEXING, or mark the driver as in review"),
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
]


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
    bad: list[str] = []

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
    close("gpus_at_10pct = rep*hit*t_index", float(F["gpus_at_10pct"]),
          float(F["rep_rate_hz"]) * float(F["hit_rate"]) * float(F["fused_b120_ms"]) / 1000.0, tol=0.12)
    close("ffbidx_speedup = pipelined/fused (throughput:throughput)", float(F["ffbidx_speedup"]),
          float(F["ffbidx_pipelined_ms"]) / float(F["fused_b120_ms"]), tol=0.05)

    # the claim that motivates the whole live-merge caveat
    if float(F["stream_fps"]) >= float(F["hits_per_s"]):
        bad.append("  FACTS: stream_fps now meets hits_per_s -- the 'not a live merge' caveat in the "
                   "deliverables is stale and must be revisited, not just this table")
    # the fused kernels being a minority of the streaming budget is why peakfind is the wall
    if float(F["fused_share_ms"]) >= float(F["stream_ms"]) / 2:
        bad.append("  FACTS: fused kernels are no longer a minority of stream_ms -- the 'peakfind is "
                   "the wall' framing needs rechecking")
    return bad


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

    fails = check_arithmetic()
    if fails:
        print("ARITHMETIC (the facts table contradicts itself):")
        print("\n".join(fails) + "\n")

    checked = skipped = 0
    for path in paths:
        text = _read(path)
        if text is None:
            print(f"  SKIP {path} (missing, or pdftotext unavailable)")
            skipped += 1
            continue
        checked += 1
        found = scan(path, text)
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
