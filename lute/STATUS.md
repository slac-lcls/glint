# GLINT in LUTE — what works, what is measured, what is not

Written 2026-08-05. The point of this file is that the state of this integration should live next to
the code, not in a chat log or one person's head. Every number below has a named source; anything
without one is marked as an assumption.

**Bottom line: all seven are closed** (item 6 as a measurement with a reframing — see its section).
The raw-xtc route runs on real data and produces a correct cell; with `--integrate` it emits a
stream partialator can merge; with `--gpu-calib` it calibrates on the device for 4.1x end-to-end
wall and a byte-identical stream; it runs under an ana release that can see Jungfrau; it checks its
own geometry provenance at startup instead of failing silently; and CI now runs the CPU-only test
layer on every push (glint#107). Item 6's answer, measured on raw Jungfrau **16M** images
(`mfx101555026` r0013) against the beamline's own event-mapped hit list: **the calibrated object is
the pair (`threshold`, `min_snr`), not `min_snr` alone** — the floor-less Epix10ka2M ladder below is
not comparable to any practitioner setting. That ladder ran with the detector's interior ASIC seams
live, and the 2026-08-25 re-measurement (glint#139) found **94.6% of its peaks sitting on a seam**:
masked, precision saturates at `min_snr` 10 and the corrected pair is (110, 8). `PF8_MIN_SNR` stays
at 15 until `asic_seam_mask` is wired, because the two knobs only move together — see item 6.

---

## The three frame sources

`IndexGLINTParameters` (`lute/glint_index.py`) takes exactly one of:

| source | field | destination | status |
|---|---|---|---|
| PeakFinderSFX peaks | `peaks` | `glint.glint_cli` | unchanged, pre-existing |
| CXI images | `images` | `glint.glint_cli` | unchanged, pre-existing |
| **raw xtc** | `exp` + `run` | `experiments/xtc_bridge/glint_xtc.py` | **new, this branch** |

`glint_launch.sh` routes on `--exp` and *whitelists* flags per destination rather than forwarding
them, because `glint_xtc.py`'s argparse rejects unknown flags and several `glint_cli` options carry
non-empty defaults that LUTE emits on every run. Dropped flags are reported on stderr.

---

## Measured on real data

All on `mfxx49820` r0016 (Epix10ka2M, 16x352x384), btx's refined `r0016.geom`, scored against btx's
own peakfinder8 hit list. Jobs 34218485 / 34220944 / 34223329, S3DF ampere.

**Peakfinder agreement**, 6000 events, "of btx-idx" = share of the 1490 frames btx went on to index:

| finder | frames | vs btx hits | of btx-idx | pickup on non-hits | pk/frame |
|---|---|---|---|---|---|
| v4 (default) | 2228 | 96.3% | 99.9% | 1.7% | 35 |
| pf8 whole-detector | 2237 | 96.2% | **100.0%** | 2.0% | 31 |
| pf8 per-panel | 2281 | 96.8% | 99.9% | 2.7% | 31 |

The full-image row reproduced byte-identically across two jobs on two nodes. pf8 and pf8-panel
differ by about one frame, so the seam-masked whole-detector finder is real but earns little on this
detector — recorded as a negative result.

**Per-event time budget**, 120 events, job 34235469:

| stage | median ms | share |
|---|---|---|
| read (`ds.events()` + smd decode) | 0.94 | 0.6% |
| `det.raw` (uint16, 4.33 MB) | 0.86 | 0.6% |
| **`det.calib`** (float32, 8.65 MB) | **145.82** | **97.7%** |
| host->device upload | 1.41 | 0.9% |
| GPU peakfind-equivalent | 0.17 | 0.1% |
| total | 149.21 | |

---

## What has actually been EXECUTED, as opposed to read

This distinction is the reason this file exists. Until 2026-08-05 every measurement on this route
called `run_to_qframes_psana1` directly, from a process whose `PYTHONPATH` already carried the repo
root — i.e. the *reader* was exercised and the *entry point* never was. Running
`glint_launch.sh` for the first time found two defects that reading had missed (a fatal
`ModuleNotFoundError` at the first `import glint`, and every peak-finder threshold being unreachable
from `IndexGLINTParameters`). Both are fixed; both are the kind of defect only execution finds.

| component | status |
|---|---|
| `glint_launch.sh` xtc routing + flag whitelist | **run** — S3DF job 34240308 |
| `glint_xtc.py` read + peak-find + index + write | **run** — same job |
| `IndexGLINTParameters` validators (`_one_source`, per-source `peakfinder`) | **tested** — `lute/test_glint_index.py`, 44 pass; still no CI (item 1) |
| the `peaks` and `images` routes | untouched by this branch; not re-run |
| psana2 / `--psana 2` over envbridge | **never run on this branch** |
| MPI sharding (`glint_xtc_mpi.py`) | **never run on this branch** |

**The one end-to-end run, job 34240308** — `glint_launch.sh --exp mfxx49820 --run 16 --det
MfxEndstation.0:Epix10ka2M.0 --zdist 0.102973 --wavelength 1.290757 --max-events 200 --min-peaks 6
--peakfinder v4 --geom <btx r0016.geom> --calib-dir <private> --top-peaks 100 -o lute_e2e.stream`:

- `--top-peaks` correctly reported as dropped on stderr (it is a `glint_cli`-only flag)
- 200 events -> 54 frames with >= 6 peaks -> **50 indexed (92%)**: blind, +5 N-best recovered,
  +30 rescued
- cell `38.3 / 79.2 / 79.9 A`, angles 89.2 / 89.7 / 90.0, against a truth of `38.4 / 79.3 / 79.5`
- stream has 54 chunks and 50 indexed crystals

**Correction, 2026-08-06.** This bullet used to read "stream is valid CrystFEL 2.3 ... and 50
`indexed_by = glint`". It was neither: CrystFEL 0.10.2 could not open that file at all, and
`indexed_by = glint` was one of the three reasons. Nobody had run a CrystFEL binary against the
orientation-only writer — a round trip through GLINT's own reader passes either way. Fixed in
`ba54639`; `experiments/test_stream_crystfel.py` now runs `process_hkl` itself. Measured before and
after on the same 3000-frame stream: **0 -> 14,514 hkl lines**. The `--integrate` writer
(`predict.write_stream_integrated`, item 3) was always correct and is unaffected.

Two things that run also shows. The stream reports `lattice_type = triclinic, centering = P`
because GLINT imposes no symmetry — the true lysozyme is tetragonal *P*4<sub>3</sub>2<sub>1</sub>2,
so symmetry has to be supplied downstream. And the reflection rows carry the placeholder
`I`/`sigma(I)`/`fs`/`ss` described in #3 below; this is what a merge would have to be built on.

---

## The seven things that stand between this and production

**Progress: all 7 struck through.** A struck-through item carries the commit that closed it and how
it was verified. Item 1's CI half landed after its tests did (glint#107). Item 6 took three passes:
2026-08-06 against an indexing reference on a run the beamline kept, 2026-08-13 on raw Jungfrau 16M
images with the ADU floor (the pass that closed it), and a 2026-08-25 re-measurement (glint#139) that
leaves the item closed but **replaces the operating point it recommended** — read its section to the
end before taking a number out of it.

(This paragraph previously said "6 of 7 ... item 6 is the one still open", contradicting both the
header and the "What is left" footer, which have said all seven since 2026-08-13.)

**1. ~~The LUTE task model has no test.~~ TESTS DONE (`bdbe67b`), CI STILL OPEN.**
`lute/test_glint_index.py` covers all eight validators and the launcher's per-destination flag
filter: 41 tests, no GPU, no psana, no data. The launcher tests **run** `glint_launch.sh` with a fake
`python` on `PATH` that records argv, rather than reimplementing its filter — the two defects found
by hand were a shell-level import failure and a missing whitelist entry, neither of which a
Python-only test would see.

Verified by mutation rather than by passing: removing `--pf8-min-snr` from the whitelist fails
exactly the regression test written for it, and disabling the two-source check fails three.

Two things the exercise exposed. The model is **pydantic V1 only** — `_xtc_requires` and `_xtc_only`
take the `field` argument, which v2's shim refuses outright, so the class cannot be constructed under
v2 at all; LUTE pins v1, and on a v2 box the tests point `pydantic` at the bundled `pydantic.v1`.
And the first draft passed 23 of 41 **vacuously**: a stand-in base from the wrong pydantic attached
no validators, so every negative test failed loudly and every positive one passed for no reason.

**Still open:** this repo has no CI. A GitHub runner has no GPU, so a workflow could cover exactly
this pure-Python layer — the one that until now had no tests.

**2. ~~`--pf8-min-snr` cannot reach the program.~~ DONE (`7867d88`).** It exists in `glint_xtc.py`'s argparse and is a real
tuning knob (see #6), but it is missing from `XTC_FLAGS` in `glint_launch.sh` and has no field in
`IndexGLINTParameters`. A user who sets it gets it reported as dropped. Fixed by adding all five threshold
fields to `IndexGLINTParameters` and completing the launcher whitelist, so the whole peak-finder
group is now settable from LUTE rather than just this one flag. It was exactly the class of gap
that only running the entry point reveals.

**3. ~~The emitted `.stream` is not mergeable.~~ DONE (`bfb8a0e`), verified on real data.** `glint/stream.py:51` writes every reflection row as
`h k l 0.00 0.00 0.00 0.00 0.0 0.0 p0` — only the Miller indices are real; `I`, `sigma(I)`, `peak`,
`background` and the `fs/ss` detector positions are placeholders. What the stream genuinely carries
is the cell and the per-frame orientation. `partialator`/`process_hkl` need real intensities, so
downstream must re-predict and integrate from the orientation.

`--integrate` now adds a SECOND PASS: `frames_for_events()` re-reads the run and calls
`det.calib` **only on the events pass 1 indexed**, then predicts each frame's reflections from
its recovered orientation and box-integrates. Cost is about `n_events*0.94ms +
n_indexed*145.8ms` rather than a second full pass, because calibration is the entire cost of an
event and skipped events cost only the stream walk.

Verified on real data, S3DF job 34274599 (mfxx49820 r0016, 200 events, `--int-dmin 2.5`):
50 integrated chunks with real `I`, `sigma(I)`, `peak`, `background` and real `fs/ss` — every
one of which was `0.00` before.

**Requires `--geom`, and says so rather than guessing.** Prediction projects q onto named
CrystFEL panels (corner, fs/ss basis, res, coffset); psana per-pixel coordinates are positions,
not a tiling, and do not define that model. `glint.lute_bridge.parse_geom` already emits the
panel dicts `predict.project_q` wants, and the `(nseg,H,W)` frame reshapes to the `(nseg*H, W)`
slab a `.geom` addresses, so no new geometry code was needed.

Two things to expect in the output. **Negative `I` is normal** — `I = signal - nbox*bg`, so a
predicted reflection with nothing there integrates to noise about zero; partialator handles it.
And GLINT imposes no symmetry, so the stream says `lattice_type = triclinic`: the sample's
symmetry still has to be supplied downstream.

**~~4. ~150 ms/event, 98% of it CPU calibration.~~ DONE, and it is exact.** `experiments/xtc_bridge/gpu_calib.py`
reproduces `det.calib` on the device **bit for bit** on mfxx49820 r0016 (Epix10ka2M, 40 events, job
34277115): 0.000 ADU residual, and the same PeakFinderV4 run on both images returns **the identical
445 peaks — 0 only-CPU, 0 only-GPU, Jaccard 100.0%**.

| path | median ms | speedup |
|---|---|---|
| `det.calib` (CPU) | 141.32 | |
| GPU, full (ped + common mode + gain + mask) | 3.82 | **37x** |
| GPU, no common mode (ped + gain + mask) | 0.73 | **193x** |

Getting there took correcting the plan above, which was written from `Reader.cu` and was wrong in
two places. Both were found by reading psana's own `Detector/UtilsEpix10ka.py` and
`Detector/UtilsCommonMode.py`, after SLAC's Confluence notes on
[det.calib algorithms](https://confluence.slac.stanford.edu/spaces/PSDM/pages/349284620/Method+det.calib+algorithms)
and [common mode algorithms](https://confluence.slac.stanford.edu/spaces/PSDM/pages/165089547/Common+mode+correction+algorithms)
pointed at them:

- **`* gains[i]` is wrong for this detector family.** `det.gain()` holds gain in **ADU/keV** for
  epix10ka and Jungfrau, and psana builds `gfac = divide_protected(ones, gain)` and multiplies by
  *that* — it DIVIDES. Only CSPAD and epix100a hold a keV/ADU factor that is multiplied, which is
  what `Reader.cu` does. Multiplying here is wrong by gain squared.
- **`det.calib` ends with `* det.mask_total`.** Not in `Reader.cu`, not previously here.

The **gain-range decode** warning above stands and was already handled: the mode is not the raw high
bits, it is per-pixel detector configuration OR'd with data bit 14, so it collapses to two
precomputed plane sets from psana's own `gain_maps_epix10ka_any`.

The **common mode** warning does NOT stand as written, on two counts. It is not per-ASIC: groups are
bounded by the **bank** — panel (352,384) splits at row 176 into two ASIC rows, each into 8 banks of
(176,48) — and it is a masked median over H/M-gain pixels only, applied banks→rows→cols per the
`mode` bitmask. It is implemented and verified against psana's own routines over modes 2/1/4/3/7
(`experiments/xtc_bridge/test_gpu_calib_cm.py`, numpy-only, no GPU needed) — to a tolerance of
2e-3, against the live library where psana is importable and against checked-in golden outputs
generated from it otherwise.

And on this run it does **nothing at all** — for psana as much as for us (job 34277169). The run's
`common_mode` constants are the default `(7,2,10,10)`, so `cormax` = 10 ADU, while the actual
pedestal-subtracted column medians are **290 ADU (p90 485, max 1051)**. Every one of the 12,288
column groups fails the guard, so psana applies zero correction: `det.calib(evt)` and
`det.calib(evt, cmpars=(7,0,0,0))` differ by **0 pixels**. Two consequences:

- This is the only reason the 193x row is safe. It is a property of this run's calib constants, not
  of the detector, so **it must not be hardcoded** — `cmpars` is read from `det.common_mode(run)`.
  The 3.1 ms the correction costs is the segmented sort, and on a run whose `cormax` is set
  appropriately that cost buys real changes to the image.
- The experiment's common-mode parameters are ineffective. Confluence says `par[2]` "needs to be
  adjusted by users per experiment"; here it was left at the default and there is an uncorrected
  ~290 ADU per-column baseline. Worth raising with the beamline — it does not hurt GLINT (the
  peak-finder's annulus is local, hence the 100% Jaccard) but it will hurt anything integrating
  absolute intensities.

**Wired into the psana1 reader behind `--gpu-calib`**, and verified end to end (job 34277504,
mfxx49820 r0016, 1500 events, A100). The two runs differ only in that flag:

| | `det.calib` | `--gpu-calib` |
|---|---|---|
| wall | **4m38s** | **1m08s** (4.1x) |
| frames with >=6 peaks | 629 | 629 |
| blind indexed | 612/629 (97%) | 612/629 (97%) |
| consensus cell | [38.3 79.1 80.3] A, support 239 | identical |
| final indexed | 589/629 (93%) | 589/629 (93%) |
| stream sha256 | `728ce3c5bd6572a4` | `728ce3c5bd6572a4` |

The streams are **byte-identical** — not equal within a tolerance, the same file. 4.1x rather than
193x is the honest end-to-end figure and is the one to quote: calibration was 98% of the event and
is now ~0, so what remains is GLINT itself, and indexing plus consensus is most of the 68 s.

In pass 1 the frame never leaves the device — the peak-finders take cupy arrays — so the host upload
of every frame goes too, and only peak coordinates come back. Pass 2 must copy back, since
`integrate_spots` is numpy and its `np.asarray` refuses a cupy array.

The flag **raises rather than falling back** on an unsupported detector. A silent fallback would let
a caller believe it got the speedup, or worse believe the two paths agreed because both quietly ran
`det.calib`. The guard cannot key on 4-D pedestals: Jungfrau's are 4-D too, and epix10ka's decode
would have produced garbage there **without raising**. It demands 7 gain modes and a (352,384)
panel. Jungfrau and epixHR need their own decode before they can use this.

**~~5. The launcher's env cannot satisfy both psana and torch.~~ DONE -- GLINT now runs on torch 1.11,
so BOTH envs work.** (Marked done once before in `5d6c9e0` on a false premise; this time it is
measured. See the correction below, which is kept deliberately.)

4.0.58 silently drops a detector whose `ConfigV` it cannot parse: the configStore has no entry and
`psana.Detector()` raises `KeyError: Source string not found in configStore`, which reads exactly
like a mistyped detector name. The reader turns that into a message naming the real cause.

| env | torch | `Jungfrau.ConfigV4` |
|---|---|---|
| `ana-4.0.58-py3-minipytorch` | **2.1.0** | **cannot parse** |
| `ana-4.0.59-py3-minipytorch` | 1.11.0 | parses |

These are the only two ana envs carrying torch at all, so the fix had to be to make GLINT run on
1.11 rather than to find a better env. **Every torch API the `glint` package uses was probed against
both envs** (jobs 34277932, 34278651) rather than inferred from the changelog, which mattered: the
grep that found "files importing torch" missed those written `import numpy as np, torch`, so the
package has EIGHT torch files, not three. Exactly two APIs were missing on 1.11:

- `Tensor.scatter_reduce_(reduce="amin")` (torch 1.12+), at `glint_index.py` 431 and 456. Every
  fallback is also dead on 1.11, measured: `torch.scatter_reduce` has a different signature taking
  no `src`; `Tensor.scatter_(reduce=)` offers only add/multiply and multiply is unimplemented for
  Long on CUDA; `Tensor.index_reduce_` does not exist. Replaced by
  `glint_index.py::_first_index_per_group`.
- `torch.backends.mps` -- the submodule does not exist at all before 1.12, and it was evaluated at
  MODULE scope in `glint_index.py:31` **and `replica_gpu.py:16`**, which is on the rescue path.
  Masked on a GPU node because `torch.cuda.is_available()` short-circuits ahead of it, so it would
  have fired only on a CPU-only node. Both now `getattr`.

Nothing else is missing: CUDA graph capture/replay, `torch.fft.*`, `linalg.pinv`/`det`,
`meshgrid(indexing=)`, `combinations` and the whole nn/functional surface all work on 1.11.

Verified three ways in one job (34278559):

| run | env | result |
|---|---|---|
| **regression** mfxx49820 r0016 | 4.0.58 / torch 2.1 | stream sha **`728ce3c5bd6572a4`** -- byte-identical to its own pre-shim output |
| **cxilu8823 r0226 Jungfrau** | 4.0.58 / torch 2.1 | refuses at `Detector()`, message names the ConfigV cause |
| **cxilu8823 r0226 Jungfrau** | 4.0.59 / **torch 1.11** | **933 frames, 887/933 (95%) indexed, 933 chunks, 8m38s** |

That last row is the whole point: it previously died with `AttributeError` before writing anything
(job 34274283). The shim is exact rather than approximate, which the byte-identical regression is
there to prove -- it sits under lattice-candidate dedup, where a wrong answer would not crash but
would silently change which candidates survive.

**What this does NOT establish.** The Jungfrau run's **consensus REFUSED** -- best cluster 5 vectors
= 0.2% of 2659 pooled, under the `CONSENSUS_MIN_FRAC`/`MIN_LEAD` gate -- so the 95% is per-frame
top-1 with no cross-frame validation and no rescue. That is the gate working as designed: this run
was given `--zdist` from psana's `coords_z` and no `--geom`, and psana's deployed geometry is the
unrefined starting geometry (see item 7). So item 5 proves the ENV and torch half. It does not yet
show Jungfrau indexing is scientifically good, which needs a refined geometry and item 6's
thresholds.

Default stays 4.0.58 (newer torch, covers every detector whose ConfigV it can parse); for one it
cannot see, set `GLINT_ANA_ENV=ana-4.0.59-py3-minipytorch`.

The earlier failure is worth keeping visible: `5d6c9e0` asserted the two envs carried the same torch
1.11.0 after measuring only 4.0.59 and assuming the other matched. The reader's own error message
repeated that claim ("4.0.59 can, with the same torch+cupy") until this round.

Still carrying the same latent `torch.backends.mps` line, but NOT on the LUTE path and so left
alone: `experiments/paper_xg_gpu.py`, `experiments/bench_h2h.py`, `experiments/powder_ml/train.py`.

**~~6. `PF8_MIN_SNR = 15` is detector-specific, and that is measured.~~ MEASURED — AND THE NUMBER
THEN MOVED (glint#139); read to the end of this item.** Same ladder on Jungfrau4M
(`cxilu8823` r0226, 8x512x1024, 75 um), job 34224833, `thr_snr=5`/`min_pix=3`:

| min_snr | Epix10ka2M | Jungfrau4M |
|---|---|---|
| 10 | 84% of events (pass-through) | 97.7% |
| 15 | **37%**, 100% of btx-idx | **95.9%** |
| 20 | — | 85.7% |
| 30 | — | 50.9% |

Both detectors show the same shape — a flat region then a knee — but Jungfrau's knee sits roughly 2x
higher. The shipped 15 selects 37% of events on one detector and 96% on the other. Calibrate per
detector before using pf8 on new hardware; v4 (the default) is unaffected.

**REDONE ON A GOOD RUN, 2026-08-06 (job 34379248).** `cxilu8823` **r0207** — a run the beamline kept
and indexed, **1623/3000 = 54.1%** with xgandalf and the cell supplied. Its images are gone (xtc on
tape), but its peak lists survive in the beamline's stream, and a peak-finder's SNR cut keeps the
strong peaks and drops the weak, so cutting those lists by intensity walks the same axis. Every rung
is scored by the thing that matters — how many of the **3000 offered frames** still index, the same
denominator as the reference:

| keep | median pk/frame | frames with >=10 pk | known-cell | blind |
|---|---|---|---|---|
| 100% | 43.9 | 3000 | **72.3%** | 70.7% |
| 75% | 32.9 | 2983 | **72.6%** | 71.0% |
| 50% | 21.9 | 2693 | 64.3% | 62.0% |
| 35% | 15.3 | 1916 | 50.0% | 48.0% |
| 25% | 11.0 | 1251 | 35.1% | 41.6% |
| 15% | 6.6 | 508 | 14.8% | 14.3% |

Three things this says that a pass-rate ladder could not.

**The safe band is wide.** Discarding the weakest QUARTER of every frame's peaks costs nothing at
all (72.3 -> 72.6%). Even at half, GLINT still beats the reference. At 35% — two thirds of the peaks
thrown away — it is level with what xgandalf achieved using all of them.

**What collapses the yield is `min_peaks`, not indexing.** The per-frame success rate RISES as the
cut tightens (72% -> 87%), because the frames that survive are the strong ones. The 3000-frame yield
falls only because frames drop below `--min-peaks 10` and are never offered. So the risk in a
mis-set `PF8_MIN_SNR` is not degraded solutions, it is **silent frame loss through a coupled
parameter** — and `min_peaks` is where a per-detector adjustment has to be made too.

**The lattice is robust.** The blind consensus cell moves from `[49.7 56.5 68.5]` to
`[49.2 56.9 68.6]` across the whole ladder — stable after losing 85% of the peaks.

One row to read carefully: blind at keep=25% reports 1247/1251 (99%), but its consensus **REFUSED**
(best cluster 19.3% of the pool) and it fell back to per-frame top-1 with no cross-frame validation.
That 99% is the pattern item 7 exists to make visible, not a result.

**What this is NOT.** An intensity cut is a PROXY for an SNR cut, not the same knob: peakfinder8's
SNR is relative to a local background, so a weak peak on a quiet background can outrank a stronger
one in a noisy region. This measures selectivity-by-strength.

**THE REAL `min_snr` LADDER, 2026-08-06 (job 34387880).** Run on raw Jungfrau images from a run its
experiment KEPT: `cxilw5019` r0019 (A2A receptor, `CxiDs1.0:Jungfrau.0`, staged xtc + covered by the
beamline's own CrystFEL stream — 1081 Cheetah hits over the run, 80 indexed). Screening candidates
by hardware is what led to two dead datasets first: an earlier draft of this section pointed at
`cxil1005322` (57 TB staged, refined geometry) as "the place to do it" — probing it found reborn
radial-profile configs, a photon-counting histogram and an AgBeh geometry fit, i.e. **solution
scattering, not crystallography**, and GLINT on 2000 of its events duly "indexed" 100% with
consensus at 0.2%. Screen on `.cell`/`.stream` evidence, never on the detector.

pf8 over 3000 events, `thr_snr=5`/`min_pix=3`, geometry from the beamline's stream header:

| min_snr | frames >= 10 pk | pass | blind top-1 | consensus |
|---|---|---|---|---|
| 3 | 1953 | 65% | 94% | refused, 0.3% |
| 6 | 1861 | 62% | 94% | refused, 0.3% |
| 10 | 762 | 25% | 88% | refused, 0.7% |
| 15 | 444 | 15% | 80% | refused, 0.8% |
| 20 | 332 | 11% | 79% | refused, 0.9% |
| 30 | 207 | 7% | 87% | refused, 0.6% |

Three findings. **The pass-rate knee on this run sits between 6 and 10** (62% -> 25%), so the
shipped 15 is past the knee, not before it. **Practitioners run this detector far below 15**: this
experiment's Cheetah used `t100-s6-rad5` (min_snr ~6), cxilu8823's indexamajig used `--min-snr=3.5`
— the shipped value is 2.5–4x above both, which INVERTS the original conclusion that Jungfrau
"tolerates" a high cut; that tolerance was the discarded run passing everything. And **no rung
locks consensus** (best 0.9%): 3000 events of a run whose own hit yield was 1081 over its full
length simply carries too few true hits, so this dataset cannot produce a scored ladder either.
A known-cell score was tried and is recorded as a NEGATIVE: 77% of snr-3 frames "index" to the
40.4 x 180.7 x 142.8 A C-centred cell against the beamline's 7.4% — a cell that large has a dense
enough reciprocal lattice that ten noise peaks fit it, so "indexes to the known cell" is not a
noise-proof anchor.

What item 6 still needs, precisely: a Jungfrau run with raw images on disk AND an event-mapped
reference hit list (btx-style). None of the staged datasets provides both; everything above brackets
the answer without closing it.

**Caveat on the ORIGINAL Jungfrau column above.** It is measured on `cxilu8823` **r0226**, and
r0226 is a run the experiment's own processing DISCARDED — their CrystFEL stream
(`results/prabin/rhodopsin/100us/100us204-232.stream`) covers 20 runs in 204–232 and skips
226/227/228. r0226 is also the only run of that experiment with raw xtc still on disk, which is why
it was used. A pass-rate ladder on a run whose frames are noise-dominated will show a high
pass-rate at every threshold, so "Jungfrau's knee is 2x higher" may be a property of THAT RUN rather
than of the detector. The re-measurement does not need xtc: the beamline stream carries the peak
lists for the good runs (r0207: 3000 frames extracted, 54.1% indexed by xgandalf), and
`glint_cli --peaks --geom` consumes them directly. **Until that is redone, treat the Jungfrau row as
provisional.**

**RESOLVED 2026-08-13 — measured on the detector SFX actually uses.** The ladder ran on raw images
of `mfx101555026` r0013 (**Jungfrau 16M**, LCLS-II xtc2, MFX; ClCRY4, 54.15/87.29/141.33 A), 3000
events, all eight rungs peak-found from ONE calibration per event, against an event-mapped reference:
the experiment's own Cheetah `frames.txt` covers every event, hits AND misses, and psana2
`evt.timestamp` equals Cheetah's `event_id` exactly and in order (verified 3000/3000). Jobs
34783529/34786015; outputs in `~/glint_16m/` (`ladder_r0013.json`, and `r0013_pf8snr5.cxi` — a
btx-shape event-mapped peak list over all 3000 events, the reference that previously existed nowhere
on disk for any Jungfrau run).

Two findings, and the first reframes the item:

1. **The calibrated object is the PAIR (`threshold`, `min_snr`), not `min_snr` alone.** CrystFEL's
   peakfinder8 applies an absolute ADU floor (`--threshold`) on top of the relative snr test, and
   every practitioner value ever held against our 15 was paired with one (this beamline:
   `--threshold=110 --min-snr=5`). Our finder had no such floor until `thr_adu` (glint#108, default
   `None` = bit-identical to before). Without it, snr 3–10 called 92% of events hits against the
   beamline's own 30% — so the Epix10ka2M ladder above, measured floor-less, is not comparable to
   any quoted practitioner setting.

2. **With the beamline's floor (110 ADU), the shipped 15 sits near the knee on Jungfrau 16M.**
   Scored against Cheetah's per-event hit flag (1301 hits / 3000): min_snr 10 = 99.9% recall at
   48.3% precision (over-calls 2x; median 51 peaks/frame), **15 = 64.3% recall at 90.7% precision**
   (922 offered, median 22 peaks), 20 = 45.1% recall at 100.0% precision (587 offered, zero pickup
   on non-hits). The knee is between 10 and 15 — the opposite of the floor-less Epix10ka2M picture,
   and the low rungs still carried a median of 478 peaks/frame WITH the floor, which is why a bare
   snr number was never the knob anyone else was quoting.
   **SUPERSEDED — this ladder ran with the detector's interior ASIC seams live; see the 2026-08-25
   re-measurement below. The knee described here is the seams', not the sample's.**

Method caveats, stated: the reference is Cheetah's hit definition (t100-s6), so precision/recall are
consistency against the beamline's validated processing, not absolute truth; 250/3000 events had no
finite `ebeamPhotonEnergy` and used the run-median wavelength rather than being dropped (the reader's
default skip would have silently shrunk the denominator). Indexing yield could NOT serve as the
score here: blind consensus REFUSED at every rung (support 0.3–2.7%), because ClCRY4's ~667,000 A^3
cell is blind-spurious-limited — which independently replicates the paper's cxidb-62 large-cell
finding on a different detector, protein and facility, and is the acceptance gate doing its job.
(The first attempt crashed the consensus on degenerate N-best hypotheses; fixed in glint#109.)

Reproduction recipe (the environment split is real): stage 1 (read + peak-find) needs psana2+cupy —
the default conda2 release has NO cupy; `conda activate xpp_drp_gpu_311` then **`unset PYTHONPATH`**
(psconda.sh pins the release psana ahead of the env, which fails as a circular import). Stage 2
(indexing) needs torch — conda1 `ana-4.0.59-py3-minipytorch`, which cannot read xtc2. The two stages
hand off q-vectors as an npz, which also makes re-scoring free.

**RE-MEASURED 2026-08-25 (glint#139) — the ladder above was calibrating the detector, not the
sample.** The finder on that route is the stacked pf8 of `xtc_core.py`, whose "panel seams masked"
is a synthetic one-row separator between the 32 modules; the **interior** ASIC seams (row 256, cols
256/512/768 within each 512x1024 module) were live, and psana's `_mask_edges()` masks perimeters
only. The screen was re-run on the same 3000 events of `mfx101555026` r0013, from the same frozen
checkout, in three arms differing ONLY by the mask handed to `prep_geometry(..., good=)`: none,
`asic_seam_mask(width=1)`, `asic_seam_mask(width=2)` (glint#127). Jobs 35797915/16/17, indexing
35800179; outputs `~/glint_16m/seams_{nomask,seam1,seam2}.json`. The no-mask arm reproduces
`ladder_r0013.json` exactly at all eight rungs, so this is an A/B and not a re-derivation.

**a. The peaks were the seams.** Fraction of returned peaks sitting on a seam pixel, against the
0.97% expected if they were spread over the live area: **94.6% at min_snr 3-6 (97x)**, 78.5% at 10,
**18.7% at the shipped 15 (19x)**, 6.2% at 20. Masking removes 92.6% of all returned peaks at the
low rungs (1,460,404 -> 108,506). The contamination is a ONE-PIXEL line: the +-1 and +-2 bands add
~0.1 points over d=0, so `width=1` suffices — its surviving d=1 ring is 0.3-0.8% against a 0.98%
by-area expectation, i.e. at or below chance. `width=2` costs a further 1% of the module and moves
no metric by more than 0.5 points.

**b. The knee moves from "between 10 and 15" to between 6 and 8.** Masked (width=2), scored the same
way: snr 6 = 81.2% recall / 55.6% precision / 843 pickup; **snr 8 = 66.8% / 98.1% / 17 pickup**;
snr 10 = 58.2% / **100.0%** / 0; snr 15 = 50.1% / 100.0% / 0; snr 20 = 43.1% / 100.0% / 0. Precision
SATURATES at 10, so the shipped 15 costs 8.1 points of recall for nothing and 20 costs 15.1. The
corrected pair is **(`thr_adu` 110, `min_snr` 8)** for recall, or (110, 10) for zero pickup.

**c. The two knobs are coupled, so DO NOT move the default alone.** (110, 8) is valid only WITH the
mask. Unmasked at snr 8 the finder offers 2720 of 3000 frames at 47.8% precision — a pass-through,
worse than the 15 it would replace. `asic_seam_mask` is still deliberately unwired into every ingest
path (glint#127), so `PF8_MIN_SNR` stays at 15 until it is wired into `_StackedFinder`'s mask; that
wiring and the default change belong together, in one change, re-verified against this table.

**d. Cheetah does not absorb the artifact, so nothing cancelled.** Cheetah reports a median of 64
peaks/frame on hits and a maximum of 30 on non-hits (its hit test is essentially ">30 peaks"), and
its geometry declares every ASIC a panel, which masks the seams for free. The unmasked finder
reported a median of 478 peaks/frame over ALL 3000 events. That is why unmasked recall read 100% at
snr 3-10: it called 91% of events hits against Cheetah's 43% — a pass-through, not sensitivity. The
masked arm's 45-55 peaks/frame is the number that lives in Cheetah's regime.

**e. The consensus refusal is NOT the seams.** Blind consensus still REFUSES at every rung with the
mask on. Masking roughly doubles the support fraction (0.3-2.7% -> 1.1-3.4%, and 0.30% -> 2.89% at
snr 8) without clearing the gate, which leaves the large-cell explanation above intact: the limit is
ClCRY4's ~667,000 A^3 cell, not the detector. One baseline number should be retired, though —
unmasked known-cell indexing "succeeded" on 99.9% of frames at snr 3-6 while 94.6% of the peaks
being indexed were seam pixels. A 500-peak list laid out along four straight lines fits almost
anything; that row was never evidence of indexing quality.

**~~7. Geometry provenance is load-bearing and silent when wrong.~~ NO LONGER SILENT.** On
`mfxx49820` psana's deployed geometry is the unrefined 2021 starting calibration; blind indexing
locked a wrong doubled-*c* cell at support 23/6294 and reported success. Supplying btx's refined
`.geom` took support to 832 and the recovered cell to `[38.3 79.1 80.3]` against a truth of
`[38.4 79.3 79.5]`. `--geom` and `--calib-dir` are not optional conveniences on this route. The
consensus gate from glint#83 (pool share 2%, runner-up margin 1.5x) refuses that 23/6294 lock —
load-bearing, not tunable.

`experiments/xtc_bridge/geom_provenance.py` now runs at reader startup and reports one of three
states. It **warns, never raises** — but it never says nothing:

| state | meaning |
|---|---|
| `CORROBORATED` | a `.geom` was supplied and agrees with psana |
| `DISAGREE` | a `.geom` was supplied and does not — with the disagreement characterised |
| `UNVERIFIED` | no `.geom`; nothing corroborates psana. **Says so.** This was previously indistinguishable from success |

**The discriminator, and the one that had to be discarded.** Two geometries differ in two ways and
only one matters: a global scale/distance term (benign — `--zdist` sets the distance) or a per-panel
shape term (tilts/offsets, which `--zdist` cannot touch, and which stops blind indexing converging).
The obvious test — "does one global scale explain it?" — **does not work**, and the test suite pins
that: on a synthetic per-quadrant error a global scale still explains 54%, so keying on it would
wave the real failure through as a distance problem. What separates them is **inter-panel
dispersion** relative to the error, measured on geometries where the answer is known:

| synthetic case | scale explains | dispersion/median |
|---|---|---|
| pure distance +3% | 91.6% | **0.07** |
| per-quadrant ±2% | 54.2% | **1.01** |
| per-panel shifts | 18.1% | **0.91** |

An order of magnitude apart, so the 0.3 cut is the middle of a gap rather than a tuned number.

Verified on the real known-bad case (job 34280747, 1,946,419 pixels, psana `0-end.data` vs btx
`r0016.geom`): median 1.586%, **inter-panel dispersion 1.811%, ratio 1.14, signs 8 positive / 8
negative**, and a global scale would explain **0.0%** — `DISAGREE`, naming SHAPE. The same run with
no `.geom` returns `UNVERIFIED`, and psana-against-itself returns `CORROBORATED` at exactly 0.000%,
so the check does not cry wolf on a good geometry.

It also names the path failures psana accepts **silently**, each of which makes `--calib-dir` a
no-op while looking fine: a nonexistent path (psana falls back to the default), a path with no
geometry files for that detector source, no deployed range covering the run, and a relative path the
launcher's `cd` will re-resolve. And it flags when `--zdist` matches psana's own nominal `coords_z`
to within 1%, because then nothing independent constrains the scale and a global |q| error rides
along invisibly — which is exactly the `cxilu8823` r0226 situation.

`test_geom_provenance.py` covers all of it on synthetic coordinates: no psana, no cupy, no GPU.

---

## What is left

All seven items are closed. Item 6 closed 2026-08-13 as a measurement on Jungfrau 16M (see its
section: the calibrated object is the (`threshold`, `min_snr`) pair; `thr_adu` landed in glint#108),
and was re-measured 2026-08-25 (glint#139) with the interior ASIC seams masked, which replaces the
operating point that measurement recommended without reopening the item.
CI closed via glint#107: six CPU-only test files run on every push and pull request
(`lute/test_glint_index.py` — 44 tests — plus five `xtc_bridge` script tests), with the GPU
(`test_core.py`) and MPI (`test_mpi_smoke.py`) tests excluded as unhostable and said so in the
workflow. What remains is beyond the seven, not blocking them:

1. **`gpu_calib.py` is Epix10ka-family only** and refuses loudly on anything else (item 4).
   Jungfrau and epixHR need their own decode (`UtilsJungfrau` / `UtilsEpixHR`) before they get the
   4.1x — and this now matters more than when it was written: current SFX at MFX runs Jungfrau 16M
   on LCLS-II xtc2, where the psana1 route (and with it `gpu_calib`) does not apply at all.
2. **The psana2 route has no GPU-capable default environment**: the conda2 release lacks cupy, the
   one env with psana2+cupy (`xpp_drp_gpu_311`) lacks torch, and activating it under psconda.sh
   requires `unset PYTHONPATH` (the release psana is pinned ahead of the env). The item-6 ladder ran
   as two stages with an npz handoff for exactly this reason; anything productized for current MFX
   data inherits the same split until an env carries all three.
3. **`asic_seam_mask` is not wired into any ingest path** (glint#127, glint#139). On Jungfrau 16M
   that leaves 94.6% of the peaks the offline stacked pf8 returns sitting on an interior ASIC seam.
   Wiring it is what unblocks the corrected `(110, 8)` operating point in item 6 — and it must land
   WITH that default change, never before or after it, because neither knob is safe on its own.
   `width=1` is the measured-sufficient setting.
