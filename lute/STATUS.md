# GLINT in LUTE — what works, what is measured, what is not

Written 2026-08-05. The point of this file is that the state of this integration should live next to
the code, not in a chat log or one person's head. Every number below has a named source; anything
without one is marked as an assumption.

**Bottom line: not production-ready.** The raw-xtc route runs and produces a correct cell on real
data and, with `--integrate`, a stream partialator can merge. It is still ~150 ms/event with
98% of that in CPU calibration, and no ana env satisfies both its psana and its torch needs.

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
| `IndexGLINTParameters` validators (`_one_source`, per-source `peakfinder`) | **never run** — no test, no CI |
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
- stream is valid CrystFEL 2.3 with 54 chunks and 50 `indexed_by = glint`

Two things that run also shows. The stream reports `lattice_type = triclinic, centering = P`
because GLINT imposes no symmetry — the true lysozyme is tetragonal *P*4<sub>3</sub>2<sub>1</sub>2,
so symmetry has to be supplied downstream. And the reflection rows carry the placeholder
`I`/`sigma(I)`/`fs`/`ss` described in #3 below; this is what a merge would have to be built on.

---

## The seven things that stand between this and production

**Progress: 3 of 7 done**, item 1 partially (tests yes, CI no); item 5 was marked done and reverted. Struck-through items are closed,
with the commit that closed them and how it was verified. The rest are open and unchanged.

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
`mode` bitmask. It is implemented and verified bit-exact against a transcription of psana's routines
over modes 2/1/4/3/7 (`experiments/xtc_bridge/test_gpu_calib_cm.py`, numpy-only, no GPU needed).

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

Not yet wired into the reader — that is the next step, and it is now a plumbing change rather than
an open question.

**5. The launcher's env cannot satisfy both psana and torch.** (Was marked done in `5d6c9e0`; that was wrong -- see below.) That release
silently drops a detector whose `ConfigV` it cannot parse: the configStore has no entry, and
`psana.Detector()` raises `KeyError: Source string not found in configStore`, which reads exactly
like a mistyped detector name. Measured on `cxilu8823` r0226 (Jungfrau4M): `Jungfrau.ConfigV4` is
absent under 4.0.58 and present under **4.0.59**, same stack, same torch 1.11 and cupy 13.0. The
reader distinguishes the two causes in its error message.

**The env fix does not exist.** Repinning to 4.0.59 looked right and was reverted: these are the
only two ana envs carrying torch at all, and neither satisfies both halves.

| env | torch | `Jungfrau.ConfigV4` |
|---|---|---|
| `ana-4.0.58-py3-minipytorch` | **2.1.0** | **cannot parse** |
| `ana-4.0.59-py3-minipytorch` | 1.11.0 | parses |

4.0.59 is a torch DOWNGRADE, and GLINT's M2 dedup calls `Tensor.scatter_reduce_`, added in torch
1.12 -- so under 4.0.59 indexing dies with `AttributeError` before writing anything (S3DF job
34274283). I asserted in `5d6c9e0` that the two envs carried the same torch 1.11.0; I had only
measured 4.0.59 and assumed the other matched.

Default is back to 4.0.58, overridable with `GLINT_ANA_ENV`. It works on every detector whose
ConfigV it can parse, which is all of them except the newest. For one it cannot see, the routes
are a `scatter_reduce_` compatibility shim in `glint/glint_index.py`, or reading in 4.0.59 over
**envbridge** and indexing in 4.0.58 -- exactly the split envbridge already performs for psana2.

**6. `PF8_MIN_SNR = 15` is detector-specific, and that is measured.** Same ladder on Jungfrau4M
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

**7. Geometry provenance is load-bearing and silent when wrong.** On `mfxx49820` psana's deployed
geometry is the unrefined 2021 starting calibration; blind indexing locked a wrong doubled-*c* cell
at support 23/6294 and reported success. Supplying btx's refined `.geom` took support to 832 and the
recovered cell to `[38.3 79.1 80.3]` against a truth of `[38.4 79.3 79.5]`. `--geom` and
`--calib-dir` are therefore not optional conveniences on this route. The consensus gate merged in
glint#83 (pool share 2%, runner-up margin 1.5x) is what now refuses that 23/6294 lock — treat it as
load-bearing rather than tunable.

---

## Suggested order

1. `--pf8-min-snr` through the launcher and the task model (#2) — trivial, and it is a real knob.
2. A test for the task model and launcher (#1) — the validators and the whitelist are pure functions.
3. Integration on the xtc route (#3) — the difference between a demo and a usable plug-in.
4. GPU calibration (#4) — the largest win, and a conversation with the DAQ group rather than a
   solo port, since `Reader.cu` is theirs and the LCLStreamer path needs the same thing.

Items 5-7 are documented rather than fixed; each is a footgun with a known shape.
