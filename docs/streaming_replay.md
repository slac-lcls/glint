# The streaming driver, replayed — provenance of the README animation

The first animation in [`README.md`](../README.md) (`docs/media/streaming_cxidb17_480_dark.gif`
and `_light.gif`, one per colour scheme) is a **screen recording of a replay page** that plays back
**one recorded run** of the shipped `glint.stream_driver.StreamDriver`; the second (the two-species
replay, [below](#two-species-replay--the-cell-registry-at-work)) is built the same way from a second recorded run. It is not a live beamline, not a
simulation, and nothing on it is drawn from a model: every counter, the recovered cell, the lock
frame, the ring occupancy, the rescues and the re-lock are read out of a trace the driver wrote while
it ran. This page says exactly what that run was, how each animation frame maps onto it, what the
crop removed from the replay page, and how to regenerate the whole thing.

## What was run

| | |
|---|---|
| Input | The **480-frame extension** of the cxidb-17 sparse lysozyme run (CXIDB entry 17, Boutet *et al.* 2012, CC0) — the paper's primary streaming set. One protein, one run, the same crystals as the published 120-frame subset; 480 buys statistical power, not generality. Nothing injected, nothing planted, no second species, no drift. |
| Peaks | CrystFEL 0.12.0 `peakfinder8` peaks read out of the blind `indexamajig` stream for these 480 images and converted to reciprocal-space q-vectors through the package's own bridge, `glint.geom.read_crystfel_peaks` + `peaks_to_q`. No pixels are replayed and no peak finder runs in the replay: the driver is fed q-vectors. |
| Wavelength | **One fixed wavelength for every frame** (1.322216 Å = 9377.00 eV, frame 0's photon energy) rather than the per-shot SASE energy. That is the convention the published 120-frame q lists were built under; it is worth a few frames per arm, so it is stated rather than assumed. |
| Indexer | `glint.stream_driver.StreamDriver` at commit `2d6eacb53a33`, cold-started with no cell (`Mc=None`). |
| Arm | `B=20, dmin=2.0, tol=0.002, warmup_nbest=3, warmup_rescue=True, adaptive_relock=True, min_inliers=10`; live accept gate `min_inlier_frac=0.15`, `min_inliers=10`, `min_peaks=6`. That is the constructor's defaults apart from the batch size (20, not the default 64) and the live-gate minimum (10, not 0), plus the two opt-ins the paper publishes as arms. It is the `+relock` arm of `experiments/gap_on_real.py` (same keywords; that script derives `dmin` from the data rather than fixing 2.0). |
| Off | retry cascade, live geometry refinement, second-lattice (double-hit) detection, alias gate, lock probe, miss-buffer rescue, warm-up batch fan-out, stream output. |
| Box | NVIDIA A100-SXM4-40GB (host `sdfampere042`), Python 3.9.18, numpy 1.26.3, cupy 12.3.0, torch 2.1.0.post302; 10.5 s of wall time for the whole run. Other GPU architectures can move the count by a frame (the 120-frame arms are one frame lower on an H200), which is why the box is named. |
| Scoring | The paper's strict research bar: correct lattice (`same_lattice` against the driver's own first locked cell) **and** ≥25% of the frame's spots matched **and** ≥10 reflections. The driver's own live gate is looser (count-only); both are on screen. |

Result: consensus locked after **5 frames** on 4 votes (support 4); the 5 frames
spent on discovery were re-indexed against the locked cell and **3** of them cleared the strict bar.
The driver accepted **451 of 480** at its live gate and refused 24. At the strict bar the run indexes
**331 / 480 = 69.0%** — 328 post-lock plus the 3 rescued warm-up frames. That is the high end of the
band the paper prints for the streaming driver on this set, **67–69% (323–331 of 480)**: the low end
is the same driver with `warmup_rescue` and `adaptive_relock` both off, so the two options enabled
here are published arms, not tuned ones. The offline known-cell reference on the same frames at the
same gate is 357/480 (74.4%). The 331 is also the value `experiments/check_numbers.py` carries as
`stream_rate_rescue_of480`, and `README.md` is one of the files that guard checks.

Recovered cell, printed as measured and not re-jittered for display:
a = 78.706, b = 78.792, c = 37.813 Å, α = 90.189°, β = 90.083°, γ = 89.811° — the same lattice as the
lysozyme reference (`same_lattice_vs_reference: true`).

## Per-frame semantics

The replay page keeps a clock, "*N* frames · *T* s". Animation frame *i* (1-based, the number the
clock shows) is the state **after trace record *i* − 1 (0-based) has been pushed**; frame 0 is the
page before anything is pushed (empty chart, "Discovering…"). The display rate is 12 frames per
second of playback, so the clock reads *i*/12 s; it is a display choice and has nothing to do with the
driver's speed.

The trace has 480 records with these fields per pushed frame: `npk` (peaks), `o` (outcome: 1 =
accepted at the live gate, 2 = warm-up, else refused), `buf` (device-ring occupancy after the push),
`wu` (warm-up frame), `lock` (lock fired here), `resc`/`resc_ok` (warm-up frames rescued / clearing
the strict bar at the lock), `sup` and `lead` (consensus support and lead), `m`/`frac` (matched
reflections and matched fraction), `ok` (cleared the strict bar), `flush` (batch of this size
indexed), `wresc` (watchdog rescue), `relock` (adaptive re-lock fired here).

Events, verified against the trace (`python3 -c "import json; tr=json.load(open('docs/streaming_replay/cxidb480_strace_a100.json'))['trace']; ..."`):

| On-screen frame | Trace record (0-based) | What happens |
|---|---|---|
| 5 | 4 | `lock=0`, `sup=4`, `resc=5`, `resc_ok=3`: consensus locks; the 5 warm-up frames are re-indexed, 3 clear the strict bar. Strict count 3 / 480. |
| 170, 174, 183, 222, 236, 325 | 169, 173, 182, 221, 235, 324 | `wresc=1`: the six watchdog rescues — frames the batch pass missed, blind-solved one at a time and accepted after all. |
| 365 | 364 | `relock=1` on a 20-frame flush: the adaptive re-lock **adds** a second cell to the active set. The trace header records this as `relock_frame: 364` — the 0-based record — so "near event 364" is the safe phrasing; the on-screen badge says "re-lock ×1 at frame 365" because the clock is 1-based. The badge holds amber for 2 s of playback (through frame 389). |
| 480 | 479 | Last flush (15 frames). Strict count **331 / 480**, accepted 451, refused 24, ring 0. |

Cumulative strict counts read off the trace at the filmstrip frames: 3 (frame 5), 113 (170), 247
(365), 250 (370), 331 (480); every one matches the on-screen counter in the corresponding still
below. Command: `sum(r['ok'] for r in tr[:N])` for N in (5, 170, 365, 370, 480) → 3, 113, 247, 250, 331.

## What the crop removed — the honesty text

The GIF is the top 560 px of the replay page: the schematic, the cell card and the stream chart. Below
the crop the page carries prose that qualifies every number on it. It is reproduced here because the
GIF cannot carry it.

* **The first 120 of this list are not the published 120-frame subset.** Sliced to its first 120
  frames this input gives 79 at the strict bar with both opt-ins (`sum(ok)` over records 0–119 = 79;
  the replay page states 76 / 79 for the two arms) where the published subset gives 73 / 78: the
  rebuilt q lists keep peaks in the last fractional pixel of each panel that the pre-fix geometry code
  dropped. Different peak lists, expected, not a discrepancy — but it is why the number in the README
  is quoted with its input named, and why no 120-frame band number appears beside it.
* **Fixed wavelength.** See the table above. The published q lists were built under one fixed
  wavelength; this run follows them so the arms are comparable, at a cost of a few frames per arm.
* **Panels were stripped, not faked.** The replay page descends from a richer streaming-monitor
  template. Every panel that the shipped driver at this configuration does not fill was removed
  rather than left idle or filled from a model: no detector-geometry box (`geom_refine=False`), no
  operational-QA or spurious-meter row (no injections in a benchmark replay; `lock_probe=False`), no
  sample-mixture panel (one protein), no blind-fleet multiplier (the warm-up blind indexer is serial),
  no second-lattice KPI (`double_hit=False`), no blank/veto lane (the benchmark peak lists are already
  hit-selected). What is left is what the driver does: fill a device ring of B frames, batch-index them
  against the locked cell, accept or refuse each at the live gate, and — before any of that — discover
  the cell blind from the first frames by consensus, then rescue the frames discovery spent. The
  live-monitoring direction itself belongs with the LCLStreamer project, not with this replay.
* **No space-group symbol — metrics only.** The cell card prints the six lattice parameters as
  measured and the lattice system they imply ("tetragonal"); the label "Lysozyme" is the reference
  lattice the locked cell was checked against. Nothing on the page asserts a space group, and the
  strict bar is a lattice test, not a symmetry assignment.
* **The re-lock is spurious and adds nothing to the count.** Near event 364 the adaptive re-lock
  fires. It does not replace the locked cell; it *adds* a second cell to the active set, and that cell
  (87.464, 87.556, 109.462 Å; 69.336°, 72.747°, 100.715°) is **not** the lysozyme lattice
  (`same_lattice_vs_locked: false`, `same_lattice_vs_reference: false`). It is a cell voted by frames
  the active set could not explain — the paper's SI describes it the same way: not a second crystal
  form but the response to an accumulation of unexplained frames. Every number on the page is scored
  against the *first* locked cell only, so the re-lock changes nothing in the 331; it is on screen
  because it is what the mechanism did. The paper's "cell composition" diagnostic is the readout for
  this case — a second cell that accumulates frames mid-run indicates a sample change, one that
  accumulates almost none is an alias that survived the gate — but the trace scores against the first
  cell only and does not record the second cell's share, so that reading is left to the driver's own
  report, not asserted here.

## Regeneration

The recorded trace, the builder that turns it into the replay page, and the built replay page are
committed as read-only copies under [`docs/streaming_replay/`](streaming_replay/) (a separate commit,
so it can be dropped without touching the README or the media):

* `cxidb480_strace_a100.json` — the driver trace: a header (dataset, peaks, wavelength convention,
  `driver_kw`, gate definitions, the recovered cell, counts, the published band, host provenance) and
  480 records. Its top-level key is `trace`.
* `build_cxidb480_monitor.py` — builds the replay page from the trace. It starts from a streaming
  monitor template that is **not** in this repository, so it documents the build rather than
  reproducing it from a bare checkout; it fails fast if any anchor it edits is missing, and asserts
  `EXPECT_TOTAL, EXPECT_FRAMES = 331, 480`.
* `glint_streaming_cxidb480_real.html` — the built replay page, self-contained (the trace is inlined
  as `TRACE`). Open it in a browser to see the same replay with its prose; `?theme=dark|light`,
  `?print=1` (hide the controls, freeze the state) and `?warm=W` (pre-advance the clock by *W* seconds
  of playback) are the switches the capture used.

Capture recipe (macOS, Google Chrome 152 headless, Pillow 10.4; no ffmpeg):

1. For each animation frame *i* = 0 … 480, compute the `?warm=` value that yields exactly *i* frames on
   the clock by replaying the page's own advance loop (`for (acc=0; acc<W; acc+=0.033) advance(0.033)`,
   frames shown = Σ floor(12·0.033 + carry)) in IEEE doubles; frame 0 is the page with no `?warm`.
   Never capture past 480: `?warm ≥ 40.083` wraps to frame 1.
2. Launch one headless Chrome per frame with a fresh `--user-data-dir`, `--force-device-scale-factor=1
   --window-size=1320,896 --virtual-time-budget=1500 --screenshot=f_%04d.png` on
   `glint_streaming_cxidb480_real.html?theme=<dark|light>&print=1&warm=<W>`; poll for a size-stable PNG
   (Chrome never exits on its own because of the rAF loop), then kill it. Four in parallel, ~3.5 s per
   frame per process, ~7 min per theme for 481 frames. A blank-frame guard (grey-level std over the
   chart interior and the schematic) recaptures a frame whose canvases had not drawn; 0 of 962 needed it.
3. Crop to the footer-free top of the page: `(6, 0, 1314, 560)` in the dark theme, `(6, 0, 1314, 563)`
   in the light theme (the light stream card casts a drop shadow that reaches 3 px lower). Resize to
   900 px wide (Lanczos) and write one GIF with a **single global 256-colour median-cut palette**
   built from a mosaic of 9 frames sampled evenly across the run, no dither, `optimize=True`,
   `disposal=1`, 83 ms per frame requested → 80 ms stored (GIF delays are centiseconds), `loop=0`.
   481 frames play in 38.5 s. Sizes: dark 2.54 MB (900×385), light 2.32 MB (900×387).
4. The stills below are the same page at `--force-device-scale-factor=2` (2640×1792 raw, same crop
   box doubled → 2616×1120 / 2616×1126), one Chrome launch each.

## Filmstrip (dark theme; light-theme twins are `light_f*.png` alongside)

| Still | Frame | What it shows |
|---|---|---|
| ![frame 0](media/stills/dark_f0000.png) | 0 | Nothing pushed yet: "0 frames · 0.0 s", badge amber "Blind warm-up — no cell yet", cell card "Discovering…", consensus support 0/4, 0 / 480 pushed, indexed 0 / 480. |
| ![frame 5](media/stills/dark_f0005.png) | 5 | The lock: "5 frames · 0.4 s", badge green "Locked + rescuing warm-up frames", "locked at frame 5", "↑5 rescued at lock", 0 accepted yet at the live gate, cell card LOCKED with the measured lysozyme cell; indexed 3 / 480 (the 3 rescued warm-up frames that clear the strict bar). |
| ![frame 170](media/stills/dark_f0170.png) | 170 | Steady state with the first watchdog rescue: "170 frames · 14.2 s", badge green "Steady — locked, known-cell batch", ring 5, 155 accepted, "↑1 watchdog rescue" flashing on the missed → indexed edge, 10 of 480 missed; indexed 113 / 480 (66%). |
| ![frame 365](media/stills/dark_f0365.png) | 365 | The re-lock, near event 364: "365 frames · 30.4 s", badge amber "Watchdog re-lock — second cell added", "re-lock ×1 at frame 365 · 2nd cell", consensus node highlighted with re-lock vote dots, 338 accepted, 22 of 480 missed; indexed 247 / 480 (68%). The re-lock itself adds nothing to the count. |
| ![frame 370](media/stills/dark_f0370.png) | 370 | Five frames later the amber hold persists (it lasts through frame 389): "370 frames · 30.9 s", ring 5, 343 accepted, "↑6 watchdog rescues" now all in; indexed 250 / 480 (68%). |
| ![frame 480](media/stills/dark_f0480.png) | 480 | The end: "480 frames · 40.0 s", badge green "Steady — locked, known-cell batch", 480 / 480 pushed, 451 accepted, 24 of 480 missed, "↑5 rescued at lock", "↑6 watchdog rescues", "re-lock ×1 at frame 365 · 2nd cell", cell 78.71 · 78.79 · 37.81 Å / 90.19 · 90.08 · 89.81°; **indexed 331 / 480 (69%)**. |

The on-screen percentages are rounded to whole numbers by the page (69% at the end); the README quotes
331/480 = 69.0% to one decimal because the band it sits in is printed to the percent.

## Two-species replay — the cell registry at work

`docs/media/streaming_multicell_dark.gif` / `_light.gif` are a screen recording of a second replay page,
[`docs/streaming_replay/glint_streaming_multicell_real.html`](streaming_replay/glint_streaming_multicell_real.html),
built by [`build_multicell_monitor.py`](streaming_replay/build_multicell_monitor.py) from
[`multicell_strace_a100.json`](streaming_replay/multicell_strace_a100.json) — the JSON
`experiments/record_stream_replay.py` wrote while the shipped driver indexed a **two-species** stream. Same
rules as above: nothing is simulated, every counter is a trace readout, and the page removes what the run
cannot honestly fill. It differs from the 480-frame replay in what it *keeps*, because this run exercises
the driver's cell registry and its adaptive re-lock with a real second lattice to find.

### What was run

| | |
|---|---|
| Inputs | **cxidb-17 lysozyme**: the 480-frame extension above, as CrystFEL peakfinder8 q-lists at the fixed 1.322216 Å (digest `bf3422`; 454 frames used). **Proteinase K** (CXIDB entry 45, Masuda *et al.* 2017, CC0): `experiments/prok_q.npz`, the deposit's reciprocal-lattice-point lists capped at the **170 brightest peaks per frame** — the same lists DIALS was given in the paper's head-to-head, so the cap is part of the record (digest `f01472`; 446 of 907 used). Both real. No pixels are replayed, no peak finder runs, **nothing is integrated**: every frame is index-only (`push_q`). |
| Interleaving | **Planted.** `experiments/schedules/lyso_prok_switch.json`, seed 7: eight scenes — lysozyme hold (120) → 50/50 blend (40) → ProK-major 70/30 (200) → blend (40) → lysozyme-major 60/40 (200) → blend (40) → ProK hold (160) → lysozyme tail (100); the species is redrawn every 2 frames inside a mixed scene, frames are taken in file order, none reused. The page shows the schedule as the ground truth it is, labelled planted: the scene panel with the playhead, and the ribbon under the composition chart. |
| Indexer | `glint.stream_driver.StreamDriver` at `bd79038e77d4` (glint#199), cold-started with no cell (`Mc=None`). |
| Arm | The 480 arm — `B=20, dmin=2.0, tol=0.002, warmup_nbest=3, min_inliers=10, min_inlier_frac=0.15, warmup_rescue, adaptive_relock` — plus `rescue_buffer=64` (the retroactive index-only rescue a re-lock performs on the misses it still holds), `events=True`, and a **roster** `{lyso, prok}` of the two reference cells. The roster names a cell *after* the driver finds it; it never hands the indexer a cell. |
| Off | retry cascade, geometry refinement, second-lattice detection, alias gate, lock probe. |
| Box | NVIDIA A100-SXM4-40GB (`sdfampere036`), torch 2.1.0.post302, numpy 1.26.3; 4.1 s of wall time for the 900 frames. |
| Scoring | The strict bar (≥25% of spots *and* ≥10 reflections) against **each frame's own species' reference lattice**, so a frame claimed by the wrong cell scores zero. The trace also carries the old convention (strict against the driver's first cell only), which is meaningless for two species and is not shown. |

### What happened

Consensus locked the first cell after **5 frames**; the registry named it **lyso** from the roster, and 4 of
the 5 discovery frames were re-indexed against it (2 clear the strict bar). Proteinase K enters at frame 123
(the first blend); its frames miss, the watchdog's misses vote a recurring cell the active set does not
explain, and on the flush at record 145 (on-screen frame **146**) the driver **adds** a second cell, named
**prok** from the roster — one re-lock, no alias — and re-indexes the 10 misses still in its rescue buffer
against it (8 clear the strict bar). Fifteen individual watchdog rescues over the run (frames 189, 248, 266,
274, 285, 317, 418, 530, 585, 641, 761, 794, 815, 840, 863). Final active set: two cells, lysozyme
78.79 · 78.82 · 37.80 Å and Proteinase K 69.06 · 68.86 · 109.34 Å (medians over every frame the driver
attributed to each — accepted live or rescued retroactively, 512 and 315 frames — printed as measured).

| strict, own-species reference | lysozyme (454) | Proteinase K (446) | total |
|---|---|---|---|
| solo, same frames, same arm (`gate480.json`, `prok907.json`) | 311 | 375 | — |
| **this run** (first-fit cell assignment, the default) | **309** | **273** | **582 / 900** |
| this run with `assign="best"` (glint#204; same stream, same node) | 309 | 341 | 650 / 900 |

Attribution, planted species → cell the driver claimed the frame for: lysozyme frames → lyso 427, missed 19,
prok 8; Proteinase K frames → **lyso 85**, prok 307, missed 54. Lysozyme pays two frames for the mixture.
**Proteinase K pays about a hundred, and 85 of them are one mechanism**: the first-fit cascade tries cell 0
first, and a dense 170-peak frame gives the lysozyme cell enough chance near-integer hits to pass the live
gate (median 30 live inliers on those frames) — none of the 85 clear the strict bar under that cell. This is
the measured case for best-fit rather than first-fit cell assignment. glint#204 adds it as an opt-in
(`assign="best"`: every frame is indexed against every active cell, and a challenger takes it from the
first-fit cell only when it explains at least max(8, 5 % of the peaks) more). Re-run on this stream on the same
A100 (job 38563622; first-fit reproduces this trace record for record): **650 / 900** strict, Proteinase K
**341**, lysozyme unchanged at 309 — 69 of the 85 misclaimed frames move to the Proteinase K cell (68 clear the
strict bar; on them that cell explains a median 112 of 170 peaks against the lysozyme cell's 30), not one
lysozyme frame moves. The margin is what keeps it honest on weak frames: plain argmax (margin 0) gives 649 but
moves 21 lysozyme frames into the Proteinase K cell on a +1..+6 count difference, because a wrong cell's best
orientation explains 12–15 % of the peaks by chance, more for the larger cell. The animation stays the
first-fit run — it is the published default, and the cost it shows (the colour disagreement between the
composition bars and the planted ribbon) is real.

### Per-frame semantics of this trace

The recorder's JSON has a `header` (inputs with digests and peak-count ranges, references, roster, the
schedule as used, `driver_kw`, the strict-gate definition, totals, per-species totals, the confusion table,
the driver's counters, its cell registry, provenance) and 900 `records`, one per pushed frame, in the order
pushed: `truth` (planted species) and `src_index` (frame index in its pool); the driver's terminal outcome
`o` (`warmup_vote`, `warmup_lock`, `indexed`, `rescued_watchdog`, `miss`, …), `cell`/`cell_name` (the
registry entry the frame was attributed to), `buf` (misses held in the rescue buffer), `M` (accepted
orientation), `frac_live`/`n_inl` (the live gate's fraction and count), `wresc`, `relock`/`relock_cell`,
`resc`/`resc_cell`/`M_retro` (a retroactive rescue and the cell that rescued it), and the score `ok`, `m`,
`frac`. The builder converts these into the compact records the page plays: one terminal outcome per frame;
a rescued frame also carries the cell that rescued it and its strict verdict (`rsc`, `rok`), and the lock and
re-lock records list the frames they rescued (`rl`), so the composition chart re-colours those frames — from
grey to the rescuing cell's colour — at the moment the rescue fires, and its bars end at the driver's final
attribution (the registry's `n_frames`: 512 and 315). Each cell's printed parameters are the median of
`cell_params(M)` over every frame attributed to it, the live `M` or the rescue's `M_retro`, reordered to the
registry's standardized axes.

Animation frame *i* is the state after record *i* − 1, as above. **The GIF plays every second record at
12 frames per second — 2× real-time** — 451 frames, 80 ms each as stored (GIF delays are centiseconds), 36.1 s
per loop, so the frame counter advances by two per GIF frame; the page itself plays every record. Cumulative strict counts read off the trace: 2 after
frame 5, 96 after 146, 192 after 300, 408 after 620, 582 after 900.

### Regeneration

```bash
# 1. record (S3DF, one A100; ~/q480_fix.txt is the lysozyme q-list, digest bf3422)
PYTHONPATH=. python experiments/record_stream_replay.py \
    --input lyso=~/q480_fix.txt --input prok=experiments/prok_q.npz --ref prok=68.7,68.7,108.6,90,90,90 \
    --schedule experiments/schedules/lyso_prok_switch.json \
    --B 20 --dmin 2.0 --tol 0.002 --warmup-nbest 3 --min-inliers 10 --warmup-rescue --adaptive-relock \
    --rescue-buffer 64 --cupy --out mixed_rb64.json
# the two solo controls: the same arm on each input alone (gate480.json, prok907.json)
# 2. build the page (template outside the repo; the built HTML is committed)
GLINT_REPLAY_TEMPLATE=... python3 docs/streaming_replay/build_multicell_monitor.py \
    docs/streaming_replay/multicell_strace_a100.json --control lyso=gate480.json --control prok=prok907.json
# 3. capture and assemble (docs/streaming_replay/capture_gif.py: headless Chrome per frame, PIL GIF)
python3 docs/streaming_replay/capture_gif.py frames docs/streaming_replay/glint_streaming_multicell_real.html \
    --theme dark --last 900 --step 2 --out cap_dark --pipe-box 40,120,700,470 --stream-box 60,560,700,670
python3 docs/streaming_replay/capture_gif.py gif cap_dark docs/media/streaming_multicell_dark.gif --crop 6,0,1314,778 --width 900 --fps 12
python3 docs/streaming_replay/capture_gif.py stills docs/streaming_replay/glint_streaming_multicell_real.html \
    --theme dark --frames 0,5,146,300,620,900 --out docs/media/stills --crop 6,0,1314,778 --prefix multicell_
```
(`capture_gif.py` refuses a frame number past the page's trace — the page would wrap — and exits nonzero
when a frame or still fails to capture.)

The builder refuses any trace whose header is not this run (900 frames, 582 strict, 1 re-lock, 15 watchdog
rescues, 10 buffered rescues), and refuses to build if `build_schedule` no longer reproduces the recorded
frame order from the schedule in the header.

### Filmstrip (dark theme; light-theme twins are `multicell_light_f*.png` alongside)

| Still | Frame | What it shows |
|---|---|---|
| ![frame 0](media/stills/multicell_dark_f0000.png) | 0 | Nothing pushed: "Blind warm-up — no cell yet", one card "Discovering…", the planted schedule at scene 1 (lysozyme hold). |
| ![frame 5](media/stills/multicell_dark_f0005.png) | 5 | The lock: "locked at frame 5 · Lysozyme", card LOCKED f5 with the measured cell, "↑4 rescued at lock"; the first composition bar turns from grey to lysozyme blue for the 4 rescued votes (2 solid, 2 faint) and keeps one grey unit for the vote never rescued; strict 2 / 900. |
| ![frame 146](media/stills/multicell_dark_f0146.png) | 146 | The re-lock: "Watchdog re-lock — Proteinase K added", second card RE-LOCK f145, "re-lock ×1 · frame 146 · +Proteinase K", "↑10 rescued at re-lock", rescue buffer drained; the 10 rescued misses (records 80–143) turn orange in the bars they fell in (8 solid, 2 faint); strict 96 / 900. |
| ![frame 300](media/stills/multicell_dark_f0300.png) | 300 | ProK-major steady state: two cells, composition bars mostly orange under a mostly-orange ribbon, blue bars where the lysozyme cell claimed ProK frames; strict 192 / 900. |
| ![frame 620](media/stills/multicell_dark_f0620.png) | 620 | Lysozyme-major scene; strict 408 / 900. |
| ![frame 900](media/stills/multicell_dark_f0900.png) | 900 | The end: 900 pushed, 813 accepted, 82 missed, two cells with their final claims (427 lyso · 85 ProK; 307 ProK · 8 lyso); **strict 582 / 900**. |

### Not in the animation: the same driver on pixels

Everything above is a q-level replay. glint#200 adds pixel input to the same recorder — the detector array
through the driver's own PeakFinderV4, `peaks_to_q` under the CrystFEL geometry, prediction and integration
on the pixels, a `.stream` with reflections — and was run on the same 480 lysozyme frames (Cheetah-corrected
CSPAD data on S3DF, list order). Read against pf8's peak lists through `--peaks-in` the driver reproduces the
q-list result within a frame (334/480); on the pixels with its own finder (peakfinder8's 300-ADU threshold
as `abs_thr`, a 2-px ASIC-edge mask) it indexes and **integrates 454 of 480** and scores 362/480 at the strict
bar **on its own peak lists** — a different denominator from the pf8 lists, not a better indexer (310 frames
clear the bar under both lists, 52 only with the finder's, 24 only with pf8's). A two-species *pixel* stream
would need two species recorded on one detector; the CXIDB 45 deposit is MPCCD data, so the mixture stays at the q level.
