# StreamDriver — the options, by topic

`glint.stream_driver.StreamDriver` is the shipped streaming indexer: frames are pushed one at a time, the
driver discovers the cell (or is given one), indexes in batches on the GPU, integrates against the pixels
still resident on the device, and keeps a running merge. Most of what it can do is opt-in, and the
constructor has grown one keyword per measured mechanism, so this page is the map: every constructor
option, grouped by what it is for, with its default, the pull request or note that measured it, and the
flag of `experiments/record_stream_replay.py` that sets it (the recorder is how the driver is run on a
q-list or a pixel set without writing code). `experiments/test_stream_driver_options_doc.py` checks that
the tables below name every constructor option and nothing else, and that their displayed defaults match
the code, so this page cannot drift silently.

Read alongside [`streaming_replay.md`](streaming_replay.md) (two recorded runs of the driver, with
provenance) and the class docstring, which is the fuller treatment of each mechanism.

## Construct, feed, read

```python
import numpy as np
from glint.stream_driver import StreamDriver
d = StreamDriver(None, panels, clen_m, wavelength_A, shape,      # Mc=None: discover the cell
                 dtype=np.float32,      # calibrated frames (det.calib); the uint16 default is for raw ADU
                 B=64, dmin=2.0, min_inliers=10, warmup_rescue=True, adaptive_relock=True)
for frame in frames:            # host or device array of `shape`; or push_peaks(fs, ss) / push_q(q)
    d.push(frame)
d.flush()                       # index the resident batch, integrate, merge
s = d.stats()                   # counters, cells, merge figures of merit, effort log
```

Three ways in, one driver: `push(frame)` runs the device peak finder on the pixels and integrates;
`push_peaks(fs, ss)` takes a peak list in data-array pixels (the DRP reducer-to-indexer path, no pixels
in this process); `push_q(q)` takes reciprocal vectors in 1/Å. The last two are index-only: registered,
counted, written to the `.stream` as a crystal with zero reflections, never integrated. `flush()` at the
end of a run; `stats()` afterwards or at any time.

With `events=True` (or an `on_event` callback) the driver emits one record per frame outcome:
`ev` (arrival index), `at` (frames pushed when emitted), `outcome`, `cell`, `cell_name`, `n_active`,
`buffer`, and for indexed frames `n_peaks`, `n_inl`, `frac`, `M`; `inl_by_cell` under `assign="best"` and
`second_lattice` under `per_lattice`. Outcomes: `blank`, `warmup_vote`, `warmup_lock`, `indexed`,
`escalated`, `rescued_watchdog`, `rescued_cascade`, `rescued_per_lattice`, `miss`, `gate_rejected`; the
retroactive `rescued_warmup` and `rescued_relock` re-label a frame after the fact; `relock`, `integrated`
and `effort` are markers. `d.registry` lists the active cells (`id`, `name`, `source` given / warmup /
relock, `locked_at`, `lock_generation`, `n_frames`, `last_seen`); `recent_share()` is each cell's share of
the last `cell_window` attributed frames.

## The tables

Columns: the option and its default; what it does; where it was measured (a pull request of
`slac-lcls/glint` unless stated); the recorder flag that sets it, or — when only the constructor does.

### Construction and input

| option | default | what it does | measured / introduced | recorder |
|---|---|---|---|---|
| `Mc` | required | real-space basis (3×3) of a known cell, or `None` to discover the cell blind from the first frames | #19, #54 | the driver is always cold-started (`Mc=None`); `--ref` supplies reference cells for scoring and the roster only |
| `panels` | required | CrystFEL panel geometry (`glint.geom.parse_geom`) | #19 | `--geom` for pixel inputs; a synthetic flat panel for q lists |
| `clen_m` | required | camera length, m | #19 | `--clen` overrides the geometry's |
| `wavelength_A` | required | wavelength, Å; one value for the run | #19 | `--wavelength` |
| `shape` | required | detector array shape the ring is preallocated for | #19 | from the geometry |
| `dtype` | `uint16` | ring dtype; it must hold the pushed frames losslessly, so use `np.float32` for calibrated frames — `push()` raises `TypeError` on a frame the ring would truncate, clamp or wrap (any float or signed frame into the uint16 default) | #19 | — (pixel inputs are read as float32 into a float32 ring) |
| `mask` | `None` | good-pixel mask for the peak finder (`True` = good) | #19 | `--mask`, `--edge-mask N` |
| `use_gpu` | `True` | CuPy device path; `False` is the numpy path the CI tests run | #19 | `--cupy` turns it on |
| `pf_kw` | `None` | settings of the device peak finder (`glint.peakfinder_v4.PeakFinderV4`), e.g. `abs_thr`, `son_min`, `min_pix` | #19 | `--pf-kw JSON` |
| `per_panel_finder` | `False` | on a multi-panel slab, one peak finder per panel rectangle (`glint.peakfinder_v4.PerPanelFinder`), so no background ring, local-max window or label reaches across a panel seam into rows that are elsewhere in the lab. Off because each panel costs ~1 ms per frame on an A100: 64-panel CSPAD end to end 73.0 against 8.6 ms/frame | review r2 s6-04; GPU job 39724839 | `--per-panel-finder` |
| `min_peaks` | `6` | a frame with fewer peaks is a `blank`: counted, never indexed | #19 | — |
| `B` | `64` | ring size and index batch: frames are indexed when `B` are queued. Not a saturation point — the fused known-cell engine costs 0.17 ms/hit at B=120 (fp64, one A100) and more per hit at smaller batches | #19; #165 | `--B` (the replays use 20) |
| `hits_only` | `False` | a frame with too few peaks gives its ring slot straight back, so `B` counts hits and the batch is the size the engine was timed at | #212 | — |

### Finding the cell: blind warm-up and the lock

With `Mc=None` the first frames are indexed blind, one at a time, and vote in a running cross-frame consensus
(`glint.running_consensus`); the cell locks when the vote is unambiguous and the driver switches to the batched
known-cell path. The lock takes about six frames on cxidb-17 (median over 400 arrival orders; 90th percentile 12).

| option | default | what it does | measured / introduced | recorder |
|---|---|---|---|---|
| `warmup_nbest` | `3` | cells each warm-up frame contributes to the vote (blind N-best) | #54 | `--warmup-nbest` |
| `lock_support` | `3` | votes the leading cell needs | #54 | — |
| `lock_gap` | `2` | lead over the runner-up it needs | #54 | — |
| `adaptive_gap` | `True` | widen the gap requirement while the pool is small | #54 | — |
| `lock_frac` | `0.02` | pool-keyed acceptance: the leader must hold this fraction of the pool once the pool exceeds `lock_pool_switch` | #122 | — |
| `lock_lead` | `1.5` | ...and lead the runner-up by this ratio | #122 | — |
| `lock_pool_switch` | `72` | pool size from which the fraction and lead rules apply. On cxidb-17 the gate changes no lock in 400 arrival orders; on unrefined mfxx49820 geometry the bare gap rule locked a wrong lattice in 194 of 400 orders and the gate leaves 2 | #122 | — |
| `warm_topk` | `32` | warm-up triage: rank the start-up frames by peak count and spend blind indexing on the top `warm_topk` | #56 | — |
| `warm_floor` | `1` | peak-count floor of that triage (16 refused real MFX data) | #56 | — |
| `fanout` | `None` | callable that blind-indexes many frames at once (`glint.warmup_batch.mpi_fanout`); the serial default is bit-identical. Four ranks cut the warm-up 3.1× on 32 frames (job 37198405) | #56 | — |
| `alias_gate` | `None` | lock-time confirmation that the leader is not a small-index derivative lattice (`glint.alias_gate.AliasGate`). Off in every published result | #179 fixed its enumeration | — |
| `lock_probe` | `False` | on every relock, measure the new cell's overlap against a random-orientation null (`glint.spurious_meter.null_margin`; `stats()["lock_z"]`) | commit 7c05519 | `--lock-probe` |
| `probe_null` | `64` | random orientations that null draws | 7c05519 | — |
| `lock_min_z` | `None` | refuse a relock below this z | 7c05519 | — |

### The live gate

A frame is accepted under a cell when its near-integer inliers pass both a count and a fraction of its peaks
(`_fits`). The gate is deliberately looser than the paper's strict bar (same lattice, ≥25 % of peaks, ≥10
reflections); yields quoted from `indexed` are live-gate yields, and the recorder's strict column is the one to
compare. On azimuth-scrambled copies of the 480 cxidb-17 frames, the count-plus-fraction gate accepts 211
(44%), concentrated among the sparse frames; `null_floor=` adds a per-peak-count floor fitted on that null,
which brings the chance accepts to 2 of 480 while keeping 305 of the 327 strict frames (glint#214).

| option | default | what it does | measured / introduced | recorder |
|---|---|---|---|---|
| `min_inliers` | `0` → `min_peaks` | minimum inlier count; the published arms use 10 | #54 | `--min-inliers` |
| `min_inlier_frac` | `0.15` | minimum inlier fraction of the frame's peaks; 0.15 is the smallest value that refused every wrong-cell frame in the calibration (n=16, synthetic), costing 4 of 115 real frames | commit 907c057; #170 pins it | `--min-inlier-frac` |
| `null_floor` | `None` | a third bar on the live gate, `(a, b)` or `(a, b, c)`: the registration must explain `n_inl ≥ a·n + b + c·√n` of the frame's `n` peaks, the shape of a best-of-K null. `NULL_FLOOR_CXIDB17 = (0.0224, 5.32, 1.211)` is fitted on cxidb-17 at the 99th percentile of 15,360 scrambled copies and is specific to that peak finder, detector, cell family, HKL tolerance, and search depth: re-fit with `experiments/live_gate_null.py` before relying on it elsewhere. Applies wherever the live gate does (cascade, watchdog, warm-up rescue); refusals in `n_null_floor_refused` | #214; [`RESULTS_live_gate_null.md`](../experiments/RESULTS_live_gate_null.md) | `--null-floor a,b[,c]` or `--null-floor cxidb17` |
| `tol` | `0.002` | excitation-error window of the prediction (1/Å): which reflections count as on the Ewald sphere for integration | #19 | `--tol` |

### Recovering the misses, in execution order

Once locked the driver runs known-cell only; a frame that fits no active cell is a miss. The optional recoveries
take it in this order (`flush()`): per-lattice scoring of double hits, the deep search of `effort=`, the retry
cascade, then the miss buffer and blind watchdog of `adaptive_relock`. The rows follow that order; the last three
belong to the same family but act elsewhere — `warmup_rescue` on the pre-lock frames, `rescue_pixels` on what the two
retroactive rescues can integrate, `double_hit` on frames already accepted. Each is off by default, so the base path
stays byte-identical.

| option | default | what it does | measured / introduced | recorder |
|---|---|---|---|---|
| `per_lattice` | `False` | score the stronger of two lattices against the peaks the other does not claim: a frame failing the gate under every cell is searched for a second lattice in the residual (`rescued_per_lattice`), and an accepted frame below `per_lattice_below` gets a per-lattice QC fraction | #206 | `--per-lattice` |
| `per_lattice_below` | `0.25` | lattice-1 share below which an accepted frame is searched | #206 | `--per-lattice-below` |
| `effort` | `None` | adaptive effort: the known-cell search depth follows the hit rate, and the spare budget buys a chance-controlled deep search on the misses (`escalated`). Settings below | #213; [`RESULTS_stream_effort_a100.md`](../experiments/RESULTS_stream_effort_a100.md) | `--effort JSON` |
| `retry_cascade` | `False` | on a miss, run the measured arm union before the miss path: blind N-best, then the per-frame known-cell indexer. Streaming with the retry indexes 365 of 480 against offline's 357 (`retry_rate_of480`); at the shipped live gate it fires on few frames, and it pays with the strict bar as the live gate (`min_inliers=10, min_inlier_frac=0.25`, the constructor comment's table) | #145; `check_numbers.py --facts` | `--retry-cascade` |
| `retry_nbest` | `None` → 10 | N-best depth of that blind arm | #145 | — |
| `adaptive_relock` | `False` | a blind watchdog on the misses: their votes add a second active cell when one recurs (a sample change or a mixture), and a miss whose own candidates fit an active cell is rescued individually (`rescued_watchdog`) | #54; two-species replay in [`streaming_replay.md`](streaming_replay.md) | `--adaptive-relock` |
| `rescue_buffer` | `0` | with `adaptive_relock`: keep the last N misses' q and re-index them against a newly locked cell (`rescued_relock`) | commit 401cd98; in the two-species replay the 10 buffered misses were re-indexed against the new cell, 8 strict | `--rescue-buffer` |
| `warmup_rescue` | `False` | keep the warm-up frames' q and re-index them against the cell the instant it locks (index-only unless `rescue_pixels`) | commit 041dfa3; published arm: 323 → 331 of 480 with `adaptive_relock` | `--warmup-rescue` |
| `rescue_pixels` | `0` | keep up to N frames' pixels on the device so the warm-up and relock rescues integrate what they recover, not only count it | #212 | — |
| `double_hit` | `False` | after each accepted frame, deflate its peaks and look for a second lattice of the same cell ≥15° away; the gated rule is reported with its own 1-in-16 scrambled null | #56; #207 | `--double-hit` |

### More than one cell

| option | default | what it does | measured / introduced | recorder |
|---|---|---|---|---|
| `roster` | `None` | `{name: cell}` (six parameters or a 3×3 basis): a cell the driver finds is named from the roster if it matches one. The roster never hands the indexer a cell | #199 | `--ref NAME=a,b,c,al,be,ga` |
| `events` | `False` | keep the per-frame event records in `d.events` | #199 | always on |
| `on_event` | `None` | callback per event record (turns emission on) | #199 | — |
| `cell_window` | `200` | window, in attributed frames, of `recent_share()` | #199 | `--cell-window` |
| `assign` | `"first"` | which active cell takes a frame that fits more than one: the first-fit cascade (every published number), or `"best"`, which indexes every frame against every active cell and lets a challenger take it when it explains at least `max(assign_margin, assign_margin_frac × n_peaks)` more peaks. On the two-species stream: 582 → 650 of 900 strict | #204 | `--assign` |
| `assign_margin` | `8` | that margin, in peaks | #204 | `--assign-margin` |
| `assign_margin_frac` | `0.05` | ...or this fraction of the frame's peaks, whichever is larger | #204 | `--assign-margin-frac` |

### Integration and the running merge

| option | default | what it does | measured / introduced | recorder |
|---|---|---|---|---|
| `dmin` | `2.0` | resolution limit (Å) of prediction and of the merge's theoretical-unique count | #19 | `--dmin` |
| `half` | `3` | integration box half-width, px | #19 | — |
| `gap` | `2` | gap between the box and the background annulus, px | #19 | — |
| `ring` | `3` | annulus width, px | #19 | — |
| `bg_mode` | `"clipmean"` | annulus background estimator: `clipmean`, `median`, `mean` | commit ea9a7b9 (#142 review) | — |
| `snr_bins` | `(-inf, 0, 1, 2, 3, 5)` | I/σ floors the running merge is bucketed at; the first is −inf so I ≤ 0 is kept (on the same merged frames, `stats(thr=0.0)` gives the old I > 0 CC½, CC*, R_split — to a few ulp, rows are created in a different order — and unique, redundancy and completeness exactly; `measurements` is not thresholded and counts every merged row, the I > 0 count is unique × redundancy at `thr=0.0`) | #19; −inf floor: review r2 s1-05 | — |
| `laue` | `None` | Laue class the running merge (completeness, CC½, CC*, R_split) is accumulated under; derived from `stream_symmetry` when unset, else `4/mmm` | #186 | `--laue` |
| `ops` | `None` | explicit operator list instead of `laue` | #186 | — |
| `geom_refine` | `False` | pool predicted-vs-observed residuals into a running (clen, beam-shift) correction, reported in `stats()["geom_correction"]`. Diagnostic: it is not fed back into indexing | #55 | `--geom-refine` |
| `geom_refine_kw` | `None` | settings of `glint.geom_refine.GeomRefiner` | #55 | `--geom-refine-kw` |
| `qc_frac_threshold` | `None` | flag (never drop) an integrated frame whose matched fraction is below this; `low_conf_frames`, `n_low_confidence` | commit 87311cd | — |

### Writing a CrystFEL stream

| option | default | what it does | measured / introduced | recorder |
|---|---|---|---|---|
| `stream_out` | `None` | path: append one `.stream` chunk per integrated (or index-only) frame, stamped with the state in force for that frame | commit 0a72379 | `--stream-out` |
| `stream_geom_text` | `None` | geometry block copied into the stream header | 0a72379 | — |
| `stream_image` | `"glint.cxi"` | placeholder image name when `push(..., src=)` gives none | 0a72379 | — |
| `stream_symmetry` | `None` | `{lattice_type, centering, unique_axis}` written to every chunk; also fixes the merge class unless `laue` is given | 0a72379; #186 | `--stream-symmetry` |
| `stream_peaks` | `None` | also write the observed peaks: `"flagged"` (frames below `qc_frac_threshold`) or `"all"`, so a flagged chunk can be re-indexed offline | commit 0a07dcd | — |

### The effort settings

`effort=dict(rate_hz=..., ...)`. The GPU time each hit may have is `1000 · n_gpu / (rate_hz · hit_rate)` ms less
`overhead_ms`; the fast path then runs every hit at the deepest tier whose per-frame cost fits, and when what is
left also covers the misses' deep searches (`miss_frac · (1 + k_null)` top-tier searches per frame) the misses of a
flush get `escalate_batch` at the top tier with its azimuth-scrambled null (glint#211). Decisions are taken at
flush boundaries only, every `every` flushes, and logged: `stats()["effort"]["log"]` and one `effort` event per
change (`tier`, `deep`, `budget_ms`, `hit_est`, `miss_frac`, `n_cells`; `ev` is the first frame the decision applied
to). The first decision uses the priors; with several active cells the costs are multiplied by their number.

| setting | default | meaning |
|---|---|---|
| `rate_hz` | required | frames per second the driver must keep up with |
| `n_gpu` | `1.0` | share of a GPU left to the indexer after peak finding |
| `k_null` | `32` | scrambled copies a deep-search fit must beat (glint#211's setting: 0 null accepts) |
| `round_copies` | `8` | copies searched per round of the batched null |
| `every` | `1` | decide every N flushes |
| `hit_window` | `256` | frames the hit rate is estimated over |
| `miss_window` | `4` | flushes the miss fraction is estimated over |
| `hit_prior` | `1.0` | hit rate assumed before the window fills (every frame a hit, every frame a miss: the conservative start) |
| `tiers` | four tiers | `(topa, nc, full_grid, ms_per_frame)` per tier; the defaults are the controller's cost model at B=120 on one A100 (`EFFORT_TIERS`), so a run at another batch size passes tiers measured there |
| `seed` | `20260926` | seed of the null's scrambles |
| `deep_budget` | `12000` | chunking of the batched deep search: frames × peaks per device call stay under this (`index_known_deep_batch`) |
| `overhead_ms` | `0.0` | per-hit GPU time that is not search (integration when the driver integrates) |

Measured on one A100 at B=20 (a check of the mechanism, not of the B=120 cost table; job 39344280): the
35 kHz arm is decision-identical to `effort=None`; the 2 kHz arm runs tier 2 and switches the deep search on and
off with the miss fraction; the 120 Hz arm runs the top tier. Strict counts 333 → 360 → 376 of 480. On a
lattice-free null the live gate's own chance acceptance was the same at every tier, and the deep search was not
reached at those budgets (it is a rare-miss feature under the default tiers). Details and the reading in the results note linked above.

## Reading the counters

`stats()` returns the merge figures of merit at an I/σ floor (`completeness`, `cc_half`, `cc_star`, `rsplit`,
`unique`, `theoretical_unique`, `laue`) and the counters, among them `locked`, `pushed`, `indexed`, `integrated`,
`locked_after`, `consensus_support`, `consensus_members`, `gate_refused`; when the floor is configured, `null_floor` and
`n_null_floor_refused` (registration attempts, so one frame may count more than once); the rescues (`n_warmup_rescued`, `n_watchdog_rescued`, `n_rescued`, `n_cascade_retried`,
`n_cascade_rescued`, `n_cascade_by_arm`, `n_per_lattice_*`); the cells (`n_cells`, `cells`, `extra_cells`,
`n_relock`, `lock_z`); the diagnostics (`geom_correction`, `n_low_confidence`, `double_hit_rate`, the null rates);
the stream (`stream_out`, `stream_chunks`, `stream_indexed`); the ring (`pixels_held`, `pixels_evicted`); and
`effort` (tier in force, budget, estimates, `fast_by_tier`, deep-search counts, the log).

## Running it

- `experiments/record_stream_replay.py` — the recorder: q lists, `.npz`, or pixels under a CrystFEL geometry;
  a planted schedule for mixtures; strict scoring per species; `--expect` as a regression gate. Its module
  docstring is the manual.
- `experiments/test_streamdriver_vs_offline.py` — the driver cold-started on the 120 real frames against the
  offline known-cell rate (GPU).
- `experiments/test_stream_*.py`, `test_cell_registry.py`, `test_per_lattice.py`, `test_lock_gate_wiring.py` —
  the CPU suites, fit-oracle indexers, run by CI.
- `experiments/effort_gpu.sbatch` — the A100 job behind the effort numbers, HEAD-pinned.
