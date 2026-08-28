# Idle-GPU Activation: Parallel Blind Rescue + Diagnostics

## Overview

When the streaming driver locks a new cell (on sample change), activate idle GPUs to **parallelize** the expensive operations that would otherwise serialize:

1. **Blind re-index miss-buffer** — frames waiting on the old cell, now indexed against the new one
2. **Gather outcast diagnostics** — spurious meter, vote distribution, alias gate scores on rejected frames
3. **Inject reference validations** (periodically) — ground-truth QA check on indexing paths (e.g., lysozyme)
4. **Pre-warm next sample** (optional) — speculatively blind-search for the predicted next crystal

## Architecture

```
Streaming detector (N GPUs total)
│
├─ GPU 1 (fast path, busy)
│  └─ Known-cell indexing: 1.46 ms/frame (continuous)
│
├─ GPUs 2..N (idle pool, activated on sample change)
│  ├─ Blind re-index miss-buffer ────────────┐
│  ├─ Spurious meter sweep (null_margin) ────┼─→ Diagnostics queue
│  └─ (Optional) Pre-warm next sample ───────┘
│
└─ Async result collection (non-blocking, in-between events)
   └─ Diagnostics landing in ring buffer
```

## Key Benefits

| Scenario | GPU-1 (fast) | GPUs 2..N (idle) | Savings |
|----------|---|---|---|
| **Steady-state (known-cell repeats)** | 1.46 ms | Idle | N/A |
| **Sample change (new cell lock)** | 1.46 ms | 26 ms blind → 3 ms/GPU | Parallel speedup |
| **Diagnostics collection** | Merging | Spurious meter | Free (would be idle) |

**Real detector timing (S3DF LCLS, 10% hit rate):**
- 90% of time: GPU-1 at 1.46 ms/frame, GPUs 2..N idle
- 10% of time (new sample): GPU-1 at 1.46 ms/frame, GPUs 2..N doing blind rescue in parallel
- **Total: one GPU saturated, rest idle or helping when needed** ✓

## Reference Validation (Ground-Truth QA)

Periodically inject a known reference crystal (e.g., lysozyme) to catch drift or degradation in real-time. This validates that all indexing paths agree on a known ground-truth.

**What's checked:**

```
Reference frame (every N events):
│
├─ Blind path: should find reference cell (hit rate 95%+)
│  └─ Flag if drop in hit rate → peakfinder issue
│
├─ Known-cell path: should hit reference (100%)
│  └─ Inlier count should be >100 (clean signal)
│
├─ Spurious meter: should score very high z (>3, ideally >5)
│  └─ Flag if z drops → contamination or signal issue
│
├─ Geometry validation: radial profile correlation (via fast CUDA integration)
│  └─ Profile correlation > 0.95 = stable; < 0.90 = drift signal
│     Needs the raw frame image (`image=`) — a radial profile cannot be built
│     from the peak list alone. Without it the check is skipped, not failed.
│
└─ Consensus validation: all four paths should agree
   └─ Flag if any path disagrees → system instability
```

**Output (per reference frame):**

```python
{
    'frame_type': 'reference',
    'blind': {'found': True, 'score': 0.95, 'matches_ref': True},
    'known_cell': {'hit': True, 'inliers': 156},
    'spurious': {'z': 5.7, 'wall': False, 'blank': False},
    'geometry': {
        'profile_correlation': 0.987,
        # 'error' = configured but could not run; fails all_agree (never silent)
        'stability': 'stable'  # 'stable' | 'monitor' | 'drift' | 'seeded' | 'error'
    },
    'validation': {'all_agree': True}
}
```

**Dashboard metrics (accumulated):**

- % validations where all paths agree (target: 100%)
- Blind hit rate on reference (target: 95%+)
- Spurious z-score distribution (target: median > 5)
- Geometry profile correlation (target: median > 0.95, min > 0.90)
- Geometry drift count (target: 0; if > 0 → alert operator immediately)
- Trend alert: if any metric degrades → flag operator for inspection

## API

### GPUPool

```python
from glint.gpu_pool import GPUPool, GPUPoolDiagnostics

# Create pool with reference validation + geometry monitoring
diag_queue = GPUPoolDiagnostics(max_history=1000)
reference_cell = cell_to_Ar(79, 79, 38, 90, 90, 90)  # lysozyme as QA check
detector_geometry = detector.geometry  # or q_per_pixel array from your geometry
pool = GPUPool(
    n_gpus=4,
    use_threading=True,
    diagnostics_queue=diag_queue,
    reference_cell=reference_cell,
    reference_every=100,  # Inject every 100 frames
    detector_geometry=detector_geometry  # For fast radial integration validation
)

# On new-cell lock (in StreamDriver._watchdog or similar)
pool.activate_on_new_cell(
    frame_id=frame_number,
    miss_buffer=[(q1, frame_id_1), (q2, frame_id_2), ...],  # buffered frames
    Mn=locked_cell,  # 3×3 matrix
    blind_indexer=lambda qs: [index_blind_nbest(q, 1) for q in qs],
    spurious_meter=lambda q, M: null_margin(q, M, n_null=256, K=70400, rng=rng)
)

# Periodically collect results (non-blocking, ~1 ms overhead)
while streaming:
    # In the main loop, after every frame or batch:
    results = pool.collect_results(timeout=0.01)
    if results['tasks_completed'] > 0:
        print(f"Collected {results['tasks_completed']} idle-GPU results")
        # Results also land in diag_queue for persistent telemetry

# Periodically inject reference validation (non-blocking)
while streaming:
    if pool.should_inject_reference():
        ref_task = pool.validate_reference(
            q,  # frame's q-vectors
            blind_indexer=lambda qs: [...],
            known_indexer=lambda qs, Mc: [...],
            spurious_meter=lambda q, M: null_margin(...),
            same_lattice_fn=lambda M1, M2: [...],
            image=frame,  # raw (H, W) detector image -- REQUIRED for the geometry check
        )
        # Task runs async; results land in diag_queue

# Get QA summary (non-blocking)
qa_stats = pool.reference_stats()
print(f"Reference validation: {qa_stats['pass_rate']:.0%} pass ({qa_stats['count']} tests)")
print(f"  Blind hit rate: {qa_stats['blind_hit_rate']:.0%}")
print(f"  Known-cell hit rate: {qa_stats['known_cell_hit_rate']:.0%}")
print(f"  Spurious z: median {qa_stats['spurious_z_median']:.1f}")
if 'geometry_profile_correlation_median' in qa_stats:
    print(f"  Geometry profile correlation: {qa_stats['geometry_profile_correlation_median']:.3f}")
    print(f"    Stable: {qa_stats['geometry_stable_count']}, Drift: {qa_stats['geometry_drift_count']}")
    if qa_stats['geometry_drift_count'] > 0:
        print("    ⚠️  ALERT: Geometry drift detected!")

# Drain diagnostics at end-of-run
diagnostics = diag_queue.drain()  # list of {frame_id, task_type, result, timestamp}
```

## Integration into StreamDriver

### Add GPU pool with reference validation

In `glint/stream_driver.py`, add to `__init__`:

```python
from glint.gpu_pool import GPUPool, GPUPoolDiagnostics

class StreamDriver:
    def __init__(self, ..., n_idle_gpus=0, reference_cell=None, reference_every=100, 
                 detector_geometry=None, ...):
        # ... existing init ...
        if n_idle_gpus > 0:
            self._diag_queue = GPUPoolDiagnostics()
            self._gpu_pool = GPUPool(
                n_gpus=n_idle_gpus + 1,
                diagnostics_queue=self._diag_queue,
                reference_cell=reference_cell,  # e.g., cell_to_Ar(79, 79, 38, 90, 90, 90)
                reference_every=reference_every,  # inject every N frames
                detector_geometry=detector_geometry  # for geometry validation
            )
        else:
            self._gpu_pool = None
```

### Inject reference validations in main loop

In the main loop (e.g., after `push()` and periodically):

```python
# Check if it's time to inject reference
if self._gpu_pool and self._gpu_pool.should_inject_reference():
    self._gpu_pool.validate_reference(
        q,  # current frame
        blind_indexer=lambda qs: [index_blind_nbest(q, self.warmup_nbest) for q in qs],
        known_indexer=lambda qs, Mc: [self._known_index(q, Mc) for q in qs],
        spurious_meter=lambda q, M: null_margin(q, M, n_null=256, K=70400, rng=self._rng),
        same_lattice_fn=same_lattice
    )
```

### Check QA stats periodically

```python
# Per-minute or per-1000-frames, log QA stats
if self._gpu_pool and frame_count % 1000 == 0:
    qa = self._gpu_pool.reference_stats()
    logger.info(f"QA: {qa['pass_rate']:.0%} validation pass, "
                f"blind {qa['blind_hit_rate']:.0%}, "
                f"z={qa['spurious_z_median']:.1f}")
    if qa['pass_rate'] < 0.9:
        logger.warning("Reference validation pass rate dropped — check system health")
```

In `_watchdog()` (where new-cell locks):

```python
# After Mn locks and before returning from _watchdog:
if self._gpu_pool and self._missbuf:  # if idle-GPU pool is active and buffer has data
    self._gpu_pool.activate_on_new_cell(
        frame_id=self.n_pushed,
        miss_buffer=list(self._missbuf),
        Mn=Mn,
        blind_indexer=lambda qs: [
            index_blind_nbest(q, self.warmup_nbest) for q in qs
        ],
        spurious_meter=lambda q, M: null_margin(q, M, n_null=256, K=70400, rng=self._rng)
    )
```

In `flush()` or periodically:

```python
# Non-blocking collect of diagnostics
if self._gpu_pool:
    result = self._gpu_pool.collect_results(timeout=0.01)
    # Diagnostics auto-land in self._diag_queue
```

## Output: Diagnostics Queue

Each entry in the queue has:

```python
{
    'frame_id': int,
    'gpu_id': int,
    'task_type': str,  # 'blind_rescue' | 'spurious_sweep' | ...
    'result': dict,
    'timestamp': numpy.datetime64
}
```

### `blind_rescue` result:
```python
{
    'qs_count': 50,  # frames in miss-buffer
    'rescued': 35,   # blind successfully indexed
    'rescue_rate': 0.7
}
```

### `spurious_sweep` result:
```python
{
    'z_median': 2.5,       # spurious-meter z-score (null_margin)
    'z_min': 0.3,
    'z_max': 5.2,
    'wall_fraction': 0.4,  # % above spurious ceiling
    'blank_fraction': 0.1  # % too few peaks
}
```

## Tuning

### Threading vs sync
- `use_threading=True` (default): spawn tasks in background, non-blocking
- `use_threading=False`: run sequentially in main thread (for debugging)

### GPU count
- Start with `n_idle_gpus = (n_gpus_total - 1)` — reserve 1 for the fast path
- Adjust down if GPU memory becomes contended (miss-buffer or diagnostics exceed budget)

### Buffer size
- Default: `rescue_buffer=100` frames in `StreamDriver`
- Adjust based on detector trigger rate and sample-change frequency
- Larger buffer = more data for parallel rescue, but higher memory cost

## Example Run

```bash
# On S3DF with 4 A100s available, use 3 for idle-GPU work:
python ~/git/glint/experiments/stream_with_idle_gpus.py \
  --npz cxidb_sw120.npz \
  --cell 79 79 38 90 90 90 \
  --n-gpus 3 \
  --max-buffer 50
```

Output:
```
Streaming simulation: 120 frames, 3 GPUs available
Reference cell: (79.0, 79.0, 38.0, 90.0, 90.0, 90.0)

[Frame 30] Sample change → NEW CELL LOCK
  Miss-buffer size: 12
  Spawned 2 async task(s)

[Frame 40] Idle-GPU results collected:
  Completed: 2
  Pending: 0
    Blind rescue: 9/12 frames
    Spurious z: median=2.31, wall=25.0%
...
```

## Performance Expectations

**Miss-buffer blind rescue** (N-1 GPUs, data-parallel):
- Single GPU: 26 ms/frame
- 2 GPUs: ~13-15 ms/frame (linear scale-out, minus overhead)
- 4 GPUs: ~6-8 ms/frame
- Consensus barrier (one-time): ~165 ms

**Spurious meter sweep** (threaded):
- ~10-20 ms for 50-frame buffer (depends on peak count)
- Negligible if I/O-bound (CPUs spinning, not saturated)

**Overhead of idle-GPU activation**:
- Spawn: <1 ms (just thread creation)
- Collection: ~1-2 ms per poll (Python list ops)
- **Net**: only pays if samples change frequently

## Future Enhancements

1. **Pre-warm next sample** — if sample changer is known (robotic arm, etc.), speculatively blind-search for the predicted next crystal in parallel
2. **GPU-Buerger reduction** — move the CPU same_lattice loop to GPU (currently 25% of known-cell path)
3. **CUDA graph chaining** — overlap known-cell and blind rescue via CUDA graphs

---

**Status:** Experimental. Wiring into StreamDriver pending user approval. Test harness: `experiments/stream_with_idle_gpus.py`.
