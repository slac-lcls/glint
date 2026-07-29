# Idle-GPU Activation: Parallel Blind Rescue + Diagnostics

## Overview

When the streaming driver locks a new cell (on sample change), activate idle GPUs to **parallelize** the expensive operations that would otherwise serialize:

1. **Blind re-index miss-buffer** — frames waiting on the old cell, now indexed against the new one
2. **Gather outcast diagnostics** — spurious meter, vote distribution, alias gate scores on rejected frames
3. **Pre-warm next sample** (optional) — speculatively blind-search for the predicted next crystal

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

## API

### GPUPool

```python
from glint.gpu_pool import GPUPool, GPUPoolDiagnostics

# Create pool (threaded, non-blocking)
diag_queue = GPUPoolDiagnostics(max_history=1000)
pool = GPUPool(n_gpus=4, use_threading=True, diagnostics_queue=diag_queue)

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

# Drain diagnostics at end-of-run
diagnostics = diag_queue.drain()  # list of {frame_id, task_type, result, timestamp}
```

## Integration into StreamDriver

In `glint/stream_driver.py`, add to `__init__`:

```python
from glint.gpu_pool import GPUPool, GPUPoolDiagnostics

class StreamDriver:
    def __init__(self, ..., n_idle_gpus=0, ...):
        # ... existing init ...
        if n_idle_gpus > 0:
            self._diag_queue = GPUPoolDiagnostics()
            self._gpu_pool = GPUPool(n_gpus=n_idle_gpus + 1, diagnostics_queue=self._diag_queue)
        else:
            self._gpu_pool = None
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
