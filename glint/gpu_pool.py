"""Idle GPU pool manager for parallel blind rescue + diagnostics on sample changes.

When a new cell locks in the streaming driver, activate idle GPUs to:
1. Blind re-index the miss-buffer (frames waiting on the old cell) in parallel
2. Gather outcast diagnostics (spurious meter, vote distribution, etc.)
3. (Optional) pre-warm for the next sample

Non-blocking: results land asynchronously in a diagnostics queue.
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import numpy as np

try:
    import cupy as cp
    import torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False


class GPUPoolDiagnostics:
    """Async results from idle-GPU diagnostic tasks."""

    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.results = deque(maxlen=max_history)
        self.lock = threading.Lock()

    def append(self, frame_id, gpu_id, task_type, result):
        """Thread-safe append of a diagnostic result."""
        with self.lock:
            self.results.append({
                'frame_id': frame_id,
                'gpu_id': gpu_id,
                'task_type': task_type,  # 'blind_rescue' | 'spurious_meter' | 'vote_dist'
                'result': result,
                'timestamp': np.datetime64('now')
            })

    def drain(self):
        """Drain all results and return as list."""
        with self.lock:
            if not self.results:
                return []
            out = list(self.results)
            self.results.clear()
            return out


class GPUPool:
    """Manage idle GPUs for parallel tasks during streaming."""

    def __init__(self, n_gpus=None, use_threading=True, diagnostics_queue=None):
        """
        Args:
            n_gpus: number of idle GPUs available (None = auto-detect or default to 1)
            use_threading: if True, spawn tasks in threads; else run synchronously
            diagnostics_queue: optional GPUPoolDiagnostics queue to collect results
        """
        if n_gpus is None:
            n_gpus = 1  # Conservative default
        self.n_gpus = n_gpus
        self.use_threading = use_threading and n_gpus > 1
        self.diag = diagnostics_queue or GPUPoolDiagnostics()
        self.executor = ThreadPoolExecutor(max_workers=self.n_gpus - 1) if self.use_threading else None
        self.active_tasks = []

    def activate_on_new_cell(self, frame_id, miss_buffer, Mn, blind_indexer=None, spurious_meter=None):
        """
        Activate idle GPUs when a new cell locks.

        Args:
            frame_id: current frame number
            miss_buffer: list of (q_vectors, old_frame_id) tuples to re-index
            Mn: newly locked cell (3×3 matrix)
            blind_indexer: function(qs) -> list[M or None]
            spurious_meter: function(q, M) -> dict with z, wall, etc.

        Returns:
            List of future objects (if threading) or results (if sync)
        """
        if not _HAVE_TORCH or not blind_indexer:
            return []

        results = []

        # Task 1: Blind re-index miss-buffer in parallel
        if miss_buffer and self.n_gpus > 1:
            qs = [q for q, _ in miss_buffer]
            task = self._spawn_blind_rescue(frame_id, qs, Mn, blind_indexer)
            results.append(task)

        # Task 2: Gather spurious meter on rejected frames (if spurious_meter is wired)
        if miss_buffer and spurious_meter and self.n_gpus > 2:
            task = self._spawn_spurious_sweep(frame_id, miss_buffer, Mn, spurious_meter)
            results.append(task)

        # Store for later collection
        if self.use_threading:
            self.active_tasks.extend(results)

        return results

    def _spawn_blind_rescue(self, frame_id, qs, Mn, blind_indexer):
        """Blind re-index buffered misses in parallel."""
        def task():
            try:
                # Batch the blind indexing across GPUs (each GPU gets a chunk)
                chunk_size = max(1, len(qs) // self.n_gpus)
                rescued = 0
                for i in range(0, len(qs), chunk_size):
                    chunk = qs[i:i+chunk_size]
                    Ms = blind_indexer(chunk)  # returns list[M or None]
                    rescued += sum(1 for M in Ms if M is not None)

                result = {
                    'qs_count': len(qs),
                    'rescued': rescued,
                    'rescue_rate': rescued / len(qs) if qs else 0.0
                }
                self.diag.append(frame_id, 0, 'blind_rescue', result)
                return result
            except Exception as e:
                self.diag.append(frame_id, 0, 'blind_rescue', {'error': str(e)})
                return None

        if self.use_threading:
            return self.executor.submit(task)
        else:
            return task()

    def _spawn_spurious_sweep(self, frame_id, miss_buffer, Mn, spurious_meter):
        """Gather spurious meter diagnostics on rejected frames."""
        def task():
            try:
                zs = []
                wall_count = 0
                blank_count = 0

                for q, old_frame_id in miss_buffer:
                    r = spurious_meter(q, Mn)
                    z = r.get('z', None)
                    if z is not None:
                        zs.append(z)
                    if r.get('wall', False):
                        wall_count += 1
                    if r.get('blank', False):
                        blank_count += 1

                result = {
                    'z_median': np.median(zs) if zs else None,
                    'z_min': min(zs) if zs else None,
                    'z_max': max(zs) if zs else None,
                    'wall_fraction': wall_count / len(miss_buffer) if miss_buffer else 0.0,
                    'blank_fraction': blank_count / len(miss_buffer) if miss_buffer else 0.0,
                }
                self.diag.append(frame_id, 0, 'spurious_sweep', result)
                return result
            except Exception as e:
                self.diag.append(frame_id, 0, 'spurious_sweep', {'error': str(e)})
                return None

        if self.use_threading:
            return self.executor.submit(task)
        else:
            return task()

    def collect_results(self, timeout=0.1):
        """
        Non-blocking collect of completed async tasks.

        Args:
            timeout: max wait time in seconds for any single task

        Returns:
            dict summarizing results collected
        """
        if not self.use_threading:
            return {'tasks_run': 0}

        completed = []
        for future in as_completed(self.active_tasks, timeout=timeout):
            try:
                completed.append(future.result())
            except Exception as e:
                completed.append({'error': str(e)})

        # Remove completed from active list
        self.active_tasks = [f for f in self.active_tasks if not f.done()]

        return {
            'tasks_completed': len(completed),
            'tasks_pending': len(self.active_tasks),
            'results': completed
        }

    def stats(self):
        """Summary of GPU pool state."""
        return {
            'n_gpus': self.n_gpus,
            'use_threading': self.use_threading,
            'active_tasks': len(self.active_tasks),
            'diag_history': len(self.diag.results)
        }

    def shutdown(self):
        """Shutdown the thread pool."""
        if self.executor:
            self.executor.shutdown(wait=True)
