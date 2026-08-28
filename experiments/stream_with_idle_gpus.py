#!/usr/bin/env python3
"""Example: streaming driver with idle-GPU parallel blind rescue + diagnostics.

Usage:
    python stream_with_idle_gpus.py --npz cxidb_120.npz --n-gpus 4 --cell 79 79 38 90 90 90

This example shows:
1. Creating a GPUPool for idle-GPU management
2. Spawning parallel blind rescue + diagnostics on new-cell lock
3. Non-blocking collection of results
"""
import argparse
import numpy as np
from collections import deque

from glint.glint_fast import index_blind_nbest, cell_to_Ar
from glint.spurious_meter import null_margin
from glint.gpu_pool import GPUPool, GPUPoolDiagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=str, required=True, help='cxidb .npz file')
    parser.add_argument('--cell', type=float, nargs=6, required=True, help='reference cell a b c α β γ')
    parser.add_argument('--n-gpus', type=int, default=4, help='number of idle GPUs available')
    parser.add_argument('--max-buffer', type=int, default=50, help='max miss-buffer size')
    args = parser.parse_args()

    # Load data
    d = np.load(args.npz, allow_pickle=True)
    n = int(d['n'])
    frames = [np.asarray(d[f's{i}'], float) for i in range(n)]
    Mc = cell_to_Ar(*args.cell)

    # Setup GPU pool with reference validation (inject lysozyme every 30 frames as QA check)
    diag_queue = GPUPoolDiagnostics(max_history=100)
    reference_cell = Mc  # Use the same cell as reference (normally would be lysozyme)
    gpu_pool = GPUPool(
        n_gpus=args.n_gpus,
        use_threading=True,
        diagnostics_queue=diag_queue,
        reference_cell=reference_cell,
        reference_every=30  # Inject every 30 frames
    )

    # Miss-buffer to simulate frames waiting on an old cell
    miss_buffer = deque(maxlen=args.max_buffer)

    print(f"Streaming simulation: {n} frames, {args.n_gpus} GPUs available")
    print(f"Reference cell: {args.cell}")
    print()

    # Simulate streaming: blind index frames, accumulate misses, trigger on sample change
    sample_change_every = 30
    rng = np.random.default_rng(42)

    indexed_count = 0
    rescued_count = 0

    for i, q in enumerate(frames):
        # Check if it's time to inject a reference validation frame
        if gpu_pool.should_inject_reference():
            print(f"\n[Frame {i}] Reference validation → injecting lysozyme")
            ref_task = gpu_pool.validate_reference(
                q,
                blind_indexer=lambda qs: [
                    index_blind_nbest(q, 1)[0][0] if index_blind_nbest(q, 1) else None
                    for q in qs
                ],
                known_indexer=lambda qs, Mc: [Mc for _ in qs],  # stub: always "hit" reference cell
                spurious_meter=lambda q, M: null_margin(q, M, n_null=256, K=70400, rng=rng),
                same_lattice_fn=lambda M1, M2: True  # stub: always same
            )
            print(f"  Reference validation spawned")

        # Simulate: every N frames, trigger a "sample change" (new cell lock)
        if i > 0 and i % sample_change_every == 0:
            print(f"\n[Frame {i}] Sample change → NEW CELL LOCK")
            print(f"  Miss-buffer size: {len(miss_buffer)}")

            # Activate idle GPUs for parallel rescue + diagnostics
            if len(miss_buffer) > 0:
                tasks = gpu_pool.activate_on_new_cell(
                    frame_id=i,
                    miss_buffer=list(miss_buffer),
                    Mn=Mc,
                    blind_indexer=lambda qs: [
                        index_blind_nbest(q, 1)[0][0] if index_blind_nbest(q, 1) else None
                        for q in qs
                    ],
                    spurious_meter=lambda q, M: null_margin(q, M, n_null=256, K=70400, rng=rng)
                )
                print(f"  Spawned {len(tasks)} async task(s)")
                miss_buffer.clear()

        # Blind index this frame
        nb = index_blind_nbest(q, 1)
        if nb:
            M = np.asarray(nb[0][0], float)
            # Simulate: 70% hit the current cell, 30% miss
            if rng.random() < 0.7:
                indexed_count += 1
            else:
                # Miss → buffer for later rescue
                miss_buffer.append((q, i))
        else:
            miss_buffer.append((q, i))

        # Periodically collect results from idle GPUs (non-blocking)
        if i % 10 == 0:
            collect_result = gpu_pool.collect_results(timeout=0.01)
            if collect_result['tasks_completed'] > 0:
                print(f"\n[Frame {i}] Idle-GPU results collected:")
                print(f"  Completed: {collect_result['tasks_completed']}")
                print(f"  Pending: {collect_result['tasks_pending']}")
                for r in collect_result['results']:
                    if r and 'rescued' in r:
                        print(f"    Blind rescue: {r['rescued']}/{r['qs_count']} frames")
                        rescued_count += r['rescued']
                    elif r and 'z_median' in r:
                        print(f"    Spurious z: median={r['z_median']:.2f}, wall={r['wall_fraction']:.1%}")

    # Final collect
    print(f"\n[Frame {n}] Final collect...")
    final = gpu_pool.collect_results(timeout=1.0)
    print(f"Completed {final['tasks_completed']} final task(s)")

    # Print diagnostics summary
    diag_results = diag_queue.drain()
    print(f"\nDiagnostics summary: {len(diag_results)} entries")
    for entry in diag_results[:5]:  # Show first 5
        print(f"  Frame {entry['frame_id']:3d} | Task: {entry['task_type']:20s} | Result: {entry['result']}")

    # Print GPU pool stats
    print(f"\nGPU Pool stats:")
    stats = gpu_pool.stats()
    for k, v in stats.items():
        if k == 'reference':
            print(f"  Reference validation QA:")
            for rk, rv in v.items():
                if isinstance(rv, float):
                    print(f"    {rk}: {rv:.2%}" if 'rate' in rk or 'pass' in rk else f"    {rk}: {rv:.2f}")
                else:
                    print(f"    {rk}: {rv}")
        else:
            print(f"  {k}: {v}")

    print(f"\nSummary:")
    print(f"  Total frames: {n}")
    print(f"  Indexed (fast path): {indexed_count}")
    print(f"  Rescued (blind rescue): {rescued_count}")
    print(f"  Total indexed: {indexed_count + rescued_count} ({100*(indexed_count+rescued_count)/n:.0f}%)")

    gpu_pool.shutdown()


if __name__ == '__main__':
    main()
