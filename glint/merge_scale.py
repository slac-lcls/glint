"""Per-frame scale for the merges: 1/mean(I), or refuse the frame when its mean intensity is not measured.

Dependency-free (numpy only) so the offline merges (experiments/merge_stats.py) can use the exact rule the
live MergeAccumulator / MergeAccumulatorDevice use, without importing the driver.
"""
import numpy as np

MERGE_MIN_FRAME_SNR = 3.0
# I/sigma bucket floors of the running merges. The first is -inf so that no measurement is dropped on the
# sign of I: selecting I > 0 biases weak reflections up and makes Rsplit look better than it is (review r2
# s1-05; #130 removed the same selection from integration). On the same merged frames stats(thr=0.0) gives
# the old I > 0 CC1/2, CC*, Rsplit (to a few ulp: accumulator rows are created in a different order), unique, common,
# redundancy and completeness. stats()["measurements"] is not thresholded: it counts every merged row,
# I <= 0 included, whatever thr is; the old I > 0 count is unique * redundancy at thr=0.0.
MERGE_SNR_BINS = (-np.inf, 0.0, 1.0, 2.0, 3.0, 5.0)


def frame_scale(I, k=MERGE_MIN_FRAME_SNR):
    """Per-frame merge scale 1/mean(I), or None when the frame's mean intensity is not measured.

    A frame is scaled only when it has more than 5 measurements and mean(I) > k * std(I)/sqrt(n).
    Otherwise it must not be merged: 1/mean(I) is unbounded as mean(I) -> 0, and the old fallback
    (scale = 1, raw detector units among frames normalised to mean 1) was worse (review r2 s1-01,
    s1-04). Every merge in the repo (the live accumulators, merge_stats.py, bench_stream_driver.py,
    run_partiality.py) uses this rule, so they stay comparable.
    """
    n = I.size
    if n <= 5:
        return None
    m = I.mean()
    if not (m > k * I.std() / np.sqrt(n)):
        return None
    return 1.0 / m
