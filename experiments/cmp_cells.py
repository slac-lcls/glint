"""Is fp32 LATTICE-identical to fp64, not merely rate-identical? Compare the per-frame cell edges
returned by index_fused at B=120 under KC_FP=64 vs KC_FP=32 (saved by index_batch_sweep.py)."""
import numpy as np
a = np.load('/tmp/idx_cells_64.npy', allow_pickle=True)[0]
b = np.load('/tmp/idx_cells_32.npy', allow_pickle=True)[0]
print('frames:', len(a), ' None fp64/fp32:', sum(x is None for x in a), sum(x is None for x in b))
print('None positions agree:', [x is None for x in a] == [x is None for x in b])
d = [float(np.max(np.abs(np.array(x) - np.array(y))))
     for x, y in zip(a, b) if x is not None and y is not None]
print('cells compared:', len(d))
print('max  edge diff fp32 vs fp64: %.6f A' % max(d))
print('median edge diff:            %.6f A' % float(np.median(d)))
print('cells differing by >0.01 A:  %d / %d' % (sum(1 for x in d if x > 0.01), len(d)))
