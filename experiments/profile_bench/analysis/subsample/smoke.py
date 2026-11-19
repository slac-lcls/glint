"""No merges: index two runs of each arm, write one 50-lattice subset per arm, check it, and run stats() on the full ProK merge."""
import sys, time, numpy as np
sys.path.insert(0, '/pscratch/sd/k/kmdalton/cxidb_45/work/probe/subsample')
import subsample as S
W = '/pscratch/sd/k/kmdalton/cxidb_45/work'
runs = ['run296940-1', 'run296941-1']
idx = {a: [S.index_stream(f'{W}/streams/{a}/{r}.stream') for r in runs] for a in S.ARMS}
for a in S.ARMS:
    print(a, [len(s) for _, s in idx[a]], 'header bytes', len(idx[a][0][0]))
lat = [(r, k) for r in range(2) for k in range(len(idx['box'][r][1]))]
rng = np.random.default_rng(0); pick = sorted(lat[i] for i in rng.choice(len(lat), 50, replace=False))
for a in S.ARMS:
    out = f'/pscratch/sd/k/kmdalton/cxidb_45/work/probe/subsample/smoke_{a}.stream'
    S.write_subset([f'{W}/streams/{a}/{r}.stream' for r in runs], idx[a][0][0], [s for _, s in idx[a]], pick, out)
    t = open(out).read()
    print(a, 'chunks', t.count('----- Begin chunk'), 'crystals', t.count('--- Begin crystal'), 'ends', t.count('----- End chunk'), 'starts with', t[:30].replace('\n', ' '))
t0 = time.time()
s = S.stats(f'{W}/partialator/merge/profile', [68.5, 68.5, 108.6, 90, 90, 90], 1.06, 14, [1.4014, 1.353], 1.353)
print('stats', round(time.time() - t0, 1), 's; cchalf overall/outer', round(s['cchalf'][14], 3), round(s['cchalf'][15], 3), 'ccano_w', round(s['ccano_w'][14], 4), 'rank', round(s['ccano_rank'][14], 4), 'n', s['n'][14], s['n_ano'][14])
