"""Per-shell CC1/2, CCanom (half-dataset) and R-work/R-free (refined.mtz) for every arm. JSON to stdout."""
import sys, json, warnings, numpy as np, gemmi, reciprocalspaceship as rs
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
sys.path.insert(0, '/pscratch/sd/k/kmdalton/glint/experiments/profile_bench')
from hkl_to_mtz import read_hkl as _read_hkl, reindex as _reindex
import os
def read_hkl(p):
    """The merge in the model's setting: REINDEX (env, default h,k,l) maps GLINT's indices to SG's."""
    x = _read_hkl(p).astype(float)
    x[:, :3] = _reindex(x[:, :3], os.environ.get("REINDEX", "h,k,l"))
    return x
PM, cellstr, dmin_all, nsh_all, nsh_r = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
ARMS = ('box', 'profile', 'crystfel')
cell, sg = gemmi.UnitCell(*map(float, cellstr.split(','))), gemmi.SpaceGroup(os.environ.get('SG', 'P 43 21 2'))
cut = json.load(open(f'{PM}/merge/cut.json'))['common']

def load(p):
    x = read_hkl(p)
    d = rs.DataSet({'H': x[:, 0].astype(int), 'K': x[:, 1].astype(int), 'L': x[:, 2].astype(int), 'I': x[:, 3], 'S': x[:, 4], 'N': x[:, 5]},
                   cell=cell, spacegroup=sg).infer_mtz_dtypes().set_index(['H', 'K', 'L'])
    d = d[d.S > 0].hkl_to_asu(anomalous=True).remove_absences(); d.merged = True
    return d.drop(columns=['M/ISYM'], errors='ignore')

def edges(dmax, dmin, n):
    return np.linspace(dmax ** -3, dmin ** -3, n + 1) ** (-1 / 3)

def cc(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else None

def rcc(a, b):
    return float(spearmanr(a, b)[0]) if len(a) > 2 else None

out = {'common_cut': cut, 'arms': {}}
E_all = edges(30.0, dmin_all, nsh_all)
E_r = edges(30.0, cut, nsh_r)
out['edges_all'] = E_all.round(4).tolist(); out['edges_r'] = E_r.round(4).tolist()
for arm in ARMS:
    a, b = load(f'{PM}/merge/{arm}/merged.hkl1'), load(f'{PM}/merge/{arm}/merged.hkl2')
    j = a[['I']].join(b[['I']], rsuffix='2', how='inner').dropna().compute_dHKL()
    # anomalous: unstack each half into I(+)/I(-), then dI per half
    def dano(h):
        u = h[['I']].unstack_anomalous(['I']).dropna()
        return (u['I(+)'] - u['I(-)']).rename('D')
    da, db = dano(a), dano(b)
    jj = rs.DataSet({'D1': da, 'D2': db}).dropna()
    jj.cell, jj.spacegroup = cell, sg
    jj = jj.compute_dHKL()
    r = {'cchalf': [], 'ccanom': [], 'ccanom_rank': [], 'ccanom_model_rank': [], 'n_half': [], 'n_anom': [], 'r_work': [], 'r_free': [], 'n_work': [], 'n_free': []}
    for hi, lo in zip(E_all[:-1], E_all[1:]):
        s = j[(j.dHKL < hi) & (j.dHKL >= lo)]; t = jj[(jj.dHKL < hi) & (jj.dHKL >= lo)]
        r['cchalf'].append(cc(s.I.to_numpy(float), s.I2.to_numpy(float))); r['n_half'].append(len(s))
        r['ccanom'].append(cc(t.D1.to_numpy(float), t.D2.to_numpy(float))); r['n_anom'].append(len(t))
        r['ccanom_rank'].append(rcc(t.D1.to_numpy(float), t.D2.to_numpy(float)))
    r['ccanom_overall_to_cut'] = cc(*(jj[jj.dHKL >= cut][c].to_numpy(float) for c in ('D1', 'D2')))
    r['ccanom_rank_overall_to_cut'] = rcc(*(jj[jj.dHKL >= cut][c].to_numpy(float) for c in ('D1', 'D2')))
    m = rs.read_mtz(f'{PM}/refine/{arm}/refined.mtz').compute_dHKL()
    g = lambda c: m[c].to_numpy(float)
    fo = np.concatenate([g('F-obs(+)'), g('F-obs(-)')]); fm = np.concatenate([g('F-model(+)'), g('F-model(-)')])
    fl = np.concatenate([g('R-free-flags')] * 2); d = np.concatenate([g('dHKL')] * 2)
    ok = np.isfinite(fo) & np.isfinite(fm) & (fo > 0)
    fo, fm, fl, d = fo[ok], fm[ok], fl[ok], d[ok]
    both = np.isfinite(g('F-obs(+)')) & np.isfinite(g('F-obs(-)')) & np.isfinite(g('F-model(+)')) & np.isfinite(g('F-model(-)'))
    dFo, dFm, dd = (g('F-obs(+)') - g('F-obs(-)'))[both], (g('F-model(+)') - g('F-model(-)'))[both], g('dHKL')[both]
    r['model_dano_nonzero_frac'] = float(np.mean(np.abs(dFm) > 1e-6))
    free = fl == 0 if (fl == 0).sum() < (fl != 0).sum() else fl != 0
    R = lambda s: float(np.abs(fo[s] - fm[s]).sum() / fo[s].sum())
    r['overall_check'] = {'rwork': R(~free), 'rfree': R(free), 'n_free': int(free.sum())}
    for hi, lo in zip(E_r[:-1], E_r[1:]):
        s = (d < hi) & (d >= lo)
        sa = (dd < hi) & (dd >= lo); r['ccanom_model_rank'].append(rcc(dFo[sa], dFm[sa]))
        r['r_work'].append(R(s & ~free)); r['r_free'].append(R(s & free)); r['n_work'].append(int((s & ~free).sum())); r['n_free'].append(int((s & free).sum()))
    out['arms'][arm] = r
print('JSON ' + json.dumps(out))
for arm in ARMS:
    r = out['arms'][arm]; print(arm, 'rank CCano to cut', r['ccanom_rank_overall_to_cut'], 'model dano nonzero', r['model_dano_nonzero_frac'])
    print('  ccanom_rank', [None if v is None else round(v, 3) for v in r['ccanom_rank']])
    print('  ccanom_model_rank', [None if v is None else round(v, 3) for v in r['ccanom_model_rank']])
for arm in ARMS:
    r = out['arms'][arm]; print(arm, 'overall', r['overall_check'], 'CCano to cut', r['ccanom_overall_to_cut'])
    print('  ccanom', [None if v is None else round(v, 3) for v in r['ccanom']])
    print('  rfree ', [round(v, 3) for v in r['r_free']])
