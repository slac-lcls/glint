"""Half-dataset CCanom per shell, inverse-variance weighted: w = 1/(var dI1 + var dI2), var dI = s(+)^2 + s(-)^2."""
import sys, json, warnings, numpy as np, gemmi, reciprocalspaceship as rs
warnings.filterwarnings('ignore')
sys.path.insert(0, '/pscratch/sd/k/kmdalton/glint/experiments/profile_bench')
from hkl_to_mtz import read_hkl as _read_hkl, reindex as _reindex
import os
def read_hkl(p):
    """The merge in the model's setting: REINDEX (env, default h,k,l) maps GLINT's indices to SG's."""
    x = _read_hkl(p).astype(float)
    x[:, :3] = _reindex(x[:, :3], os.environ.get("REINDEX", "h,k,l"))
    return x
PM, cellstr, dmin, nsh = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
cell, sg = gemmi.UnitCell(*map(float, cellstr.split(','))), gemmi.SpaceGroup(os.environ.get('SG', 'P 43 21 2'))
cut = json.load(open(f'{PM}/merge/cut.json'))['common']
def load(p):
    x = read_hkl(p)
    d = rs.DataSet({'H': x[:, 0].astype(int), 'K': x[:, 1].astype(int), 'L': x[:, 2].astype(int), 'I': x[:, 3], 'S': x[:, 4]},
                   cell=cell, spacegroup=sg).infer_mtz_dtypes().set_index(['H', 'K', 'L'])
    d = d[d.S > 0].hkl_to_asu(anomalous=True).remove_absences(); d.merged = True
    u = d.drop(columns=['M/ISYM'], errors='ignore').unstack_anomalous(['I', 'S']).dropna()
    return rs.DataSet({'D': u['I(+)'] - u['I(-)'], 'V': u['S(+)'] ** 2 + u['S(-)'] ** 2}, cell=cell, spacegroup=sg)
def wcc(x, y, w):
    if len(x) < 3: return None
    w = w / w.sum(); mx, my = (w * x).sum(), (w * y).sum()
    return float((w * (x - mx) * (y - my)).sum() / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum()))
E_ = np.linspace(30.0 ** -3, dmin ** -3, nsh + 1) ** (-1 / 3)
out = {'edges': E_.round(4).tolist(), 'cut': cut}
for arm in ('box', 'profile', 'crystfel'):
    a, b = load(f'{PM}/merge/{arm}/merged.hkl1'), load(f'{PM}/merge/{arm}/merged.hkl2')
    j = a.join(b, rsuffix='2', how='inner').dropna(); j.cell, j.spacegroup = cell, sg; j = j.compute_dHKL()
    x, y, w, d = j.D.to_numpy(float), j.D2.to_numpy(float), 1 / (j.V.to_numpy(float) + j.V2.to_numpy(float)), j.dHKL.to_numpy(float)
    sh = [wcc(x[s], y[s], w[s]) for s in ((d < hi) & (d >= lo) for hi, lo in zip(E_[:-1], E_[1:]))]
    tot = d >= cut
    out[arm] = {'wccanom': sh, 'overall_to_cut': wcc(x[tot], y[tot], w[tot]), 'unweighted_overall_to_cut': float(np.corrcoef(x[tot], y[tot])[0, 1])}
    print(arm, 'weighted overall', round(out[arm]['overall_to_cut'], 3), 'unweighted', round(out[arm]['unweighted_overall_to_cut'], 3))
    print('  shells', [None if v is None else round(v, 3) for v in sh])
print('JSON ' + json.dumps(out))
