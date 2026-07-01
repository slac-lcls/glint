"""Benchmark fftindex on real LCLS peaks (same peaks xgandalf indexes via LUTE).

Modes:
  geom    <geom>             parse + summarize geometry (no data needed)
  inspect <cxi> <geom>       dump CXI h5 tree + sample peaks -> confirm formats
  run     <cxi> <geom> [N]   convert peaks -> q -> index_shot -> rate vs ref cell

Reference cell (maybe_lyso.cell): monoclinic P, a=28.0 b=62.5 c=60.9, beta=90.8.
Run inspect FIRST -- the peak(fs,ss)->panel mapping and dataset names must be
confirmed against the real CXI before trusting `run`.
"""

import sys

sys.path.insert(0, "..")
import numpy as np

from glint import index_shot
from glint.lattice import cell_to_Ar
from glint.lute_bridge import lambda_from_eV, parse_geom, peaks_to_q
from glint.multishot import cell_signature, same_lattice

REF_CELL = (28.0, 62.5, 60.9, 90.0, 90.8, 90.0)
M_REF = cell_to_Ar(*REF_CELL)


def cmd_geom(geom):
    panels, glob = parse_geom(geom)
    print(f"{len(panels)} panels; res={panels[0]['res']:.1f} px/m "
          f"(pixel {1e6/panels[0]['res']:.1f} um)")
    print(f"  data slab: fs<= {max(p['max_fs'] for p in panels)}, "
          f"ss<= {max(p['max_ss'] for p in panels)}")
    print(f"  clen src = {glob.get('clen')!r}   photon src = {glob.get('photon_energy')!r}")
    print(f"  ref cell signature (1/A-less, A): {np.round(cell_signature(M_REF), 1)}")


def cmd_inspect(cxi, geom):
    import h5py
    cmd_geom(geom)
    print("\nCXI datasets:")
    with h5py.File(cxi, "r") as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}  shape={obj.shape} dtype={obj.dtype}")
        f.visititems(show)
        print("\nsample (event 0):")
        for k in ["/entry_1/result_1/nPeaks", "/entry_1/result_1/peakXPosRaw",
                  "/entry_1/result_1/peakYPosRaw", "/LCLS/photon_energy_eV",
                  "/LCLS/detector_1/EncoderValue"]:
            if k in f:
                v = np.asarray(f[k])
                print(f"  {k}: shape {v.shape} -> {v.ravel()[:6]}")


def cmd_run(cxi, geom, max_events=300, verbose=8):
    """Detector distance (calibrated vs CrystFEL stream): clen = EncoderValue/1000
    (mm->m) PLUS per-panel coffset -> z=0.2 m here."""
    import h5py
    panels, glob = parse_geom(geom)
    clen_p = glob.get("clen", "").strip()
    phot_p = glob.get("photon_energy", "").strip()
    with h5py.File(cxi, "r") as f:
        nP = np.asarray(f["/entry_1/result_1/nPeaks"])
        X = f["/entry_1/result_1/peakXPosRaw"]
        Y = f["/entry_1/result_1/peakYPosRaw"]
        nev = min(max_events, len(nP))

        def rd(path, e):
            a = np.asarray(f[path])
            return float(a[e]) if a.ndim and a.shape[0] == len(nP) else float(a.ravel()[0])

        used = solved = matched = 0
        npks = []
        for e in range(nev):
            k = int(nP[e])
            if k < 6:
                continue
            fs, ss = np.asarray(X[e])[:k], np.asarray(Y[e])[:k]
            clen_m = rd(clen_p, e) / 1000.0          # EncoderValue mm -> m (+coffset inside)
            q = peaks_to_q(fs, ss, panels, clen_m, lambda_from_eV(rd(phot_p, e)))
            q = q[~np.isnan(q).any(axis=1)]
            if len(q) < 6:
                continue
            used += 1
            npks.append(len(q))
            res = index_shot(q, float(np.linalg.norm(q, axis=1).max()))
            cell = np.sort(np.linalg.norm(res.M, axis=0)) if res.M is not None else None
            ok = res.M is not None and same_lattice(res.M, M_REF)
            solved += res.M is not None
            matched += ok
            if used <= verbose:
                print(f"  ev{e}: {len(q):3d} peaks  qmax={np.linalg.norm(q,axis=1).max():.3f}  "
                      f"cell={np.round(cell,1) if cell is not None else None}  match={ok}")
        print(f"\nframes used (>=6 peaks): {used} of {nev}; "
              f"median peaks/frame: {np.median(npks):.0f}")
        print(f"fftindex blind:  any-solution {100*solved/max(used,1):.0f}%   "
              f"matches ref cell {100*matched/max(used,1):.0f}%   (ref {REF_CELL[:3]})")


def cmd_diag(cxi, geom):
    """Deep dive on the best frame at the CALIBRATED distance: blind vs known-cell,
    and a tolerance sweep -- isolates 'our indexer is weak on real data' from geometry."""
    import h5py
    from glint.multishot import index_known_pairangle, reference_lattice
    panels, glob = parse_geom(geom)
    phot_p = glob.get("photon_energy", "").strip()
    clen_p = glob.get("clen", "").strip()
    with h5py.File(cxi, "r") as f:
        nP = np.asarray(f["/entry_1/result_1/nPeaks"])
        ev = int(np.argmax(nP[:50]))
        k = int(nP[ev])
        fs = np.asarray(f["/entry_1/result_1/peakXPosRaw"][ev])[:k]
        ss = np.asarray(f["/entry_1/result_1/peakYPosRaw"][ev])[:k]
        lam = lambda_from_eV(float(np.asarray(f[phot_p])[ev]))
        clen_m = float(np.asarray(f[clen_p])[ev]) / 1000.0     # calibrated: +coffset -> 0.2 m
    q = peaks_to_q(fs, ss, panels, clen_m, lam)
    q = q[~np.isnan(q).any(axis=1)]
    qmax = float(np.linalg.norm(q, axis=1).max())
    iu, ju = np.triu_indices(len(q), 1)
    d = np.sort(np.linalg.norm(q[iu] - q[ju], axis=1))
    print(f"event {ev}: {len(q)} peaks  qmax={qmax:.3f}  pairwise-d: "
          f"min={d[0]:.4f} 5%={np.percentile(d,5):.4f} (shortest recip ~{1/62.5:.4f})")
    print(f"ref cell signature: {np.round(cell_signature(M_REF), 1)}")
    Vref = reference_lattice(M_REF, qmax)
    for tf in [0.02, 0.04, 0.06]:
        b = index_shot(q, qmax, tol_frac=tf)
        bc = np.round(np.sort(np.linalg.norm(b.M, axis=0)), 1) if b.M is not None else None
        print(f"  blind tol={tf}: cell={bc}  indexed_peaks={b.n_indexed}")
    kn = index_known_pairangle(q, qmax, M_REF, Vref=Vref)
    print(f"  known-cell (taketwo): match={kn.M is not None and same_lattice(kn.M, M_REF)}  "
          f"cell={np.round(np.sort(np.linalg.norm(kn.M,axis=0)),1) if kn.M is not None else None}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "geom":
        cmd_geom(a[1])
    elif a and a[0] == "diag":
        cmd_diag(a[1], a[2])
    elif a and a[0] == "inspect":
        cmd_inspect(a[1], a[2])
    elif a and a[0] == "run":
        cmd_run(a[1], a[2], int(a[3]) if len(a) > 3 else 300)
    else:
        print(__doc__)
