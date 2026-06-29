"""Self-contained correctness test for the CrystFEL .geom -> q bridge (no real data / no GPU):
lysozyme cell + random orientation -> Ewald spots q -> forward-project to detector pixels via a
synthetic single-panel geom -> write a real .geom -> parse_geom + peaks_to_q -> q' -> index.
Passes if (a) q' reproduces q and (b) the bridged peaks index back to lysozyme.

  python test_geom_bridge.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.spatial.transform import Rotation
from fftindex import index_shot
from fftindex.lattice import cell_to_Ar
from fftindex.multishot import same_lattice
from fftindex.geom import parse_geom, peaks_to_q

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
LAM = 1.322                                              # A (cxidb)
RES, CLEN = 10000.0, 0.10                                # px/m, m
CX, CY, NPX = -750.0, -750.0, 1500                       # beam-centred panel corner (px), size

GEOM = f"""photon_energy = {12398.419843320026 / LAM:.4f}
clen = {CLEN}
res = {RES}
coffset = 0.0
adu_per_eV = 0.001
p0/min_fs = 0
p0/max_fs = {NPX - 1}
p0/min_ss = 0
p0/max_ss = {NPX - 1}
p0/corner_x = {CX}
p0/corner_y = {CY}
p0/fs = +1.0x +0.0y
p0/ss = +0.0x +1.0y
"""


def ewald_spots(seed):
    R = Rotation.random(random_state=seed).as_matrix()
    B = np.linalg.inv(LYSO).T
    hk = np.array([[h, k, l] for h in range(-7, 8) for k in range(-7, 8) for l in range(-13, 14)])
    g = (R @ B @ hk.T).T
    s0 = np.array([0, 0, 1 / LAM])
    res = np.abs(np.linalg.norm(s0 + g, axis=1) - 1 / LAM)
    return g[(res < 0.0012) & (np.linalg.norm(g, axis=1) > 0.02)]


def project(q):
    """q (near Ewald) -> (detector pixels (fs,ss), exact-Ewald q_ref). Snaps the scattered ray to
    the sphere (shat normalized), so q_ref is what real detector pixels encode and the bridge must
    reproduce exactly; the input q's small off-Ewald slack is the only difference."""
    shat = np.array([0, 0, 1.0]) + LAM * q
    shat = shat / np.linalg.norm(shat, axis=1, keepdims=True)   # exact unit (on sphere)
    t = CLEN / shat[:, 2]
    px, py = t * shat[:, 0], t * shat[:, 1]                 # metres
    fs = px * RES - CX; ss = py * RES - CY                  # pixels
    q_ref = (shat - np.array([0, 0, 1.0])) / LAM
    return np.stack([fs, ss], 1), q_ref


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile("w", suffix=".geom", delete=False) as f:
        f.write(GEOM); gpath = f.name
    geom = parse_geom(gpath)
    print(f"parsed geom: {len(geom['panels'])} panel(s), wavelength={geom['wavelength_A']:.4f} A "
          f"(expect {LAM})")

    nidx = nlyso = qmax_err = 0; trials = 12
    worst = 0.0
    for s in range(trials):
        q = ewald_spots(s)
        if len(q) < 10:
            continue
        peaks, q_ref = project(q)
        on = (peaks[:, 0] >= 0) & (peaks[:, 0] < NPX) & (peaks[:, 1] >= 0) & (peaks[:, 1] < NPX)
        peaks = peaks[on]; q_ref = q_ref[on]
        qb = peaks_to_q(peaks, geom)                        # the bridge under test
        err = np.linalg.norm(qb - q_ref, axis=1).max()      # pixel<->q round-trip (must be exact)
        worst = max(worst, err)
        r = index_shot(qb, float(np.linalg.norm(qb, axis=1).max()))
        ok = r.M is not None and same_lattice(r.M, LYSO)
        nidx += r.M is not None; nlyso += ok
        if s < 4:
            cell = np.round(np.sort(np.linalg.norm(r.M, axis=0)), 1) if r.M is not None else None
            print(f"  trial {s}: {len(qb):2d} spots  bridge_err={err:.2e} 1/A  cell={cell}  lyso?={ok}")
    print(f"\nbridge round-trip max error: {worst:.2e} 1/A  ({'PASS' if worst < 1e-9 else 'CHECK'})")
    print(f"indexed {nidx}/{trials}, correct lysozyme {nlyso}/{trials}  "
          f"({'PASS' if nlyso >= trials - 1 else 'CHECK'})")

    # --- CrystFEL peak-search stream input path (what LUTE/peakfinder8 emits) ---
    from fftindex.geom import read_crystfel_peaks
    buf = ["CrystFEL stream format 2.3\n"]
    expect = []
    for s in range(3):
        peaks, _ = project(ewald_spots(s + 100))
        on = (peaks[:, 0] >= 0) & (peaks[:, 0] < NPX) & (peaks[:, 1] >= 0) & (peaks[:, 1] < NPX)
        peaks = peaks[on]
        buf.append("----- Begin chunk -----\nImage filename: synth.cxi\nEvent: //%d\n" % s)
        buf.append("num_peaks = %d\nPeaks from peak search\n  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel\n" % len(peaks))
        for fs, ss in peaks:
            buf.append(f"  {fs:.2f}  {ss:.2f}  0.0  1000.0  p0\n")
        buf.append("End of peak list\n----- End chunk -----\n")
        expect.append(len(peaks))
    with tempfile.NamedTemporaryFile("w", suffix=".stream", delete=False) as f:
        f.write("".join(buf)); spath = f.name
    chunks = read_crystfel_peaks(spath); os.unlink(spath); os.unlink(gpath)
    ok_stream = 0
    for ch, npk in zip(chunks, expect):
        assert len(ch["peaks"]) == npk, f"peak count {len(ch['peaks'])} != {npk}"
        qb = peaks_to_q(ch["peaks"], geom)
        r = index_shot(qb, float(np.linalg.norm(qb, axis=1).max()))
        ok_stream += r.M is not None and same_lattice(r.M, LYSO)
    print(f"CrystFEL peak-stream input: {len(chunks)} chunks read, "
          f"{ok_stream}/{len(chunks)} indexed to lysozyme  "
          f"({'PASS' if ok_stream == len(chunks) else 'CHECK'})")
