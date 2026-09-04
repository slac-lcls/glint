"""FPGA-grade peak EMISSION vs GLINT indexing -- the SELECTION half, from committed data.

Question (from the ePixUHR/readout-FPGA edge-indexing thread): if Bragg peaks are emitted by a
readout FPGA or an in-pixel sparsifier instead of GPU peakfinder8 -- integer positions, a small
per-tile buffer, seams where a local finder cannot compute background -- does GLINT still index?

This reconstructs the SELECTION half of the 2026-07 study from data committed to this repo:
`experiments/frames_cxidb_clean.txt` is the exact 120-frame cxidb-17 benchmark (the reciprocal-space
vectors every indexer in tab:summary was fed). We project each frame's q onto a synthetic detector
panel with the geom bridge's own projection, impose a synthetic ePixUHR tile grid, apply the FPGA
emission constraints in PIXEL space, invert back to q, and index with the shipped consensus path.

Round-trip q -> (fs,ss) -> q is idempotent to machine precision (both directions normalise onto the
Ewald sphere), asserted at startup: if it fails, nothing below is a controlled comparison.

WHAT THIS DOES NOT COVER (needs raw pixels, which are purged -- do not fake):
  * intensity-weighted top-K per tile: the committed lists carry positions only, no intensity. We
    test count caps that are UNBIASED (random) and RESOLUTION-BIASED (low-|q|, a worst-case proxy);
    a real intensity cap keeps strong peaks at all resolutions and should land at the unbiased result.
  * the DETECTION model (pf8-radial vs a tile-local finder = WHICH peaks are found): needs frames.

Usage:  python experiments/fpga_peaks/fpga_selection.py [--device cpu|cuda] [--json out.json]
"""
import os, sys, json, argparse
import numpy as np

ROOT = os.environ.get("GLINT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
# Torch-backed imports are DEFERRED into main(). Importing glint.hybrid_stream initialises torch and
# fixes the visible devices, so it must happen AFTER --device has had its say -- otherwise the option
# is advertised and inert, and `--device cpu` still runs on CUDA wherever a GPU is present.
# glint/glint_cli.py:124-137 gates the same import for the same reason.
hybrid_index = None                                   # bound in main(), after the device gate
same_lattice = None                                   # bound in main(), after the device gate

# Synthetic panel geometry -- the same constants q_to_peaks.py uses to invert q-frames to a detector.
RES, CLEN, NPX, CORNER = 10000.0, 0.15, 3000, -1500.0     # px/m, m, panel px, corner (px)
LAM = 1.322                                                # A, cxidb-17 (LCLS-CXI lysozyme)
LYSO = None                                                # set after first blind solve; see baseline
BEAM = np.array([0.0, 0.0, 1.0])

# ePixUHR-style tiling. ASIC 192x168; a readout FPGA sees a 6-ASIC panel (Doering/Dragone IFDEPS).
ASIC_FS, ASIC_SS = 192, 168
PANEL_FS, PANEL_SS = ASIC_FS * 3, ASIC_SS * 2              # 576 x 336, the readout-FPGA tile


def load_frames(path):
    """Parse FRAME blocks of 3-column q (1/A) into a list of (N,3) arrays."""
    frames, cur = [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("FRAME"):
            if cur:
                frames.append(np.asarray(cur, float)); cur = []
            continue
        cur.append([float(x) for x in line.split()])
    if cur:
        frames.append(np.asarray(cur, float))
    return frames


def q_to_fsss(q):
    """Project reciprocal-space q (1/A) onto the synthetic panel; return (fs, ss, on_panel mask)."""
    shat = BEAM + LAM * np.asarray(q, float)
    shat = shat / np.linalg.norm(shat, axis=1, keepdims=True)
    t = CLEN / shat[:, 2]
    fs = t * shat[:, 0] * RES - CORNER
    ss = t * shat[:, 1] * RES - CORNER
    on = (fs >= 0) & (fs < NPX) & (ss >= 0) & (ss < NPX) & (shat[:, 2] > 0)
    return fs, ss, on


def fsss_to_q(fs, ss):
    """Invert panel pixels back to q (1/A) -- exact inverse of q_to_fsss on the Ewald sphere."""
    ray = np.stack([(fs + CORNER) / RES, (ss + CORNER) / RES, np.full_like(fs, CLEN)], axis=1)
    shat = ray / np.linalg.norm(ray, axis=1, keepdims=True)
    return (shat - BEAM) / LAM


def tile_id(fs, ss, tfs, tss):
    return (np.floor(fs / tfs).astype(int), np.floor(ss / tss).astype(int))


def near_seam(fs, ss, tfs, tss, w):
    """True for peaks within w px of any tile boundary (where a local finder loses its background)."""
    dfs = np.minimum(fs % tfs, tfs - (fs % tfs))
    dss = np.minimum(ss % tss, tss - (ss % tss))
    return (dfs < w) | (dss < w)


# ---- the emission-constraint arms: each takes (fs, ss) on-panel arrays, returns a keep mask ----

def arm_identity(fs, ss, rng):
    return np.ones(fs.shape, bool)

def make_seam(tfs, tss, w):
    def arm(fs, ss, rng):
        return ~near_seam(fs, ss, tfs, tss, w)
    return arm

def make_tilecap(cap, tfs, tss, bias="none"):
    """Keep at most `cap` peaks PER TILE -- the actual small-buffer model an edge emitter imposes.

    This is not the same experiment as the frame-global cap below, and the difference is the point:
    a per-tile buffer overflows locally, so the peaks it drops are SPATIALLY CORRELATED (a bright
    tile loses many, an empty tile loses none), whereas a frame-global sample of the same total size
    thins uniformly. A frame-global cap therefore cannot answer the per-tile-buffer question the
    README poses -- it can only bound it."""
    def arm(fs, ss, rng):
        keep = np.zeros(fs.shape[0], bool)
        ti, tj = tile_id(fs, ss, tfs, tss)
        # one bucket per occupied tile; enforce the cap inside each independently
        for key in set(zip(ti.tolist(), tj.tolist())):
            m = np.flatnonzero((ti == key[0]) & (tj == key[1]))
            if m.size <= cap:
                keep[m] = True
                continue
            if bias == "lowq":
                r = np.hypot(fs[m] + CORNER, ss[m] + CORNER)
                keep[m[np.argsort(r)[:cap]]] = True
            else:
                keep[m[rng.choice(m.size, cap, replace=False)]] = True
        return keep
    return arm


def make_countcap(cap, bias):
    """Keep at most `cap` peaks/frame, FRAME-GLOBALLY -- not a per-tile buffer (see make_tilecap).
    bias='none' random; bias='lowq' keeps smallest-radius first
    (a worst-case proxy for a resolution-biased keep, since the lists have no intensity)."""
    def arm(fs, ss, rng):
        n = fs.shape[0]
        if n <= cap:
            return np.ones(n, bool)
        if bias == "lowq":
            r = np.hypot(fs + CORNER, ss + CORNER)           # radius from beam center ~ |q|
            idx = np.argsort(r)[:cap]                         # keep the LOW-resolution ones
        else:
            idx = rng.choice(n, cap, replace=False)
        keep = np.zeros(n, bool); keep[idx] = True
        return keep
    return arm


def degrade(frames, arm, integer=False, tfs=PANEL_FS, tss=PANEL_SS, seed=0):
    """Apply projection -> (optional integer) -> arm -> inverse, per frame. Returns degraded q-frames
    and the mean kept peaks/frame."""
    rng = np.random.default_rng(seed)
    out, kept = [], []
    for q in frames:
        fs, ss, on = q_to_fsss(q)
        fs, ss = fs[on], ss[on]
        if integer:
            fs, ss = np.round(fs), np.round(ss)
        m = arm(fs, ss, rng)
        fs, ss = fs[m], ss[m]
        kept.append(fs.shape[0])
        out.append(fsss_to_q(fs, ss) if fs.size else np.zeros((0, 3)))
    return out, float(np.mean(kept))


def score(frames, ref_cell):
    """Index with the shipped consensus path; count frames whose picked cell is same_lattice(ref).
    Returns (n_indexed_at_ref, consensus_cell Mc, stats). If ref_cell is None (baseline), the run's
    own Mc is the reference."""
    results, stats = hybrid_index(frames)
    Mc = stats.get("Mc")
    ref = ref_cell if ref_cell is not None else (np.asarray(Mc, float) if Mc is not None else None)
    n_ok = 0
    if ref is not None:
        for d in results:
            M = d.get("M")
            if M is not None and same_lattice(np.asarray(M, float), ref):
                n_ok += 1
    return n_ok, (np.asarray(Mc, float) if Mc is not None else None), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=os.path.join(ROOT, "experiments", "frames_cxidb_clean.txt"))
    # 'auto' is the honest default: run_fpga.sh relies on picking up the A100, and the previous
    # default of 'cpu' described a behaviour the script never had.
    ap.add_argument("--device", choices=("auto", "cpu"), default="auto")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    # BEFORE importing anything torch-backed -- see the note at the top of this file.
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    global hybrid_index, same_lattice
    from glint.hybrid_stream import hybrid_index as _hybrid_index
    from glint.multishot import same_lattice as _same_lattice
    hybrid_index, same_lattice = _hybrid_index, _same_lattice

    frames = load_frames(a.frames)
    print(f"loaded {len(frames)} frames, mean {np.mean([len(f) for f in frames]):.0f} peaks/frame")

    # Self-check. The committed q carry real excitation error (not exactly on the Ewald sphere), so
    # q -> pixels -> q shifts them by that error -- which is physics, not a bug. What must be exact is
    # that the projection is a true INVERSE: projecting an ALREADY-projected q is idempotent to
    # machine precision. (The first-pass residual below is the mean excitation error, reported.)
    q0 = frames[0]
    fs, ss, on = q_to_fsss(q0)
    q1 = fsss_to_q(fs[on], ss[on])                        # elastic-projected
    fs2, ss2, on2 = q_to_fsss(q1)
    q2 = fsss_to_q(fs2[on2], ss2[on2])
    err = np.abs(q2 - q1[on2]).max()
    assert err < 1e-9, f"projection is not a true inverse (max {err:.2e}) -- geometry is wrong"
    exc = float(np.mean(np.abs(q1 - q0[on]).sum(1)))
    print(f"geometry self-check OK (inverse exact to {err:.1e}); mean excitation shift {exc:.2e} 1/A")

    # Baseline: projection round-trip. Establishes the reference cell and the ceiling.
    baseline_frames, _ = degrade(frames, arm_identity)
    base, ref_cell, bstats = score(baseline_frames, None)
    if ref_cell is None:
        sys.exit("baseline formed no consensus cell -- cannot proceed")
    edges = np.round(np.sort(np.linalg.norm(ref_cell, axis=0)), 1)
    print(f"baseline consensus cell edges = {edges} A   indexed {base}/{len(frames)}")

    arms = [
        ("baseline (projection round-trip)", arm_identity, False, PANEL_FS, PANEL_SS),
        ("integer positions", arm_identity, True, PANEL_FS, PANEL_SS),
        ("seam mask w=8, PANEL grain", make_seam(PANEL_FS, PANEL_SS, 8), True, PANEL_FS, PANEL_SS),
        ("seam mask w=8, ASIC grain", make_seam(ASIC_FS, ASIC_SS, 8), True, ASIC_FS, ASIC_SS),
        ("seam mask w=12, ASIC grain", make_seam(ASIC_FS, ASIC_SS, 12), True, ASIC_FS, ASIC_SS),
        ("count cap 48/frame, unbiased", make_countcap(48, "none"), True, PANEL_FS, PANEL_SS),
        ("count cap 48/frame, low-|q| biased", make_countcap(48, "lowq"), True, PANEL_FS, PANEL_SS),
        # PER-TILE caps: the actual edge-buffer model. Swept rather than matched to the frame-global
        # 48, because the mean kept/frame a given per-tile cap yields is not known in advance -- read
        # the kept-per-frame column to find the arm that matches the global cap's budget, and compare
        # THAT pair: same total peaks, different spatial correlation in which ones were dropped.
        ("per-tile cap 2/tile, PANEL grain", make_tilecap(2, PANEL_FS, PANEL_SS), True, PANEL_FS, PANEL_SS),
        ("per-tile cap 4/tile, PANEL grain", make_tilecap(4, PANEL_FS, PANEL_SS), True, PANEL_FS, PANEL_SS),
        ("per-tile cap 8/tile, PANEL grain", make_tilecap(8, PANEL_FS, PANEL_SS), True, PANEL_FS, PANEL_SS),
    ]
    rows = []
    for name, arm, integer, tfs, tss in arms:
        fr, kpf = degrade(frames, arm, integer=integer, tfs=tfs, tss=tss)
        n_ok, _, st = score(fr, ref_cell)
        rows.append(dict(arm=name, indexed=n_ok, total=len(frames), kept_per_frame=round(kpf, 1),
                         support=st.get("support")))
        print(f"  {name:38s}  {n_ok:3d}/{len(frames)}   {kpf:5.1f} pk/frame")

    if a.json:
        json.dump(dict(baseline=base, cell_edges=edges.tolist(), rows=rows),
                  open(a.json, "w"), indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
