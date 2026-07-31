"""Verify SUB-RUN SEPARATION in the driver's .stream: under adaptive_relock a mid-run sample change
adds a second active cell, and every chunk must be stamped with the cell that ACTUALLY indexed it.

This is the claim that matters for offline merging. If cell_id is wrong -- or worse, absent -- an
offline merger silently co-merges two different crystals into one dataset, which is a corrupted
result rather than a failed one. The existing tests only ever ran a single cell, so cell_id was 0
everywhere and the labelling was never exercised.

No suitable two-cell dataset exists (~/glint_sim is one cell), so frames are synthesised here: two
clearly distinct cells, no simple supercell/alias relation between them, painted as Gaussian spots
through the real predict_spots geometry so the driver's own peak-finder and indexer do the work.

The test is not "does a cell_id appear" but "does the cell_id AGREE with the cell in that chunk's own
Cell parameters line" -- a label that does not predict the content is worse than no label.

  python test_stream_subruns.py          # needs a GPU (indexing is CUDA)
"""
import os, sys, tempfile
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from glint.lattice import cell_to_Ar
from glint.predict import predict_spots
import glint.stream_driver as sd

LAM = 1.322
NPX = 1000
RES = 10000.0                      # px per metre -> 0.1 mm pixels
CLEN = 0.1
PAN = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=RES,
            cx=-(NPX / 2.0 - 0.5), cy=-(NPX / 2.0 - 0.5), coffset=0.0,
            min_fs=0, max_fs=NPX - 1, min_ss=0, max_ss=NPX - 1)]

CELL_A = (79.02, 79.02, 37.98, 90, 90, 90)      # lysozyme, tetragonal
CELL_B = (62.50, 72.30, 85.10, 90, 90, 90)      # orthorhombic; not a supercell/alias of A
NA, NB = 40, 40
NPEAKS = 120        # realistic sparse-SFX spot count, see below
MIN_INLIERS = 30
# On the miss-gate, which decides whether a frame reaches the watchdog at all. Painting EVERY spot
# predict_spots returns gave ~830 peaks for cell A and ~1340 for cell B, and at that density a
# cell-B frame picks up a median 84 CHANCE inliers against cell A's metric (6.2% of its peaks, from
# the tol=0.15 max-norm box) -- clearing any count-based gate, so nothing ever missed and no relock
# could be triggered. The fractions separate cleanly where the counts do not: 0.647 for cell-A frames
# vs 0.062 for cell-B. That is min_inliers being a COUNT rather than a fraction, and it is why a
# frame with many peaks can pass on chance alone. Frames are subsampled to NPEAKS here so the test
# runs in the sparse regime the driver is actually built for.
fails = []


def check(name, cond, msg=""):
    print(f"  {name:38s}: {'PASS' if cond else 'FAIL'}   {msg}")
    if not cond:
        fails.append(name)


def rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def frames_for(cell, n, seed):
    """n detector images of one cell at random orientations, via the real prediction geometry."""
    Ar = cell_to_Ar(*cell)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[-3:4, -3:4]
    g = np.exp(-(xx * xx + yy * yy) / 2.0); g /= g.max()
    out = []
    while len(out) < n:
        M = rot(rng) @ Ar
        pred = predict_spots(M, PAN, CLEN, LAM, dmin=2.5, tol=0.004)
        if len(pred) < 60:                                   # too few spots to index -- redraw
            continue
        if len(pred) > NPEAKS:                               # sparse regime -- see NPEAKS above
            pred = pred[rng.choice(len(pred), NPEAKS, replace=False)]
        img = np.zeros((NPX, NPX), np.float32)
        for p in pred:
            cf, cs = int(round(p["fs"])), int(round(p["ss"]))
            if 4 <= cs < NPX - 4 and 4 <= cf < NPX - 4:
                img[cs - 3:cs + 4, cf - 3:cf + 4] += 4000.0 * g
        img += rng.normal(8.0, 2.0, img.shape).astype(np.float32)   # flat background
        out.append(np.clip(img, 0, 65535).astype(np.uint16))
    return out


def reduced(a, b, c):
    return tuple(round(x, 1) for x in sorted([a, b, c]))


def main():
    imgs = frames_for(CELL_A, NA, 11) + frames_for(CELL_B, NB, 22)   # sample change at frame NA
    print(f"synthesised {NA} frames of cell A {CELL_A[:3]} then {NB} of cell B {CELL_B[:3]}")

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "subruns.stream")
        drv = sd.StreamDriver(None, PAN, CLEN, LAM, (NPX, NPX), dtype=np.uint16, B=8, dmin=2.5,
                              tol=0.004, adaptive_relock=True, min_inliers=MIN_INLIERS,
                              warmup_nbest=3, stream_out=path, qc_frac_threshold=0.25)
        for f in imgs:
            drv.push(f)
        drv.close()                                   # flushes the resident batch, then the stream
        s = drv.stats()
        txt = open(path).read()                       # read INSIDE the tempdir's lifetime

    print(f"locked_after={s.get('locked_after')}  n_cells={s.get('n_cells')}  "
          f"n_relock={s.get('n_relock')}  integrated={s.get('integrated')}  "
          f"chunks={s.get('stream_chunks')}")

    check("driver relocked on the sample change", s.get("n_relock", 0) >= 1 and s.get("n_cells", 1) >= 2,
          f"n_relock={s.get('n_relock')}, n_cells={s.get('n_cells')}")
    if not (s.get("n_cells", 1) >= 2):
        print("\nno second cell was added -- cannot test separation")
        print(f"\n{'FAILED: ' + ', '.join(fails)}")
        return 1

    # ---- parse the stream into (cell_id, reduced cell, lock_generation) per chunk ----------------
    chunks = [c for c in txt.split("----- Begin chunk -----")[1:]]
    got = []
    for c in chunks:
        if "--- Begin crystal" not in c:
            continue
        cid = gen = None; cellp = None
        for ln in c.split("\n"):
            if ln.startswith("Cell parameters "):
                p = ln.split()
                cellp = reduced(float(p[2]) * 10, float(p[3]) * 10, float(p[4]) * 10)   # nm -> A
            elif ln.startswith("glint/cell_id = "):
                cid = int(ln.split(" = ")[1])
            elif ln.startswith("glint/lock_generation = "):
                gen = int(ln.split(" = ")[1])
        got.append((cid, cellp, gen))

    check("every chunk carries a cell_id", all(g[0] is not None for g in got), f"{len(got)} chunks")
    ids = sorted({g[0] for g in got})
    check("both cells appear in the stream", len(ids) >= 2, f"cell_ids present: {ids}")

    # ---- THE REAL TEST: does the label predict the content? -------------------------------------
    RA, RB = reduced(*CELL_A[:3]), reduced(*CELL_B[:3])
    by_id = {}
    for cid, cellp, _ in got:
        by_id.setdefault(cid, []).append(cellp)
    print("\n  cell_id -> observed reduced cells (A)")
    for cid in sorted(by_id):
        uniq = {}
        for cp in by_id[cid]:
            uniq[cp] = uniq.get(cp, 0) + 1
        print(f"    {cid}: n={len(by_id[cid]):3d}  {dict(sorted(uniq.items(), key=lambda kv: -kv[1]))}")
    print(f"  truth: A={RA}  B={RB}")

    # Assign each chunk to whichever truth cell is NEARER, rather than to an absolute tolerance
    # around its own. That is exactly the question a merger asks when it splits a stream by cell_id
    # ("do these group into two coherent crystals?"), and it does not mistake the unconstrained
    # anneal's few-tenths-of-an-angstrom per-frame drift for a mislabelling: A and B differ by tens
    # of angstroms, so nearest-of-two is unambiguous even for a visibly drifted chunk.
    def nearer(obs):
        da = sum(abs(o - r) / r for o, r in zip(obs, RA))
        db = sum(abs(o - r) / r for o, r in zip(obs, RB))
        return "A" if da < db else "B"

    # each cell_id must map to ONE consistent cell, and the two ids must map to DIFFERENT cells
    pure = {}
    for cid, cps in by_id.items():
        na = sum(1 for cp in cps if nearer(cp) == "A")
        pure[cid] = ("A" if na * 2 >= len(cps) else "B", max(na, len(cps) - na) / len(cps))
    for cid, (lab, frac) in sorted(pure.items()):
        check(f"cell_id {cid} is internally consistent", frac == 1.0,
              f"{100*frac:.0f}% of its chunks are nearer cell {lab}")
    labs = {cid: lab for cid, (lab, _) in pure.items()}
    check("the two cell_ids are DIFFERENT cells", len(set(labs.values())) == len(labs),
          f"cell_id -> cell: {labs}  (identical would mean the label separates nothing)")

    # a merger splitting on cell_id must recover both crystals
    check("splitting on cell_id recovers A and B", set(labs.values()) == {"A", "B"},
          f"{labs}")
    # lock_generation must advance for the cell added by the relock
    gens = {cid: {g for c, _, g in got if c == cid} for cid in ids}
    check("lock_generation recorded per chunk", all(None not in v for v in gens.values()),
          f"{gens}")

    print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
