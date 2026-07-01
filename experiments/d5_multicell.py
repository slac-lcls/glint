"""D5: discover MULTIPLE cells blind in a mixed-protein run (different species/cells across
frames). multishot.consensus_cell groups per-frame M's by same_lattice but returns only the
LARGEST group; here consensus_cells returns ALL clusters with support>=min_support. Test on
a mixture of two distinct cells (lysozyme + an orthorhombic cell): index each frame blind,
cluster, check BOTH cells are recovered as separate clusters.

  python d5_multicell.py [N_frames]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
from glint_fast import index_blind_fast
from glint.lattice import cell_to_Ar, Ar_to_Br, random_rotation
from glint.multishot import same_lattice

LAM = 1.322; K0 = 1.0 / LAM
CELLS = [("lyso  ", (79.02, 79.02, 37.98, 90, 90, 90)),
         ("orthoB", (60.0, 70.0, 80.0, 90, 90, 90))]


def sim_cell(cell, rng, dmin=4.0, ncap=120, spur=0.2, jitter=5e-4, tol=0.002):
    Ar = cell_to_Ar(*cell); Br = Ar_to_Br(Ar); R = random_rotation(rng)
    qmax = 1.0 / dmin
    hmax = int(np.ceil(qmax * max(np.linalg.norm(Ar, axis=0)))) + 1
    rh = np.arange(-hmax, hmax + 1)
    H = np.array(np.meshgrid(rh, rh, rh, indexing="ij")).reshape(3, -1).T
    H = H[np.any(H != 0, 1)]
    g = (R @ Br @ H.T).T
    g = g[np.linalg.norm(g, axis=1) <= qmax]
    eps = np.linalg.norm(np.array([0, 0, K0]) + g, axis=1) - K0
    g = g[np.abs(eps) < tol]
    if len(g) > ncap:
        g = g[rng.choice(len(g), ncap, replace=False)]
    g = g + rng.normal(0, jitter, g.shape)
    if spur > 0 and len(g):
        ns = int(spur * len(g)); d = rng.normal(size=(ns, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        g = np.vstack([g, d * (qmax * rng.random(ns) ** (1 / 3))[:, None]])
    return g


def consensus_cells(Ms, min_support=3):
    """Return ALL lattice clusters (rep cell, support) with support>=min_support."""
    groups = []
    for M in (M for M in Ms if M is not None):
        for grp in groups:
            if same_lattice(M, grp[0]):
                grp[1].append(M); break
        else:
            groups.append([M, [M]])
    return sorted([(g[0], len(g[1])) for g in groups if len(g[1]) >= min_support],
                  key=lambda t: -t[1])


def params(M):
    G = M.T @ M; L = np.sqrt(np.diag(G))
    return tuple(np.round(np.sort(L), 1))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    rng = np.random.default_rng(0)
    index_blind_fast(sim_cell(CELLS[0][1], rng))                 # warmup
    refs = [np.asarray(cell_to_Ar(*c[1]), float) for c in CELLS]
    Ms = []; truth = []; ok_per = [0, 0]; n_per = [0, 0]
    for i in range(N):
        ci = i % len(CELLS)
        g = sim_cell(CELLS[ci][1], rng)
        M = index_blind_fast(g)
        Ms.append(M); truth.append(ci); n_per[ci] += 1
        if M is not None and same_lattice(M, refs[ci]):
            ok_per[ci] += 1
    print(f"D5 multi-cell consensus  N={N}  cells: " +
          ", ".join(f"{nm.strip()} {params(refs[i])}" for i, (nm, _) in enumerate(CELLS)))
    for i, (nm, _) in enumerate(CELLS):
        print(f"  per-frame blind {nm}: {ok_per[i]}/{n_per[i]} correct")
    clusters = consensus_cells(Ms, min_support=3)
    print(f"  consensus clusters (support>=3): {len(clusters)}")
    for M, s in clusters:
        match = next((CELLS[i][0].strip() for i in range(len(CELLS)) if same_lattice(M, refs[i])), "??UNKNOWN")
        print(f"    cell {params(M)}  support {s:2d}  -> {match}")
    both = all(any(same_lattice(M, refs[i]) for M, _ in clusters) for i in range(len(CELLS)))
    print(f"  BOTH cells recovered as clusters: {both}")
