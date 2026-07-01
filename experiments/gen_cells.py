"""Corroboration: generate dense (full-rotation, 3D-complete) reciprocal-lattice clouds for a RANGE of
unit cells, so the GLINT-cluster-vs-DPS speedup can be checked across cells (not just lysozyme). Saved
as npz shared by glint_cells.py (GPU) and dials_cells.py (cctbx) -> identical input to both.

  python gen_cells.py   ->  /pscratch/sd/s/smarches/glint_real/cells.npz
"""
import numpy as np

DMIN = 3.0; QMAX = 1.0 / DMIN; REPS = 4
CELLS = [                                   # name, (a,b,c,alpha,beta,gamma); axes <=135A (cluster FOV)
    ("lyso_tet",   (79.0, 79.0, 38.0, 90, 90, 90)),
    ("prok_tet",   (68.5, 68.5, 109.0, 90, 90, 90)),
    ("thaum_tet",  (58.0, 58.0, 130.0, 90, 90, 90)),    # long c
    ("hex_P6",     (105.0, 105.0, 75.0, 90, 90, 120)),  # hexagonal
    ("cubic_ins",  (78.0, 78.0, 78.0, 90, 90, 90)),     # insulin-like cubic
    ("ortho_lg",   (60.0, 110.0, 135.0, 90, 90, 90)),   # large anisotropic
]


def cell_to_B(a, b, c, al, be, ga):
    al, be, ga = np.radians([al, be, ga])
    va = [a, 0, 0]
    vb = [b * np.cos(ga), b * np.sin(ga), 0]
    cx = c * np.cos(be); cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    vc = [cx, cy, np.sqrt(max(c * c - cx * cx - cy * cy, 1e-9))]
    A = np.array([va, vb, vc]).T            # real-space cell (columns a,b,c)
    return np.linalg.inv(A).T               # reciprocal (columns a*,b*,c*)


def rand_rot(r):
    u1, u2, u3 = r.random(3)
    q = np.array([np.sqrt(1-u1)*np.sin(2*np.pi*u2), np.sqrt(1-u1)*np.cos(2*np.pi*u2),
                  np.sqrt(u1)*np.sin(2*np.pi*u3), np.sqrt(u1)*np.cos(2*np.pi*u3)])
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


rng = np.random.default_rng(0)
out = {"names": np.array([n for n, _ in CELLS]), "reps": REPS, "qmax": QMAX}
for name, cp in CELLS:
    B = cell_to_B(*cp)
    out[name + "_axes"] = np.sort(np.array(cp[:3], float))     # truth axis lengths
    H = (np.ceil(QMAX / np.linalg.norm(B, axis=0)) + 1).astype(int)
    g = np.mgrid[-H[0]:H[0]+1, -H[1]:H[1]+1, -H[2]:H[2]+1].reshape(3, -1).T
    g = g[np.any(g != 0, axis=1)]
    for r in range(REPS):
        A = rand_rot(rng) @ B
        q = g @ A.T
        out[f"{name}_{r}"] = q[np.linalg.norm(q, axis=1) <= QMAX].astype(np.float64)

np.savez("/pscratch/sd/s/smarches/glint_real/cells.npz", **out)
print("saved cells.npz:", {n: int(np.median([len(out[f'{n}_{r}']) for r in range(REPS)])) for n, _ in CELLS})
