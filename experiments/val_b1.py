"""B1 validation helper: generate 3 full-rotation lysozyme clouds (dense) so the CLI --mode auto routes
them to the cluster-FFT path."""
import os, sys
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
import numpy as np
def rand_rot(rng):
    A = rng.normal(size=(3, 3)); Q, R = np.linalg.qr(A); Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0: Q[:, 0] *= -1
    return Q
def full_cloud(axes, rng, dmin=3.0):
    a, b, c = axes; QMAX = 1.0 / dmin
    B = np.diag([1.0 / a, 1.0 / b, 1.0 / c])
    H = (np.ceil(QMAX * np.array([a, b, c])) + 1).astype(int)
    hkl = np.mgrid[-H[0]:H[0]+1, -H[1]:H[1]+1, -H[2]:H[2]+1].reshape(3, -1).T
    hkl = hkl[np.any(hkl != 0, 1)]
    q = hkl @ (rand_rot(rng) @ B).T
    return q[np.linalg.norm(q, axis=1) <= QMAX]
rng = np.random.default_rng(0)
clouds = [full_cloud((79., 79., 38.), rng) for _ in range(3)]
with open("/sdf/home/s/smarches/git/glint/experiments/densecloud.txt", "w") as f:
    for i, q in enumerate(clouds):
        f.write("FRAME %d %d\n" % (i, len(q)))
        for x, y, z in q: f.write("%.6f %.6f %.6f\n" % (x, y, z))
print("densecloud.txt rlps:", [len(q) for q in clouds])
