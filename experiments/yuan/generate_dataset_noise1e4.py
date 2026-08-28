"""Twin of data/simulated_data/ (generate_dataset.py, noise=2e-4) at noise=1e-4, using the EXACT
SAME 20 orientations (Rt) -- so this reproduces yesterday's PR'ed experiment's shape (20 crystals
x {1e-4, 2e-4} noise) as a fair PAIRED comparison across noise, without yesterday's rng-crystal-
identity bug (see generate_dataset.py's docstring).

#streaks doesn't depend on noise (the admission mask only depends on Rt/NA, not on the position
jitter), so reusing the same Rt set for both noise levels keeps the streak-count distribution
identical across the pair -- only the noise magnitude differs.

  python generate_dataset_noise1e4.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import simulate

from generate_dataset import load_dataset as load_base

NOISE = 1e-4
OUTDIR = os.path.join(os.path.dirname(__file__), "data", "simulated_data_noise1e4")


def generate_twin(noise=NOISE, outdir=OUTDIR):
    base = load_base()                          # the 20 crystals at noise=2e-4
    os.makedirs(outdir, exist_ok=True)
    ns = []
    for i, c in enumerate(base):
        Rt = c["Rt"]
        rng = np.random.default_rng(9000 + i)    # independent, reproducible per-crystal stream
        kobs, lab, cents = simulate(Rt, rng, noise)
        ns.append(len(cents))
        np.savez(os.path.join(outdir, f"crystal_{i:03d}.npz"),
                 Rt=Rt, kobs=kobs, lab=lab, cents=cents, noise=noise)
    return ns


def load_dataset(outdir=OUTDIR, n=20):
    # regenerate transparently if the (gitignored) npz aren't materialized yet -- same pattern
    # as generate_dataset.py::load_dataset, which this twin dataset was missed by in #60.
    if not os.path.exists(os.path.join(outdir, f"crystal_{0:03d}.npz")):
        generate_twin(outdir=outdir)
    crystals = []
    for i in range(n):
        d = np.load(os.path.join(outdir, f"crystal_{i:03d}.npz"))
        crystals.append(dict(Rt=d["Rt"], kobs=d["kobs"], lab=d["lab"], cents=d["cents"],
                             noise=float(d["noise"])))
    return crystals


if __name__ == "__main__":
    ns = generate_twin()
    base_ns = [len(c["cents"]) for c in load_base()]
    print(f"noise={NOISE:.0e}  n=20  saved to {OUTDIR}/")
    print(f"#streaks (this noise level): {ns}")
    print(f"#streaks (2e-4 twin, for comparison): {base_ns}")
    assert ns == base_ns, "streak counts should be IDENTICAL across noise (same Rt, mask independent of noise)"
    print("streak counts match the 2e-4 twin exactly, as expected.")
