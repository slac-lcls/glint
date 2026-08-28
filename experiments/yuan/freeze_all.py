"""One-time helper: freeze the flat CBXD datasets so their bulky npz can be gitignored + regenerated.

Only the two FLAT datasets (generate_dataset.py / generate_dataset_lowna.py) need their own selection
file -- they draw a pool and pick 20 by #streaks. The GRID and MULTISHOT datasets are ALREADY frozen:
generate_dataset_grid.py caches its 20 orientations per NA to data/grid/orientations_na*.npz (committed,
7 files), and generate_dataset_multishot.py REUSES orientations_na0.028.npz -- both regenerate their
per-crystal npz deterministically from those, so nothing here is needed for them.

This writes data/selection_simulated_data.npz and data/selection_simulated_data_lowna.npz: the 20
ground-truth orientations (Rt) of the currently-committed crystals, in crystal-index order, so
build_dataset() reproduces the exact same crystals (result-identical). Run once from experiments/yuan/:

    python freeze_all.py            # dry-run
    python freeze_all.py --write    # write the two selection_*.npz
"""
import os
import sys
import glob

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data")
FLAT = ["simulated_data", "simulated_data_lowna"]     # the datasets that select-by-#streaks


def main(write=False):
    for name in FLAT:
        d = os.path.join(DATA, name)
        crystals = sorted(glob.glob(os.path.join(d, "crystal_*.npz")))
        if not crystals:
            print(f"  {name:24s}  (no crystals found -- skip)")
            continue
        Rt = np.stack([np.load(c)["Rt"] for c in crystals])
        out = os.path.join(DATA, f"selection_{name}.npz")
        kb = sum(os.path.getsize(c) for c in crystals) / 1024
        print(f"  {name:24s}  {len(crystals):3d} crystals  {kb:6.0f} KB -> selection_{name}.npz ({Rt.nbytes} B)")
        if write:
            np.savez(out, Rt=Rt)
    print("\ngrid + multishot: already frozen via data/grid/orientations_na*.npz (7 files) -- nothing to do.")
    if not write:
        print("(dry run -- re-run with --write)")


if __name__ == "__main__":
    main(write="--write" in sys.argv)
