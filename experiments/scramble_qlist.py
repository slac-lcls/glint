#!/usr/bin/env python3
"""Azimuth-scramble a FRAME-block q list: the lattice-free null for a chance-accept arm.

Every peak of every frame is rotated by its own random azimuth about the beam axis
(glint.multilattice.scramble_azimuth), which keeps |q| and q_z -- so each peak's Ewald excitation error and
the radial structure -- and destroys only the lattice coherence. Frame i is scrambled with
np.random.default_rng([seed, i]), so the output is a function of (input, seed) alone and any arm that reads
it can be rerun from the repository. The file is written in the same FRAME-block format as the input
(glint.glint_fast.load / stream2q.py) and read back before the script exits.

    python experiments/scramble_qlist.py ~/q480_fix.txt q480_scrambled.txt --seed 20260928

Used by the effort= chance-accept arm (feat/adaptive-effort): the recorder replays the real 480 first, so the
driver locks on them, then this file as a second species -- its indexed count is the fast path's chance-accept
rate under the live gate at the tier in force, its escalated count the deep search's.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from glint.glint_fast import load                    # noqa: E402
from glint.multilattice import scramble_azimuth      # noqa: E402


def scramble_frames(frames, seed):
    return [scramble_azimuth(np.asarray(q, float), np.random.default_rng([int(seed), i])) for i, q in enumerate(frames)]


def write_qlist(path, frames):
    with open(path, "w") as f:
        for i, q in enumerate(frames):
            f.write(f"FRAME {i} {len(q)}\n")
            np.savetxt(f, np.asarray(q, float), fmt="%.9f")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="FRAME-block q list (qx qy qz in 1/A, one FRAME header per frame)")
    ap.add_argument("dst", help="output q list, same format")
    ap.add_argument("--seed", type=int, default=20260928, help="rng seed; frame i uses default_rng([seed, i])")
    a = ap.parse_args(argv)
    frames = [np.asarray(q, float) for q in load(os.path.expanduser(a.src))]
    if not frames:
        raise SystemExit(f"{a.src}: no FRAME blocks")
    out = scramble_frames(frames, a.seed)
    write_qlist(os.path.expanduser(a.dst), out)
    back = load(os.path.expanduser(a.dst))
    if len(back) != len(frames) or any(len(b) != len(q) for b, q in zip(back, frames)):
        raise SystemExit("round trip lost frames or peaks")
    for b, q in zip(back, frames):
        if not (np.allclose(np.linalg.norm(b, axis=1), np.linalg.norm(q, axis=1), atol=2e-9)
                and np.allclose(b[:, 2], q[:, 2], atol=2e-9)):
            raise SystemExit("round trip changed |q| or q_z")
    md5 = hashlib.md5(open(os.path.expanduser(a.dst), "rb").read()).hexdigest()
    print(f"{len(frames)} frames, {sum(len(q) for q in frames)} peaks scrambled (seed {a.seed}) -> {a.dst}  md5 {md5}")


if __name__ == "__main__":
    main()
