#!/usr/bin/env python3
"""Input preparation. Text only: FRAME blocks are copied byte for byte, so every arm feeds the
driver exactly the lines of the source list (no float re-formatting), and frame ids are kept.

    xg_prep.py SRC OUTDIR [--ninst 32] [--subset 24]

Writes under OUTDIR:
  rot/rot_<i>.txt   instance i's copy of the whole list, starting at block round(i*n/ninst)
  one/f_<k>.txt     one file per block, k = position in SRC (the queue arms pop k = 0, 1, ...)
  sub<S>.txt        the first S blocks (with --subset S)
  manifest.json     sha256 of SRC, n, the rotation offsets, the frame ids in order
"""
import argparse
import hashlib
import json
import os


def read_blocks(path):
    blocks, cur = [], None
    with open(path) as f:
        for line in f:
            if line.startswith("FRAME"):
                if cur is not None:
                    blocks.append(cur)
                parts = line.split()
                cur = {"fid": int(parts[1]), "npk": int(parts[2]), "lines": [line]}
            elif cur is not None and line.strip():
                cur["lines"].append(line)
    if cur is not None:
        blocks.append(cur)
    for b in blocks:
        if len(b["lines"]) != 1 + b["npk"]:
            raise SystemExit(f"FRAME {b['fid']}: header says {b['npk']} peaks, found {len(b['lines']) - 1}")
    return blocks


def write(path, blocks):
    with open(path, "w") as f:
        for b in blocks:
            f.writelines(b["lines"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("outdir")
    ap.add_argument("--ninst", type=int, default=32)
    ap.add_argument("--subset", type=int, default=0)
    a = ap.parse_args()

    blocks = read_blocks(a.src)
    n = len(blocks)
    os.makedirs(os.path.join(a.outdir, "rot"), exist_ok=True)
    os.makedirs(os.path.join(a.outdir, "one"), exist_ok=True)
    offsets = [round(i * n / a.ninst) % n for i in range(a.ninst)]
    for i, o in enumerate(offsets):
        write(os.path.join(a.outdir, "rot", f"rot_{i}.txt"), blocks[o:] + blocks[:o])
    for k, b in enumerate(blocks):
        write(os.path.join(a.outdir, "one", f"f_{k}.txt"), [b])
    if a.subset:
        write(os.path.join(a.outdir, f"sub{a.subset}.txt"), blocks[:a.subset])
    sha = hashlib.sha256(open(a.src, "rb").read()).hexdigest()
    man = {"src": os.path.abspath(a.src), "sha256": sha, "n": n, "ninst": a.ninst,
           "offsets": offsets, "fids": [b["fid"] for b in blocks],
           "peaks_min": min(b["npk"] for b in blocks), "peaks_max": max(b["npk"] for b in blocks),
           "peaks_mean": sum(b["npk"] for b in blocks) / n, "subset": a.subset}
    json.dump(man, open(os.path.join(a.outdir, "manifest.json"), "w"), indent=1)
    print(f"{a.src}: n={n} sha256={sha[:16]} peaks {man['peaks_min']}-{man['peaks_max']} "
          f"mean {man['peaks_mean']:.1f}; offsets {offsets[:4]}...")


if __name__ == "__main__":
    main()
