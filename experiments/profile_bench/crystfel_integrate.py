"""Integrate GLINT's orientations with CrystFEL's own integrator, so the arm differs only in integration.

Reads the lattices from a GLINT stream, replays them through indexamajig --indexing=file with no
refinement or checks, and writes the integrated stream with its chunks in the GLINT stream's order
(report.py pairs arms by chunk; indexamajig's workers finish in any order).

The events name the deposit's raw Cheetah file, not GLINT's stack: the stack is lzf-compressed, a
filter only h5py carries, and libhdf5 under CrystFEL cannot read it. Stack frame i is /LCLS/tag[i].

--tol is GLINT's excitation-error gate. CrystFEL keeps a reflection when its partiality exceeds
exp(-0.5 * 1.7^2), that is |excitation error| < 1.7 R, so R = tol / 1.7 predicts the same reflections.

indexamajig skips indexing on a frame with no peaks, so a permissive peak search runs to satisfy
that. With the checks and refinement off, and no --push-res, the peaks do not reach integration.

    python crystfel_integrate.py GLINT.stream --stack STACK.h5 --raw RAW.h5 --geom RAW.geom \\
        --crystfel-bin BIN --lattice tPc --tol 0.002 -j 16 -o RUN.stream
"""
import argparse
import json
import os
import subprocess
import tempfile
import time

import h5py

BEGIN = "----- Begin chunk -----\n"


def write_solutions(stream, tags, raw, lattice, sol, lst):
    """One solution line a lattice. Returns the events, in the stream's order, and the lattice count."""
    ev = None
    star = {}
    events = {}
    n = 0
    with open(stream) as f, open(sol, "w") as s:
        for ln in f:
            if ln.startswith("Event: "):
                ev = f"tag-{tags[int(ln.split('//')[1])]}//"
            elif ln[:6] in ("astar ", "bstar ", "cstar "):
                star[ln[0]] = ln.split("=")[1].split()[:3]
                if len(star) == 3:
                    s.write(f"{raw} {ev} {' '.join(star['a'] + star['b'] + star['c'])} 0 0 {lattice}\n")
                    events[ev] = None
                    star = {}
                    n += 1
    with open(lst, "w") as f:
        for ev in events:
            f.write(f"{raw} {ev}\n")
    return list(events), n


def reorder(path, events, out):
    """Rewrite the stream with its chunks in `events` order. Returns (frames, reflections)."""
    head = cur = []
    chunks = {}
    n_refl = 0
    with open(path) as f:
        for ln in f:
            if ln == BEGIN:
                cur = [ln]
                continue
            cur.append(ln)
            if ln.startswith("Event: "):
                chunks[ln.split(": ", 1)[1].strip()] = cur
            elif ln.startswith("num_reflections = "):
                n_refl += int(ln.split("=")[1])
    missing = [e for e in events if e not in chunks]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} of {len(events)} frames have no chunk, e.g. {missing[0]}")
    with open(out + ".part", "w") as f:
        f.writelines(head)
        for e in events:
            f.writelines(chunks[e])
    os.replace(out + ".part", out)
    return len(events), n_refl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stream")
    ap.add_argument("--stack", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--geom", required=True)
    ap.add_argument("--crystfel-bin", required=True)
    ap.add_argument("--lattice", required=True, help="CrystFEL latt_cen, e.g. tPc")
    ap.add_argument("--tol", type=float, required=True, help="GLINT's excitation-error gate, 1/A")
    ap.add_argument("--integration", default="rings-nocen")
    ap.add_argument("--int-radius", default="4,5,7")
    ap.add_argument("-j", type=int, default=1)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    with h5py.File(a.stack, "r") as f:
        tags = f["/LCLS/tag"][()]
    # 1/A -> 1/m
    radius = a.tol * 1e10 / 1.7
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(a.out))) as tmp:
        sol, lst, raw_out = f"{tmp}/run.sol", f"{tmp}/run.lst", f"{tmp}/run.stream"
        events, n_lat = write_solutions(a.stream, tags, a.raw, a.lattice, sol, lst)
        t0 = time.time()
        subprocess.run([f"{a.crystfel_bin}/indexamajig", "-i", lst, "-g", os.path.abspath(a.geom), "-o", raw_out,
                        "-j", str(a.j), "--indexing=file", f"--fromfile-input-file={sol}",
                        "--peaks=zaef", "--threshold=100", "--min-squared-gradient=1000", "--no-peaks-in-stream",
                        "--no-refine", "--no-check-cell", "--no-check-peaks", "--no-retry", "--no-multi",
                        f"--integration={a.integration}", f"--int-radius={a.int_radius}",
                        f"--fix-profile-radius={radius:.4e}", "--no-non-hits-in-stream"], check=True, cwd=tmp)
        dt = time.time() - t0
        n, n_refl = reorder(raw_out, events, a.out)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump({"t_integrate_s": dt, "n_frames": n, "n_refl": n_refl, "n_lattices": n_lat, "cores": a.j}, f, indent=1)
    print(f"{a.out}: {n} frames, {n_lat} lattices, {n_refl} reflections, {dt:.0f} s on {a.j} cores")


if __name__ == "__main__":
    main()
