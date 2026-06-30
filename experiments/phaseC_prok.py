"""Phase C on REAL data (NERSC): merge the ProK .stream with CrystFEL and report CC1/2, CC*, Rsplit,
completeness, redundancy, SNR (no ground-truth |F| -- real data). 4/mmm = ProK Laue group.

  module load pytorch/2.6.0 ; python phaseC_prok.py [stream]
"""
import os, sys, subprocess, re

CF = "/global/cfs/cdirs/lcls/chuck/crystfel/bin"
D = "/pscratch/sd/s/smarches/glint_real"
stream = sys.argv[1] if len(sys.argv) > 1 else f"{D}/prok_glint.stream"
SYM = "4/mmm"
base = stream.rsplit(".", 1)[0]
CELL = f"{D}/prok.cell"

with open(CELL, "w") as f:
    f.write("CrystFEL unit cell file version 1.0\n\nlattice_type = tetragonal\ncentering = P\n"
            "unique_axis = c\n\na = 68.70 A\nb = 68.70 A\nc = 108.60 A\n"
            "al = 90.00 deg\nbe = 90.00 deg\nga = 90.00 deg\n")


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout + p.stderr


def merge(tag, extra):
    out = f"{base}_{tag}.hkl"
    log = run([f"{CF}/process_hkl", "-i", stream, "-o", out, "-y", SYM, "--scale"] + extra)
    return out, log


allhkl, log = merge("all", [])
odd, _ = merge("odd", ["--odd-only"])
even, _ = merge("even", ["--even-only"])
print("merge:", (log.strip().splitlines() or ["(no output)"])[-1])
for fom in ("Rsplit", "CC", "CCstar"):
    o = run([f"{CF}/compare_hkl", odd, even, "-y", SYM, "-p", CELL, f"--fom={fom}", "--nshells=1"])
    val = next((l.strip() for l in o.splitlines() if re.search(r"[Oo]verall", l)), None)
    print(f"  {fom:8}: {val if val else (o.strip().splitlines() or ['n/a'])[-1]}")
chk = run([f"{CF}/check_hkl", allhkl, "-y", SYM, "-p", CELL, "--nshells=1"])
for l in chk.splitlines():
    if re.search(r"completeness|redundancy|measurements in total|reflections in total|<snr>", l):
        print(f"  check_hkl: {l.strip()}")
