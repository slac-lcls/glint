"""Phase C: merge the integrated .stream with CrystFEL and score it.

process_hkl (all + odd/even half-sets, -y 4/mmm) -> compare_hkl for CC1/2, CC*, Rsplit and check_hkl
for completeness/redundancy/SNR; then correlate the merged intensities against the nanoBragg ground
-truth |F|^2 (CC_true), canonicalising hkl under the 4/mmm Laue group so the two index settings match.

  source psconda.sh ; conda activate ana-... ; python phaseC_merge.py [stream] [-y SYM]
"""
import os, sys, subprocess, re
import numpy as np

CF = "/sdf/group/lcls/ds/tools/crystfel/0.12.0/bin"
SIM = "/sdf/home/s/smarches/glint_sim"
stream = sys.argv[1] if len(sys.argv) > 1 else f"{SIM}/glint_truth.stream"
SYM = "4/mmm"
base = stream.rsplit(".", 1)[0]
CELL = f"{SIM}/lyso.cell"

with open(CELL, "w") as f:
    f.write("CrystFEL unit cell file version 1.0\n\nlattice_type = tetragonal\ncentering = P\n"
            "unique_axis = c\n\na = 79.00 A\nb = 79.00 A\nc = 38.00 A\n"
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
m = re.search(r"(\d+)\s+reflections.*?(\d+)\s+measurements", log, re.S)
print(f"merge log tail: {log.strip().splitlines()[-1] if log.strip() else '(empty)'}")

# CrystFEL FOMs (overall)
for fom in ("Rsplit", "CC", "CCstar"):
    o = run([f"{CF}/compare_hkl", odd, even, "-y", SYM, "-p", CELL, f"--fom={fom}", "--nshells=1"])
    line = [l for l in o.splitlines() if "Overall" in l or "overall" in l or fom in l]
    val = next((l.strip() for l in o.splitlines() if re.search(r"[Oo]verall", l)), None)
    print(f"  {fom:8}: {val if val else o.strip().splitlines()[-1] if o.strip() else 'n/a'}")
chk = run([f"{CF}/check_hkl", allhkl, "-y", SYM, "-p", CELL, "--nshells=1"])
for l in chk.splitlines():
    if re.search(r"[Oo]verall|completeness|Compl|redundancy|measurements|reflections", l):
        print(f"  check_hkl: {l.strip()}")

# ---- truth CC: correlate merged I vs nanoBragg |F|^2 ----
def laue_ops():
    gens = [np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),      # 4-fold c
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),     # 2-fold a
            -np.eye(3, dtype=int)]                             # inversion
    G = [np.eye(3, dtype=int)]
    changed = True
    while changed:
        changed = False
        for g in list(G):
            for s in gens:
                h = (s @ g).astype(int)
                if not any(np.array_equal(h, x) for x in G):
                    G.append(h); changed = True
    return G

OPS = laue_ops()
def canon(hkl):
    eqs = np.stack([(hkl @ op.T) for op in OPS])              # (nops, n, 3)
    keys = eqs[:, :, 0] * 10**8 + eqs[:, :, 1] * 10**4 + eqs[:, :, 2]
    best = keys.argmax(0)
    return eqs[best, np.arange(eqs.shape[1])]

T = np.load(f"{SIM}/truth.npz")
th = canon(T["hkl"].astype(int)); tF = T["Fsq"].astype(float)
tkey = th[:, 0] * 10**8 + th[:, 1] * 10**4 + th[:, 2]
truth = {int(k): v for k, v in zip(tkey, tF)}

mh, mI = [], []
for line in open(allhkl):
    p = line.split()
    if len(p) >= 4 and re.match(r"^-?\d+$", p[0]) and re.match(r"^-?\d+$", p[1]) and re.match(r"^-?\d+$", p[2]):
        try:
            mh.append([int(p[0]), int(p[1]), int(p[2])]); mI.append(float(p[3]))
        except ValueError:
            pass
mh = np.array(mh); mI = np.array(mI)
if len(mh) == 0:
    print("\nTRUTH CC: no merged reflections parsed (merge failed) -- skipping"); sys.exit(0)
ch = canon(mh); ckey = ch[:, 0] * 10**8 + ch[:, 1] * 10**4 + ch[:, 2]
mF = np.array([truth.get(int(k), np.nan) for k in ckey])
ok = np.isfinite(mF) & (mI > 0)
cc = np.corrcoef(mI[ok], mF[ok])[0, 1]
ccl = np.corrcoef(np.log(mI[ok] + 1), np.log(mF[ok] + 1))[0, 1]
print(f"\nTRUTH CC: {ok.sum()} merged refl matched to ground truth; "
      f"CC(I,|F|^2)={cc:.3f}  CC(logI,log|F|^2)={ccl:.3f}")
