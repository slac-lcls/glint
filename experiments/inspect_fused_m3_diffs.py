"""#28 follow-up: WHICH 3 cells does the fused M3 kernel move, and does it matter?

bench_fused_m3.py reported blind same_lattice unchanged at 85/120 with 117/120 cells identical.
An unchanged rate is consistent with two very different stories:

  wash   -- the 3 frames were already wrong under torch and are still wrong under fused
  swap   -- fused loses a frame torch got right and gains a different one, netting zero

The second would mean the kernel is trading correct answers, which an aggregate rate hides. This
resolves it per frame, and also checks something never tested: whether the kernel is DETERMINISTIC
run to run. A non-deterministic kernel would make every rate measurement in #28 noise.

  python inspect_fused_m3_diffs.py
"""
import os, sys, time, importlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from glint.multishot import same_lattice
from glint.lattice import cell_to_Ar


def load_gf(fused):
    os.environ["M3_FUSED"] = "1" if fused else "0"
    import glint.glint_fast as gf
    importlib.reload(gf)
    return gf


gf0 = load_gf(False)
frames = [q for q in gf0.load(os.path.join(HERE, "frames_cxidb_clean.txt")) if len(q) >= 6]
n = len(frames)
LYSO = gf0.LYSO


def run(fused):
    gf = load_gf(fused)
    gf.index_blind_fast(frames[0])
    return [gf.index_blind_fast(q) for q in frames]


def axes(M):
    if M is None:
        return None
    A = np.asarray(M, float)
    L = np.sort(np.linalg.norm(A, axis=0))
    return L


def nspots(M, q, tol=0.15):
    if M is None:
        return 0
    r = np.asarray(q) @ M
    return int((np.abs(r - np.rint(r)).max(1) < tol).sum())


def gate(M, q):
    if M is None or not same_lattice(np.asarray(M, float), LYSO):
        return 0
    m = nspots(M, q)
    return int(m / len(q) >= 0.25 and m >= 10)


T = run(False)
F1 = run(True)
F2 = run(True)                       # determinism: same flag, second pass

print(f"blind on {n} cxidb frames\n")
det = sum(int((a is None) == (b is None) and (a is None or np.array_equal(np.asarray(a), np.asarray(b))))
          for a, b in zip(F1, F2))
print(f"1. DETERMINISM: fused run twice -> {det}/{n} bit-identical"
      f"   {'OK' if det == n else '*** NON-DETERMINISTIC — every rate in #28 is noise ***'}")

diff = [i for i in range(n)
        if (T[i] is None) != (F1[i] is None)
        or (T[i] is not None and not np.allclose(np.asarray(T[i]), np.asarray(F1[i]), atol=1e-6))]
print(f"\n2. MOVED CELLS: {len(diff)}/{n} -> frames {diff}")
print(f"\n{'frame':>6}{'peaks':>7} | {'torch lat':>10}{'torch gate':>11}{'t spots':>9}"
      f" | {'fused lat':>10}{'fused gate':>11}{'f spots':>9} | verdict")
tally = {"both wrong (wash)": 0, "torch right -> fused WRONG": 0,
         "torch wrong -> fused right": 0, "both right": 0}
for i in diff:
    q = frames[i]
    tl = T[i] is not None and same_lattice(np.asarray(T[i], float), LYSO)
    fl = F1[i] is not None and same_lattice(np.asarray(F1[i], float), LYSO)
    tg, fg = gate(T[i], q), gate(F1[i], q)
    if tl and fl:
        v = "both right"
    elif tl and not fl:
        v = "torch right -> fused WRONG"
    elif fl and not tl:
        v = "torch wrong -> fused right"
    else:
        v = "both wrong (wash)"
    tally[v] += 1
    print(f"{i:>6}{len(q):>7} | {str(tl):>10}{tg:>11}{nspots(T[i], q):>9}"
          f" | {str(fl):>10}{fg:>11}{nspots(F1[i], q):>9} | {v}")
    for lbl, M in (("  torch axes", T[i]), ("  fused axes", F1[i])):
        a = axes(M)
        print(f"{lbl}: {np.array2string(a, precision=1) if a is not None else 'None'}")

print("\n3. VERDICT")
for k, v in tally.items():
    if v:
        print(f"   {k}: {v}")
tl_all = sum(int(M is not None and same_lattice(np.asarray(M, float), LYSO)) for M in T)
fl_all = sum(int(M is not None and same_lattice(np.asarray(M, float), LYSO)) for M in F1)
tg_all = sum(gate(M, q) for M, q in zip(T, frames))
fg_all = sum(gate(M, q) for M, q in zip(F1, frames))
print(f"   totals -- same_lattice torch {tl_all}/{n} fused {fl_all}/{n} | "
      f"gate torch {tg_all}/{n} fused {fg_all}/{n}")
print("   A wash is benign. A swap means the kernel trades correct answers and the flat rate hides it.")
os.environ["M3_FUSED"] = "0"
