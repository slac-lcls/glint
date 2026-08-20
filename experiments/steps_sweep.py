"""M3 ascent-step sweep at n=480 on the cxidb-17 benchmark.

WHY. glint_fast.STEPS defaults to 8, and its comment records a sweep -- "blind saturates >=8,
16-80 flat within +-3-frame noise", "STEPS=5 ... blind-standalone drops to 77" -- taken at n=120.
n=120 is precisely the sample size whose coincidences the n=480 benchmark exposed (three separate
paper claims read an n=120 agreement as an explanation and stopped agreeing at n=480), so a knob
calibrated there is calibrated on the weakest evidence in the project. The adversarial-review claim
this tests is sharper than "is 8 right": M3's accuracy is alleged to depend on UNDER-optimising --
CG/BB/Newton/LM all LOWER the rate -- which would make the headline blind rate a function of a step
count rather than of the objective.

WHAT. STEPS is a module global read at CALL time (`steps=STEPS` inside index_blind_fast's refiner
dispatch), so it can be set per arm in one process; everything else -- frames, warmup, gate, seed
order -- is held identical. Two rates per arm:

  blind   index_blind_fast per frame, strict gate. The quantity STEPS actually controls.
  hybrid  hybrid_index(Mc_known=None) = GLINT-(1), the published pipeline. Consensus + rescue can
          absorb a worse blind cell, so this is where "does the knob matter to the DELIVERABLE"
          gets answered, and the two can disagree.

CONTROL. The first 120 frames at STEPS=8 must give hybrid 92 (check_numbers.py FACTS:
glint1_strict_of120 = 92). Blind is 78 here, NOT the 84 that glint_fast's STEPS comment used to
quote: 84 is the same_lattice-only figure and this gate is the strict bar (same_lattice AND
matched/npk >= 0.25 AND matched >= 10). Comparing across those two bars is how a sweep quietly
starts measuring a different question.

  python steps_sweep.py <q480.txt> <out.npz> [STEPS_LIST]
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

# Resolve the checkout from THIS FILE, so a vendored copy always measures the tree it ships in.
# A hard-coded path made the harness import whichever worktree happened to exist on the author's
# box -- i.e. not necessarily the code under review, and nothing at all for anyone else.
# GLINT_ROOT overrides it, which is what a copy living outside experiments/ needs.
ROOT = os.environ.get("GLINT_ROOT") or str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import glint.glint_fast as gf
from glint.glint_fast import matched, index_blind_fast
from glint.multishot import same_lattice
from glint.hybrid_stream import hybrid_index
LYSO = gf.LYSO

QFILE = sys.argv[1]
OUT = sys.argv[2]
STEPS_LIST = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else \
    [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 80]

frames = [q for q in gf.load(QFILE) if len(q) >= 6]
n = len(frames)
print(f"# {n} frames from {QFILE}; STEPS arms: {STEPS_LIST}", flush=True)
print(f"# shipped default STEPS={gf.STEPS}\n", flush=True)


def strict(M, q):
    if M is None or not same_lattice(M, LYSO):
        return False
    m = matched(M, q)
    return m / len(q) >= 0.25 and m >= 10


rows = []
gates = {}
for s in STEPS_LIST:
    gf.STEPS = s                                   # read at call time by the refiner dispatch
    t0 = time.time()
    index_blind_fast(frames[0])                    # warm the kernels for THIS step count
    tb = time.time()
    blind_M = [index_blind_fast(q) for q in frames]
    t_blind = time.time() - tb
    gb = np.array([strict(M, q) for M, q in zip(blind_M, frames)])

    res, st = hybrid_index(frames, Mc_known=None, warmup=True)
    gh = np.array([strict(r["M"], q) for r, q in zip(res, frames)])

    gates[f"blind_{s}"] = gb
    gates[f"hybrid_{s}"] = gh
    rows.append((s, int(gb.sum()), int(gh.sum()), int(gb[:120].sum()), int(gh[:120].sum()),
                 1000 * t_blind / n, time.time() - t0))
    print(f"STEPS {s:3d}   blind {gb.sum():3d}/{n} ({100*gb.mean():4.1f}%)   "
          f"hybrid {gh.sum():3d}/{n} ({100*gh.mean():4.1f}%)   "
          f"| first-120 blind {gb[:120].sum():3d} hybrid {gh[:120].sum():3d}   "
          f"| {1000*t_blind/n:5.1f} ms/frame blind", flush=True)

# The control, CHECKED BEFORE ANYTHING IS WRITTEN. A harness that prints "this measured the wrong
# setup" and then exits 0 hands an unattended runner a result it has itself disowned.
d = dict((r[0], r) for r in rows)
if 8 in d:
    b120, h120 = d[8][3], d[8][4]
    ok = (h120 == 92)                             # check_numbers FACTS glint1_strict_of120
    print(f"\nCONTROL STEPS=8 first-120: blind {b120} (expect 78, strict bar), "
          f"hybrid {h120} (expect 92)  -> {'OK' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        raise SystemExit(f"control failed: first-120 hybrid {h120} != 92; this sweep is measuring "
                         f"something other than the published pipeline, so {OUT} was NOT written")
else:
    print("\n!! STEPS=8 not in the arm list -- no control was checked", flush=True)

np.savez(OUT, steps=np.array([r[0] for r in rows]),
         blind=np.array([r[1] for r in rows]), hybrid=np.array([r[2] for r in rows]),
         blind120=np.array([r[3] for r in rows]), hybrid120=np.array([r[4] for r in rows]),
         ms_per_frame=np.array([r[5] for r in rows]), n=n, **gates)
print(f"wrote {OUT}", flush=True)
