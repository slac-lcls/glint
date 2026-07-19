"""Regression gate for the per-cell azimuth grid (_azimuth_grid).

The fix must be a NO-OP wherever the two shortest axes are perpendicular -- which is every cell the
engine is currently benchmarked on, LYSO included. Checks, in order:

  1. grid dispatch    -- perpendicular cells get the ORIGINAL half-turn tensor (identity, not copy)
  2. bit-identity     -- per-frame + batched + fused rescue on the 120 sparse cxidb frames vs the
                         pre-fix engine (half turn forced), elementwise exact
  3. cxidb gate       -- bare rescue and hybrid known-cell rates on the same 120 frames
  4. batch == single  -- the batched path agrees with the per-frame path on an OBLIQUE cell too,
                         i.e. the grid really is threaded through _prep

  python azimuth_validate.py
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import glint.glint_fast as gf
import glint.replica_gpu as rg
from glint.replica_gpu import index_known_gpu_cell, _axes_from_cell
from glint.replica_gpu_batch import index_known_gpu_cell_batch, index_fused
from glint.hybrid_stream import hybrid_index
from glint.multishot import same_lattice
from glint.lattice import cell_to_Ar

LYSO = gf.LYSO
frames = [q for q in gf.load(os.path.join(HERE, "frames_cxidb_clean.txt")) if len(q) >= 6]
n = len(frames)
fail = []

# ---- 1. dispatch ----------------------------------------------------------------
_, c01_lyso, _, _, _ = _axes_from_cell(LYSO)
ca, sa = rg._azimuth_grid(c01_lyso)
perp_is_half = (ca is rg.CA) and (sa is rg.SA)
OBL = cell_to_Ar(45.0, 55.0, 65.0, 80, 85, 95)              # triclinic, |c01| = 0.087
_, c01_obl, _, _, _ = _axes_from_cell(OBL)
cao, sao = rg._azimuth_grid(c01_obl)
obl_is_full = (cao is rg.CA2) and (sao is rg.SA2)
print(f"1. dispatch: LYSO c01={c01_lyso:+.2e} -> half turn: {perp_is_half} | "
      f"triclinic c01={c01_obl:+.4f} -> full turn: {obl_is_full}")
print(f"   grids: half span {float(rg.TH[-1]):.4f} rad over {len(rg.CA)}, "
      f"full span {float(rg.TH2[-1]):.4f} rad over {len(rg.CA2)}  (equal sample count: "
      f"{len(rg.CA) == len(rg.CA2)})")
if not (perp_is_half and obl_is_full and len(rg.CA) == len(rg.CA2)):
    fail.append("dispatch")

# ---- 2. bit-identity on LYSO vs the forced-half (pre-fix) engine -----------------
def force_half():
    rg.CA2, rg.SA2 = rg.CA, rg.SA          # what the engine did before: half turn for every cell


def restore():
    rg.CA2, rg.SA2 = torch.cos(rg.TH2), torch.sin(rg.TH2)


def run_all():
    single = [index_known_gpu_cell(q, LYSO) for q in frames]
    batch = [M for i in range(0, n, 32) for M in index_known_gpu_cell_batch(frames[i:i + 32], LYSO)]
    fused = index_fused(frames, LYSO, B=32)
    return single, batch, fused


post = run_all()
force_half(); pre = run_all(); restore()
names = ["per-frame", "batched", "fused"]
print("\n2. bit-identity vs forced-half engine (LYSO, 120 sparse cxidb frames):")
for nm, a, b in zip(names, post, pre):
    bad = [i for i, (x, y) in enumerate(zip(a, b))
           if (x is None) != (y is None) or (x is not None and not np.array_equal(np.asarray(x), np.asarray(y)))]
    print(f"   {nm:<10} identical: {not bad}" + (f"  ({len(bad)} differ: {bad[:5]})" if bad else ""))
    if bad:
        fail.append(f"bitid-{nm}")

# ---- 3. cxidb gate --------------------------------------------------------------
def gate(M, q):
    """Three metrics that the repo quotes interchangeably -- keep them apart:
       lat    = same_lattice only          (glint_fast.py:37 'same_lattice 84/120', hybrid 117/120)
       >=25%  = lattice AND >=25% of spots (kf_validate.py 'frac')
       >=10   = lattice AND >=10 refl      (kf_validate.py 'loose')"""
    if M is None or not same_lattice(np.asarray(M, float), LYSO):
        return (0, 0, 0)
    r = np.asarray(q) @ M
    m = int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())
    return (1, int(m / len(q) >= 0.25), int(m >= 10))


def show(label, gs, extra=""):
    print(f"   {label:<34} same_lattice {sum(a for a,_,_ in gs)}/{n}   "
          f">=25% {sum(b for _,b,_ in gs)}/{n}   >=10refl {sum(c for _,_,c in gs)}/{n}{extra}")


# Which repo figure is which (reconciled 2026-07-19 -- these get quoted interchangeably and are NOT
# the same measurement; all three below were confirmed against their sources on this run):
#   "blind 84/120"   = BLIND FRONT END alone, index_blind_fast, same_lattice only, no consensus/rescue
#                      (validate_gpu_dedup.py:42, stream_probe.py:4). Measured 85 -- glint_fast.py:52
#                      records exactly this drift ("85 vs 84 same_lattice" at the ANNEAL_ITERS=3 default).
#   "hybrid 117/120" = hybrid same_lattice / stats['n_idx'], NOT the >=25% gate. Measured 115, inside
#                      ROADMAP.md:56's documented 115-118/120 band.
#   "73 / 113"       = the bare rescue's >=25% ('frac') and >=10refl ('loose'), kf_validate.py:16. Exact.
# The >=25%-of-spots gate is a FOURTH number (blind-hybrid 93, known-hybrid 91) and matches none of the
# quoted figures -- do not compare it against 84 or 117.
print(f"\n3. cxidb gate ({n} frames) -- reference values in the repo:")
print(f"   glint_fast.py:37  blind same_lattice 84/120 (85 at ANNEAL_ITERS=3), hybrid 117/120 (band 115-118)")
print(f"   kf_validate.py:16 bare rescue 73/120 frac, 113/120 loose")
show("bare rescue (per-frame)", [gate(M, q) for M, q in zip(post[0], frames)])
results, stats = hybrid_index(frames, Mc_known=LYSO, warmup=True)
show("hybrid Mc_known=LYSO", [gate(res["M"], q) for res, q in zip(results, frames)],
     f"   n_idx {stats['n_idx']}/{n}")
rb, sb = hybrid_index(frames, None, Mc_known=None, nbest=3)
show("hybrid BLIND (self-derived cell)", [gate(res["M"], q) for res, q in zip(rb, frames)],
     f"   n_idx {sb['n_idx']}/{n}")
# The 84/120 figure is the BLIND FRONT END alone, per-frame index_blind_fast scored by
# same_lattice only (validate_gpu_dedup.py:42, stream_probe.py:4) -- no consensus, no rescue.
from glint.glint_fast import index_blind_fast
show("blind front end (index_blind_fast)", [gate(index_blind_fast(q), q) for q in frames])

# ---- 4. batched == per-frame on an OBLIQUE cell ---------------------------------
sys.path.insert(0, HERE)
from azimuth_coverage import cell_to_A, rand_rot, rlps
rng = np.random.default_rng(3)
A0 = cell_to_A(45.0, 55.0, 65.0, 80, 85, 95)
obf, obM = [], []
for t in range(32):
    Ar = rand_rot(rng) @ A0
    q = rlps(Ar, rng)
    if len(q) > 800:
        q = q[rng.choice(len(q), 800, replace=False)]
    obf.append(q); obM.append(Ar)
s1 = [index_known_gpu_cell(q, OBL) for q in obf]
b1 = index_known_gpu_cell_batch(obf, OBL)
agree = sum(int((x is None) == (y is None) and (x is None or np.allclose(x, y, atol=1e-8)))
            for x, y in zip(s1, b1))
okr = sum(int(M is not None and same_lattice(np.asarray(M, float), OBL)) for M in s1)
print(f"\n4. oblique cell (tricl 45/55/65/80/85/95): batched==per-frame {agree}/{len(obf)}, "
      f"per-frame correct-lattice {okr}/{len(obf)}")
if agree != len(obf):
    fail.append("batch-vs-single-oblique")

print("\n" + ("FAILURES: " + ", ".join(fail) if fail else "ALL CHECKS PASSED"))
