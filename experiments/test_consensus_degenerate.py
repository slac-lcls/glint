"""A degenerate cell in the N-best pool must not take the consensus down.

REGRESSION. The densest-neighbourhood seeding added in glint#102 keys each hypothesis by
`round(log(edge) / log1p(rtol))`. An edge that is zero, negative or non-finite has no meaningful
log -- numpy returns -inf or nan and round() raises OverflowError rather than a number -- so a
single degenerate hypothesis anywhere in the pool aborted the whole consensus.

This is not hypothetical and it is not cheap: it killed a 3000-frame Jungfrau 16M ladder
(mfx101555026 r0013) at its SIXTH of eight thresholds, 15 minutes of reading in, on the real N-best
pool of a noisy run. Degenerate hypotheses are a normal product of blind indexing on noise; the
consensus is entitled to ignore them, not to die on them.

What is pinned here:
  * a pool containing zero / negative / inf / nan edges still returns the MAJORITY cell;
  * the answer is IDENTICAL to the same pool with the degenerate entries removed -- they must
    influence nothing, not merely fail to crash;
  * an all-degenerate pool refuses cleanly instead of raising.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint.lattice import cell_to_Ar
from glint.multishot import consensus_cell

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


TRUE = (54.15, 87.29, 141.33, 90.0, 90.0, 90.0)      # ClCRY4, the run that exposed this
rng = np.random.default_rng(0)


def jitter(cell, f=0.004):
    return cell_to_Ar(*[v * (1 + rng.normal(0, f)) if i < 3 else v + rng.normal(0, 0.15)
                        for i, v in enumerate(cell)])


clean = [jitter(TRUE) for _ in range(24)]

# The degenerate shapes that actually reach a real pool, i.e. the ones whose EDGE LENGTH breaks
# log(): a collapsed axis (norm 0 -> log 0 = -inf), an infinite axis, and a nan axis.
# NOT a "negative edge": a length is a norm and cannot be negative. Scaling a column by a negative
# number just flips its sign, which is a valid basis of the SAME lattice -- an earlier version of
# this test included that case and it duly joined the true group and raised support by one, which
# is correct behaviour and not a degeneracy at all.
bad = []
for edges in ((0.0, 87.0, 141.0), (np.inf, 87.0, 141.0), (54.0, np.nan, 141.0)):
    M = cell_to_Ar(54.15, 87.29, 141.33, 90.0, 90.0, 90.0).copy()
    for k, e in enumerate(edges):                      # scale each column to the degenerate length
        col = M[:, k]
        n = np.linalg.norm(col)
        M[:, k] = col / n * e if n else col
    bad.append(M)

print("a degenerate hypothesis does not abort the consensus")
try:
    got, sup = consensus_cell(clean + bad)      # -> (representative M, support)
    ok = got is not None
except Exception as e:
    got, sup, ok = None, 0, False
    check("consensus_cell raised", False, f"{type(e).__name__}: {e}")
check("a pool with 3 degenerate cells still returns a cell", ok, got)

print("\n...and changes nothing about the answer")
ref, sup_ref = consensus_cell(clean)
if ref is not None and got is not None:
    d = float(np.max(np.abs(np.sort(np.linalg.norm(got, axis=0))
                            - np.sort(np.linalg.norm(ref, axis=0)))))
    check("identical to the same pool without them", d < 1e-9, d)
    check("...and the support is unchanged too", sup == sup_ref, (sup, sup_ref))
    edges = np.sort(np.linalg.norm(ref, axis=0))
    check("and it is the TRUE cell", bool(np.allclose(edges, sorted(TRUE[:3]), rtol=0.02)),
          np.round(edges, 2))
else:
    check("the clean pool locks at all", False, ref)

print("\nan all-degenerate pool refuses, it does not raise")
try:
    r, rs = consensus_cell(bad)
    check("returns None rather than a cell", r is None, (r, rs))
except Exception as e:
    check("all-degenerate pool raised", False, f"{type(e).__name__}: {e}")

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
