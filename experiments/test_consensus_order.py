"""Is the consensus vote a function of the hypothesis SET, or of the order it arrives in?

It should be the set. It is not: `_grp_reduced` tests each hypothesis against the member that
happened to SEED its group, and same_lattice-style tolerance matching is not transitive, so the
partition depends on arrival order. Permuting the pool of a real run (mfxl1038923, nothing else
changed) moves the reported support by 1.57x on r0278 and 2.09x on r0058, and on r0058 four of
twenty orderings refuse outright -- a fragmented cluster falls under min_frac/min_lead.

This matters beyond reproducibility. Offline pools frames in file order and the streaming driver
sees them in arrival order, so the two can disagree about whether a run locks at all, from nothing
but the order.

GLINT_CONSENSUS_STABLE=1 canonicalises the order first. These tests pin both halves: the winning
lattice is robust either way (that was already true), the SUPPORT is not, and under the flag the
whole result is invariant.

CPU only, synthetic hypotheses -- no GPU, no data files.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


def build_pool(seed=3, n_true=140, n_spur=260, jitter=0.020):
    """A realistic pool: many noisy copies of one true cell plus a cloud of distinct spurious ones.

    JITTER SETS WHETHER THE SEED MATTERS, and 2% is the regime real data sits in. The grouping
    tolerance is 5% on axis lengths, so at 1% every copy is within tolerance of every other, the
    relation is effectively transitive and the partition is order-independent by luck (measured:
    140/140 support under every permutation -- this test passed vacuously at that setting). At 2% the
    band is wide enough that two copies can be out of tolerance with each other while both are inside
    it from a third, which is exactly the non-transitivity the seed choice then resolves arbitrarily:
    support 81..129, a 1.59x spread that matches the 1.57x and 2.09x measured on real runs.
    """
    rng = np.random.default_rng(seed)

    def rot():
        Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
        return Q * np.sign(np.diag(R))

    true = np.diag([44.5, 70.4, 90.4])
    pool = [rot() @ (true * (1 + rng.normal(0, jitter, 3))) for _ in range(n_true)]
    pool += [rot() @ np.diag(rng.uniform(25, 120, 3)) for _ in range(n_spur)]
    return pool


def consensus(pool, stable):
    os.environ["GLINT_CONSENSUS_STABLE"] = "1" if stable else "0"
    for m in ("glint.multishot",):                      # re-read the flag each call: it is read at
        sys.modules.pop(m, None)                        # call time, but be explicit about intent
    from glint.multishot import consensus_cell
    return consensus_cell(pool, min_frac=0.02, min_lead=1.5)


def sweep(pool, stable, n_perm=12, seed=17):
    rng = np.random.default_rng(seed)
    sup, cells, refused = [], [], 0
    for _ in range(n_perm):
        p = [pool[i] for i in rng.permutation(len(pool))]
        M, s = consensus(p, stable)
        sup.append(s)
        if M is None:
            refused += 1
        else:
            cells.append(np.linalg.norm(M, axis=0))
    return np.array(sup), np.array(cells), refused


pool = build_pool()
print(f"pool: {len(pool)} hypotheses (140 noisy copies of one cell + 260 distinct spurious)")

print("\nDEFAULT (order-dependent): permuting the SAME set moves the support")
sup, cells, ref = sweep(pool, stable=False)
spread = sup.max() / max(sup.min(), 1)
print(f"  support {sup.min()}..{sup.max()} (spread {spread:.2f}x), refused {ref}/{len(sup)}")
check("support is NOT invariant by default (the defect this pins)", spread > 1.05, spread)
if len(cells):
    axis_spread = float(np.max((cells.max(0) - cells.min(0)) / cells.mean(0)))
    check("the winning LATTICE is robust anyway (<2% axis spread)", axis_spread < 0.02, axis_spread)

print("\nGLINT_CONSENSUS_STABLE=1: the result is a function of the SET")
sup_s, cells_s, ref_s = sweep(pool, stable=True)
check("support identical under every permutation", len(set(sup_s.tolist())) == 1,
      sorted(set(sup_s.tolist())))
check("never refuses under one ordering and answers under another", ref_s in (0, len(sup_s)), ref_s)
if len(cells_s) > 1:
    check("cell identical under every permutation",
          bool(np.allclose(cells_s, cells_s[0], atol=1e-9)), cells_s[:2])

print("\nwhere the stable answer falls inside the spread the default was sampling from")
print(f"  default {sup.min()}..{sup.max()} (median {int(np.median(sup))})  ->  stable {int(sup_s[0])}")
check("stable support is no worse than the WORST ordering the default could pick",
      sup_s[0] >= sup.min(), (sup_s[0], sup.min()))

print("\nsame answer on an unambiguous pool (no behaviour change where there was no ambiguity)")
clean = build_pool(seed=9, n_true=200, n_spur=40, jitter=0.001)
Ma, sa = consensus(clean, stable=False)
Mb, sb = consensus(clean, stable=True)
from glint.multishot import same_lattice
check("both find the same lattice", Ma is not None and Mb is not None and same_lattice(Ma, Mb))
check("and the same support when there is nothing to fragment", sa == sb, (sa, sb))

os.environ.pop("GLINT_CONSENSUS_STABLE", None)
print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
