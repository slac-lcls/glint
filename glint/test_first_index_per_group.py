"""`_first_index_per_group` must equal `scatter_reduce_(amin)` EXACTLY, on every torch that has both.

The shim exists so GLINT runs on torch 1.11 (`ana-4.0.59`, the only LCLS release whose psana parses
Jungfrau.ConfigV4). It sits under lattice-candidate dedup, so a wrong answer does not crash -- it
silently changes WHICH candidates survive, and the indexing rate moves for reasons nobody can trace.
That makes exactness the whole point, and the reference is only available on torch >= 1.12, so this
runs in two modes:

    torch >= 1.12  compare against scatter_reduce_ directly -- the real check
    torch 1.11     no reference; check it runs, is self-consistent, and satisfies the DEFINING
                   property (out[g] == min{i : inv[i]==g}) computed independently in python

Run it in BOTH, since the point is that they agree.

    python -m glint.test_first_index_per_group
"""
from __future__ import annotations

import sys

import torch

from glint.glint_index import _first_index_per_group

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HAVE_REF = hasattr(torch.Tensor, "scatter_reduce_")
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + detail}")
    if not ok:
        FAILS.append(name)


def reference(inv, G, N):
    out = torch.full((G,), N, dtype=torch.long, device=inv.device)
    out.scatter_reduce_(0, inv, torch.arange(N, device=inv.device), reduce="amin",
                        include_self=True)
    return out


def brute(inv, G, N):
    """The definition, in python. Slow and obviously correct."""
    out = [N] * G
    for i, g in enumerate(inv.tolist()):
        if i < out[g]:
            out[g] = i
    return torch.tensor(out, dtype=torch.long, device=inv.device)


def cases():
    g = torch.Generator(device="cpu").manual_seed(0)
    for N, G in [(1, 1), (2, 1), (3, 2), (5, 5), (64, 64), (100, 7), (1200, 300), (5000, 4096)]:
        yield f"random N={N} G={G}", torch.randint(0, G, (N,), generator=g).to(DEV), G, N
    yield "every row its own group", torch.arange(50, device=DEV), 50, 50
    yield "one group only", torch.zeros(37, dtype=torch.long, device=DEV), 1, 37
    # groups 1..3 and 5..8 are EMPTY and must come back as N, not 0
    yield "empty groups in the middle", torch.tensor([0, 0, 4, 4, 4, 9], device=DEV), 10, 6
    yield "N=0", torch.empty(0, dtype=torch.long, device=DEV), 3, 0
    yield "no rows, no groups", torch.empty(0, dtype=torch.long, device=DEV), 0, 0
    # the shape torch.unique actually produces: monotone non-decreasing inv
    yield ("monotone inv (what torch.unique gives)",
           torch.repeat_interleave(torch.arange(20, device=DEV), torch.arange(1, 21, device=DEV)),
           20, 210)
    # every row in the LAST group -- exercises the run-boundary logic at the end of the sort
    yield "all rows in the last group", torch.full((40,), 6, dtype=torch.long, device=DEV), 7, 40


print(f"torch {torch.__version__}  device {DEV}  reference available: {HAVE_REF}")

print("\nagainst the definition (brute force)")
for name, inv, G, N in cases():
    got = _first_index_per_group(inv.to(torch.long), G, N)
    want = brute(inv.to(torch.long), G, N)
    check(name, torch.equal(got, want),
          f"got {got.tolist()[:8]} want {want.tolist()[:8]}")

if HAVE_REF:
    print("\nagainst torch's scatter_reduce_(amin) -- the API being replaced")
    for name, inv, G, N in cases():
        got = _first_index_per_group(inv.to(torch.long), G, N)
        want = reference(inv.to(torch.long), G, N)
        check(name, torch.equal(got, want),
              f"got {got.tolist()[:8]} want {want.tolist()[:8]}")
else:
    print("\nscatter_reduce_ absent (torch < 1.12) -- brute force above IS the check")

print("\ndeterminism: the same input 200 times must give the same answer")
# The natural one-liner for this (reverse-order assignment, last write wins) is NOT deterministic on
# CUDA -- measured at 10/13 wrong and unstable across repeats. This is the guard against
# reintroducing it.
inv = torch.randint(0, 300, (1200,), generator=torch.Generator().manual_seed(1)).to(DEV)
first = _first_index_per_group(inv, 300, 1200)
check("200 repeats identical",
      all(torch.equal(_first_index_per_group(inv, 300, 1200), first) for _ in range(200)))

print("\ndtype and device contract")
out = _first_index_per_group(torch.tensor([0, 1, 1], device=DEV), 2, 3)
check("returns long", out.dtype == torch.long, str(out.dtype))
check("returns on the input's device", out.device.type == torch.device(DEV).type, str(out.device))
check("int32 inv accepted", torch.equal(
    _first_index_per_group(torch.tensor([0, 1, 1], dtype=torch.int32, device=DEV), 2, 3),
    torch.tensor([0, 1], dtype=torch.long, device=DEV)))

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
