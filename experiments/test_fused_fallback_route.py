"""index_fused splits out frames with more peaks than fit in shared memory (issue #69) and runs them
through the stock torch ops instead, one at a time, then stitches the two lanes back together. Two
things there can go wrong silently: the fallback could give a different answer, and the stitch could
put a frame back in the wrong slot.

Force the split by lowering fused_kernels.max_peaks() to the median peak count -- so half of the real
120 cxidb frames take the fallback -- and compare per-frame against the all-fused run of the same
frames. fp64 must agree to round-off (the fused kernels are bit-exact ports of the stock ops); fp32
agrees on rate and lattice, differing only by equivalent-basis argmax ties, as index_fused documents.

Part 2 covers the OTHER entry point. index_fused hands its over-lane one frame at a time, but
run_fused() and any direct patch() caller hand the wrappers a whole padded batch, so each wrapper
splits the batch itself (fused_kernels._fallback) rather than passing an F x K x Pmax allocation
through to the stock op. That split is invisible to part 1, so call the three wrappers directly with
a multi-frame oversized batch and check each matches the stock op run on the batch whole. Part 3
executes obj_fused with enough candidates to require multiple grid.y blocks and checks both outputs
bit-for-bit against obj_b.

GPU node + cupy; exits 0 with a SKIP without them.

  KC_FP=64 PYTHONPATH=. python experiments/test_fused_fallback_route.py
  KC_FP=32 PYTHONPATH=. python experiments/test_fused_fallback_route.py
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                 # repo root (imports glint.*)
sys.path.insert(0, HERE)                                  # experiments/ (imports glint_fast)

# The skip guard covers ONLY the optional deps -- cupy, and a torch that can see a GPU. The modules
# under test are imported below it, unguarded, on purpose: catching everything here would turn a real
# import-time regression in glint/ into a green SKIP, and a check that cannot fail is not a check.
try:
    import cupy, torch                                    # noqa: F401  (cupy: fused_kernels needs it)
except ImportError as exc:
    print(f"SKIP test_fused_fallback_route: optional dependency missing ({exc})")
    sys.exit(0)
if not torch.cuda.is_available():
    print("SKIP test_fused_fallback_route: no CUDA device")
    sys.exit(0)

import glint.replica_gpu_batch as rgb                     # NOT inside the guard: must fail loudly
import glint.fused_kernels as fk
from glint_fast import load, gpass, LYSO
from glint.multishot import same_lattice

frames = [q for q in load(os.path.join(HERE, "frames_cxidb_clean.txt")) if len(q) >= 6]
n = len(frames); npk = np.array([len(f) for f in frames])
tag = "fp32" if rgb.FP == torch.float32 else "fp64"


def rate(Ms):
    g = np.array([gpass(M, q) for M, q in zip(Ms, frames)]); return int(g[:, 0].sum()), int(g[:, 1].sum())


A = rgb.index_fused(frames, LYSO, B=32)                   # every frame on the fused kernels
cut = int(np.median(npk))
orig = fk.max_peaks
fk.max_peaks = lambda: cut                                # force the split
try:
    B = rgb.index_fused(frames, LYSO, B=32)
finally:
    fk.max_peaks = orig

routed = int((npk > cut).sum())
dmax = max(float(np.abs(np.asarray(a, float) - np.asarray(b, float)).max()) for a, b in zip(A, B))
sl = sum(1 for a, b in zip(A, B) if same_lattice(np.asarray(a, float), np.asarray(b, float)))
nones = sum(1 for b in B if b is None)
rA, rB = rate(A), rate(B)

print(f"[{tag}] part 1 -- index_fused two-lane split")
print(f"   {n} cxidb frames, {npk.min()}-{npk.max()} peaks; ceiling forced to {cut} "
      f"-> {routed} frames on the non-fused fallback")
print(f"   rate      all-fused {rA}   split {rB}   None {nones}")
print(f"   per-frame max|dM| {dmax:.3e}   same_lattice {sl}/{n}")

fail = []
if routed == 0:               fail.append("nothing was routed to the fallback -- the test is vacuous")
if rA != rB:                  fail.append(f"rate changed {rA} -> {rB}")
if sl != n:                   fail.append(f"same_lattice {sl}/{n}")
if nones:                     fail.append(f"{nones} frames came back None")
if tag == "fp64" and dmax > 1e-9:
    fail.append(f"fp64 fallback is not bit-faithful (max|dM| {dmax:.3e})")

# ---------------------------------------------------------------------------------------------
# Part 2: the wrappers' OWN split. index_fused never hands its over-lane more than one frame, so
# part 1 cannot see fused_kernels._fallback. run_fused()/patch() callers do hand over a whole padded
# batch, so call the three wrappers directly with the ceiling forced to 0 and check (a) each stock op
# was invoked once PER FRAME rather than once on the F x K x Pmax batch, and (b) the concatenated
# result still equals the stock op applied to the batch whole.
Fd = 8
fb = frames[:Fd]; Pm = max(len(f) for f in fb)
Qd, md = rgb.pad(fb, Pm)
Mcn = np.asarray(LYSO, float)
gen = torch.Generator().manual_seed(0)
u = torch.randn(Fd, 4, 3, generator=gen); u = u / u.norm(dim=2, keepdim=True)
Vd = (u * float(np.linalg.norm(Mcn, axis=0).max())).to(dtype=rgb.FP, device=rgb.DEV)
M0d = torch.as_tensor(Mcn, dtype=rgb.FP, device=rgb.DEV).expand(Fd, 4, 3, 3).contiguous()
M0d = M0d * (1.0 + 0.01 * torch.arange(4, dtype=rgb.FP, device=rgb.DEV)[None, :, None, None])

want = {"anneal_b": fk._ORIG["anneal_b"], "obj_b": fk._ORIG["obj_b"], "refine_b": fk._ORIG["refine_b"]}
exp = {"obj_b": want["obj_b"](Vd, Qd, md),                       # the stock ops on the WHOLE batch
       "refine_b": want["refine_b"](Vd, Qd, md, 5),
       "anneal_b": want["anneal_b"](M0d, Qd, md, max_iter=5)}
calls = {k: 0 for k in want}


def _counted(name):
    def f(*a, **k):
        calls[name] += 1
        return want[name](*a, **k)
    return f


fk.max_peaks = lambda: 0                                  # force EVERY wrapper down the split path
fk._ORIG.update({k: _counted(k) for k in want})
try:
    got = {"obj_b": fk.obj_fused(Vd, Qd, md),
           "refine_b": fk.refine_fused(Vd, Qd, md, steps=5),
           "anneal_b": fk.anneal_fused(M0d, Qd, md, max_iter=5)}
finally:
    fk.max_peaks = orig
    fk._ORIG.update(want)

tol = 1e-10 if tag == "fp64" else 1e-4
print(f"[{tag}] part 2 -- wrapper-level split of a {Fd}-frame oversized batch (Pmax={Pm})")
for name in ("obj_b", "refine_b", "anneal_b"):
    g_, e_ = got[name], exp[name]
    gts = g_ if isinstance(g_, tuple) else (g_,); ets = e_ if isinstance(e_, tuple) else (e_,)
    shapes = all(a.shape == b.shape for a, b in zip(gts, ets))
    rel = max(float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp(min=1))
              for a, b in zip(gts, ets))
    exact = all(bool(torch.equal(a, b)) for a, b in zip(gts, ets) if not a.is_floating_point())
    print(f"   {name:9s} stock calls {calls[name]:2d} (want {Fd})   shapes {'ok' if shapes else 'MISMATCH'}"
          f"   max rel|d| {rel:.2e}   integer outputs {'equal' if exact else 'DIFFER'}")
    if calls[name] != Fd:
        fail.append(f"{name} fallback made {calls[name]} stock calls for {Fd} frames "
                    f"(the batch was not split per frame)")
    if not shapes:  fail.append(f"{name} fallback returned the wrong shape")
    if not exact:   fail.append(f"{name} fallback changed an integer output")
    if rel > tol:   fail.append(f"{name} fallback differs from the whole-batch stock op (rel {rel:.2e})")

# ---------------------------------------------------------------------------------------------
# Part 3: candidate-split obj path. K > block is the condition that makes grid.y > 1.
Ksplit = 257
us = torch.randn(Fd, Ksplit, 3, generator=gen)
Vs = (us / us.norm(dim=2, keepdim=True) * float(np.linalg.norm(Mcn, axis=0).max())).to(
    dtype=rgb.FP, device=rgb.DEV)
want_inl, want_sub = fk._ORIG["obj_b"](Vs, Qd, md)
got_inl, got_sub = fk.obj_fused(Vs, Qd, md)
inl_exact = bool(torch.equal(got_inl, want_inl))
sub_exact = bool(torch.equal(got_sub, want_sub))
print(f"[{tag}] part 3 -- obj candidate split at K={Ksplit}: "
      f"inl {'equal' if inl_exact else 'DIFFER'}   sub {'equal' if sub_exact else 'DIFFER'}")
if not inl_exact: fail.append("candidate-split obj changed inl")
if not sub_exact: fail.append("candidate-split obj changed sub")

# ---------------------------------------------------------------------------------------------
# Part 4: the polished-vs-best guard must match the stock host tail on mean-normalised boundaries.
s = (1.0 / 0.905) ** (1.0 / 3.0)
best_np = np.stack([Mcn, Mcn])
pol_np = np.stack([Mcn * s, np.diag([Mcn[0, 0] / 1.051, Mcn[1, 1], Mcn[2, 2]])])
best_t = torch.as_tensor(best_np, dtype=rgb.FP, device=rgb.DEV)
pol_t = torch.as_tensor(pol_np, dtype=rgb.FP, device=rgb.DEV)
mp_t = torch.tensor([5, 5], device=rgb.DEV)
main_t = torch.tensor([5, 5], device=rgb.DEV)
want = rgb._cpu_stage(best_t, pol_t, mp_t, main_t, Mcn)
got = fk._cpu_stage_gpu(best_t, pol_t, mp_t, main_t, Mcn)
host_pol = [bool(np.allclose(a, b)) for a, b in zip(want, pol_np)]
fused_pol = [bool(np.allclose(a, b)) for a, b in zip(got, pol_np)]
print(f"[{tag}] part 4 -- cpu-stage boundary parity: host {host_pol}   fused {fused_pol}")
if host_pol != [True, True]:
    fail.append(f"stock _cpu_stage missed the boundary cases {host_pol}")
if fused_pol != host_pol:
    fail.append(f"_cpu_stage_gpu diverged from stock on the boundary cases {fused_pol} vs {host_pol}")

print("   " + ("FAIL: " + "; ".join(fail) if fail else "PASS"))
sys.exit(1 if fail else 0)
