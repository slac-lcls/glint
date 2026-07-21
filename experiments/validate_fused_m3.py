"""Integration gate for M3_FUSED (#28): the two things bench_fused_m3.py did NOT cover.

That bench measured the BLIND path on sparse cxidb frames. Before wiring the kernel in, two gaps:

  1. HYBRID rate -- blind is not what ships. Consensus + rescue sit downstream, and a front end that
     perturbs 3/120 cells could move the gated number either way.
  2. DENSE / cluster-seeded path -- index_blind_cluster_seeded and index_blind_nbest also route
     through _refine. Dense clouds carry FAR more peaks than the 3008 the kernel's shared-memory
     bound allows, so what is really being tested there is that the FALLBACK engages and leaves the
     torch result untouched.

Both run through the real M3_FUSED flag rather than a monkeypatch, so this exercises the dispatch
that would actually ship.

  python validate_fused_m3.py
"""
import os, sys, time, importlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from glint.multishot import same_lattice
from glint.lattice import cell_to_Ar
from azimuth_coverage import cell_to_A, rand_rot, rlps


def load_gf(fused):
    """Re-import glint_fast with M3_FUSED set, so the real module-level flag is exercised."""
    os.environ["M3_FUSED"] = "1" if fused else "0"
    import glint.glint_fast as gf
    importlib.reload(gf)
    return gf


gf0 = load_gf(False)
frames = [q for q in gf0.load(os.path.join(HERE, "frames_cxidb_clean.txt")) if len(q) >= 6]
n = len(frames)
LYSO = gf0.LYSO


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gate(M, q):
    if M is None or not same_lattice(np.asarray(M, float), LYSO):
        return 0
    r = np.asarray(q) @ M
    m = int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())
    return int(m / len(q) >= 0.25 and m >= 10)


# ---- 1. HYBRID: the number that ships --------------------------------------------------------
print(f"1. HYBRID on {n} cxidb frames (blind front end + consensus + rescue)")
print(f"{'M3':<12}{'gate':>9}{'same_lat':>10}{'n_idx':>8}{'n_resc':>8}{'ms/frame':>11}{'speedup':>9}")
rows = {}
for fused in (False, True):
    gf = load_gf(fused)
    from glint.hybrid_stream import hybrid_index
    importlib.reload(sys.modules["glint.hybrid_stream"])
    from glint.hybrid_stream import hybrid_index
    hybrid_index(frames[:4], None, Mc_known=None, nbest=3)      # warm
    sync(); t0 = time.perf_counter()
    res, st = hybrid_index(frames, None, Mc_known=None, nbest=3, warmup=True)
    sync(); ms = 1e3 * (time.perf_counter() - t0) / n
    Ms = [r["M"] for r in res]
    g = sum(gate(M, q) for M, q in zip(Ms, frames))
    sl = sum(int(M is not None and same_lattice(np.asarray(M, float), LYSO)) for M in Ms)
    rows["fused" if fused else "torch"] = (g, sl, ms, Ms)
    base = rows["torch"][2]
    print(f"{'fused' if fused else 'torch':<12}{f'{g}/{n}':>9}{f'{sl}/{n}':>10}"
          f"{st.get('n_idx', -1):>8}{st.get('n_resc', -1):>8}{ms:11.2f}{base/ms:8.2f}x")
ident = sum(int((a is None) == (b is None) and (a is None or np.allclose(a, b, atol=1e-6)))
            for a, b in zip(rows["torch"][3], rows["fused"][3]))
print(f"   hybrid cells identical: {ident}/{n}"
      f"   gate delta: {rows['fused'][0] - rows['torch'][0]:+d}")

# ---- 2. DENSE / cluster-seeded: does the FALLBACK hold? --------------------------------------
print("\n2. DENSE cluster-seeded path (peaks far exceed the kernel's shared-memory bound)")
CELLS = [("lyso_tet", (79.0, 79.0, 38.0, 90, 90, 90)),
         ("tricl",    (45.0, 55.0, 65.0, 80, 85, 95)),
         ("hex_P6",   (105.0, 105.0, 75.0, 90, 90, 120))]
rng = np.random.default_rng(0)
clouds = []
for name, cp in CELLS:
    A0 = cell_to_A(*cp)
    for r in range(3):
        Ar = rand_rot(rng) @ A0
        q = rlps(Ar, rng)
        clouds.append((name, np.sort(np.array(cp[:3], float)), q))
print(f"   {len(clouds)} dense clouds, median {int(np.median([len(c[2]) for c in clouds]))} rlps"
      f"  (kernel bound is P <= 3008)")

from glint import fused_m3
takes = sum(int(fused_m3.available(torch.as_tensor(c[2], dtype=torch.float32, device=gf0.DEV)))
            for c in clouds)
print(f"   clouds the kernel would accept: {takes}/{len(clouds)}  -> fallback must carry the rest")


def ok(M, axes):
    if M is None:
        return False
    L = np.sort(np.linalg.norm(np.asarray(M, float), axis=0))
    return bool(np.all(np.abs(L - axes) <= 0.05 * axes))


dense = {}
for fused in (False, True):
    gf = load_gf(fused)
    gf.index_blind_cluster_seeded(clouds[0][2]); sync()
    t0 = time.perf_counter()
    Ms = [gf.index_blind_cluster_seeded(q) for _, _, q in clouds]
    sync(); ms = 1e3 * (time.perf_counter() - t0) / len(clouds)
    good = sum(ok(M, ax) for M, (_, ax, _) in zip(Ms, clouds))
    dense["fused" if fused else "torch"] = (good, ms, Ms)
    print(f"   M3={'fused' if fused else 'torch':<6} correct-cell {good}/{len(clouds)}  {ms:7.1f} ms/cloud")
d_ident = sum(int((a is None) == (b is None) and (a is None or np.allclose(a, b, atol=1e-9)))
              for a, b in zip(dense["torch"][2], dense["fused"][2]))
print(f"   dense cells identical: {d_ident}/{len(clouds)}"
      f"   (must be {len(clouds)}/{len(clouds)} -- the kernel should never engage here)")

os.environ["M3_FUSED"] = "0"
print("\nGate: hybrid must not lose points, and the dense path must be untouched.")
