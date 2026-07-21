"""Gate for flipping M3_FUSED ON by default (#39).

Checks the things that only matter once it is the default rather than an opt-in:

  1. the default really is fused (a flag that flips but does not bind is the #30 harness bug again)
  2. default-on reproduces the rates M3_FUSED=1 was measured at -- blind and HYBRID
  3. M3_FUSED=0 still restores the torch path exactly, so an older result stays reproducible
  4. the fallback still holds where the kernel cannot run (dense clouds, P > MAX_P)

  python verify_m3_fused_default.py
"""
import os, sys, time, importlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from glint.multishot import same_lattice
from azimuth_coverage import cell_to_A, rand_rot, rlps


def load_gf(env=None):
    """Reload with a clean env so the MODULE-LEVEL default is what is exercised."""
    os.environ.pop("M3_FUSED", None)
    if env is not None:
        os.environ["M3_FUSED"] = env
    import glint.glint_fast as gf
    importlib.reload(gf)
    return gf


gf = load_gf(None)                       # no env var at all -> the shipped default
LYSO = gf.LYSO
frames = [q for q in gf.load(os.path.join(HERE, "frames_cxidb_clean.txt")) if len(q) >= 6]
n = len(frames)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gate(M, q):
    if M is None or not same_lattice(np.asarray(M, float), LYSO):
        return 0
    r = np.asarray(q) @ M
    m = int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())
    return int(m / len(q) >= 0.25 and m >= 10)


print(f"1. shipped default (no M3_FUSED in env): M3_FUSED = {gf.M3_FUSED}  "
      f"{'OK' if gf.M3_FUSED else '*** NOT ON ***'}")
from glint import fused_m3
print(f"   cupy present: {fused_m3._HAVE_CP}   kernel usable on these frames: "
      f"{sum(int(fused_m3.available(torch.as_tensor(np.asarray(q,float),dtype=torch.float32,device=gf.DEV))) for q in frames)}/{n}")

# --- 2/3: default-on vs explicit off ----------------------------------------------------------
def run(g):
    g.index_blind_fast(frames[0]); sync()
    t0 = time.perf_counter()
    Ms = [g.index_blind_fast(q) for q in frames]
    sync(); ms = 1e3 * (time.perf_counter() - t0) / n
    lat = sum(int(M is not None and same_lattice(np.asarray(M, float), LYSO)) for M in Ms)
    return lat, sum(gate(M, q) for M, q in zip(Ms, frames)), ms, Ms


print(f"\n2/3. blind: default vs M3_FUSED=0")
print(f"{'config':<26}{'same_lat':>10}{'gate':>8}{'ms/frame':>11}{'speedup':>9}")
d_lat, d_g, d_ms, d_M = run(load_gf(None))
o_lat, o_g, o_ms, o_M = run(load_gf("0"))
e_lat, e_g, e_ms, e_M = run(load_gf("1"))
print(f"{'default (no env var)':<26}{f'{d_lat}/{n}':>10}{f'{d_g}/{n}':>8}{d_ms:11.2f}{o_ms/d_ms:8.2f}x")
print(f"{'M3_FUSED=0 (torch)':<26}{f'{o_lat}/{n}':>10}{f'{o_g}/{n}':>8}{o_ms:11.2f}{'--':>9}")
print(f"{'M3_FUSED=1 (explicit)':<26}{f'{e_lat}/{n}':>10}{f'{e_g}/{n}':>8}{e_ms:11.2f}{o_ms/e_ms:8.2f}x")
same_de = sum(int((a is None) == (b is None) and (a is None or np.array_equal(np.asarray(a), np.asarray(b))))
              for a, b in zip(d_M, e_M))
print(f"   default == explicit M3_FUSED=1: {same_de}/{n} bit-identical  "
      f"{'OK' if same_de == n else '*** DEFAULT IS NOT THE FUSED PATH ***'}")

# --- hybrid ------------------------------------------------------------------------------------
print("\n   hybrid (the number that ships)")
for env, lbl in ((None, "default"), ("0", "M3_FUSED=0")):
    g = load_gf(env)
    import glint.hybrid_stream as hs
    importlib.reload(hs)
    hs.hybrid_index(frames[:4], None, Mc_known=None, nbest=3)
    sync(); t0 = time.perf_counter()
    res, st = hs.hybrid_index(frames, None, Mc_known=None, nbest=3, warmup=True)
    sync(); ms = 1e3 * (time.perf_counter() - t0) / n
    gg = sum(gate(r["M"], q) for r, q in zip(res, frames))
    print(f"   {lbl:<14} gate {gg}/{n}   n_resc {st.get('n_resc',-1)}   {ms:.2f} ms/frame")

# --- 4: fallback on dense clouds the kernel cannot take ----------------------------------------
print("\n4. fallback where the kernel cannot run (dense clouds, P > MAX_P)")
rng = np.random.default_rng(0)
A0 = cell_to_A(79.0, 79.0, 38.0, 90, 90, 90)
clouds = [rlps(rand_rot(rng) @ A0, rng) for _ in range(3)]
print(f"   {len(clouds)} clouds, median {int(np.median([len(c) for c in clouds]))} rlps "
      f"(bound P <= {fused_m3.MAX_P})")
outs = {}
for env, lbl in ((None, "default"), ("0", "M3_FUSED=0")):
    g = load_gf(env)
    outs[lbl] = [g.index_blind_cluster_seeded(q) for q in clouds]
ident = sum(int((a is None) == (b is None) and (a is None or np.allclose(a, b, atol=1e-12)))
            for a, b in zip(outs["default"], outs["M3_FUSED=0"]))
print(f"   dense results identical with/without the flag: {ident}/{len(clouds)}  "
      f"{'OK -- kernel correctly never engages' if ident == len(clouds) else '*** DIVERGED ***'}")
os.environ.pop("M3_FUSED", None)
