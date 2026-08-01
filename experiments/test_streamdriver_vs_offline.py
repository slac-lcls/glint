"""Does the REAL StreamDriver, cold-started (Mc=None, discovers the cell itself) and fed the SAME
120 real cxidb frames as glint1_knowncell.py, reproduce hybrid_index's known-cell rate -- and how much
of the gap do the new warmup_rescue / watchdog-individual-rescue mechanisms close?

Bypasses ONLY the raw-pixel/peak-finding step (we already have q-vectors, not pixel frames, for this
real dataset) -- everything else (blind warm-up, RunningConsensus lock, batched index_fused steady
state, adaptive_relock watchdog, the two new rescue paths) is the real, unmocked StreamDriver code.

Scoring: monkey-patches _integrate_one (class-level) to capture every M the driver actually accepts
post-lock (via its own loose count-only gate, whether from direct known-cell success OR the new
watchdog individual rescue), and wraps the instance's _known_index to capture every (q, M) pair
attempted during a warm-up-buffer drain. Both get re-scored against the SAME strict gate
(same_lattice + >=25% + >=10 refl) glint1_knowncell.py uses, for a fair comparison to hybrid_index.

  python test_streamdriver_vs_offline.py
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, "/sdf/home/s/smarches/glint_streamfix_wt")
sys.path.insert(0, "/sdf/home/s/smarches/glint_streamfix_wt/experiments")
import glint.glint_fast as gf
from glint.glint_fast import matched
from glint.multishot import same_lattice
import glint.stream_driver as sd

LYSO = gf.LYSO
GATE_FRAC = 0.25
GATE_MIN = 10
FRAMES_PATH = "/sdf/home/s/smarches/glint_streamfix_wt/experiments/frames_cxidb_clean.txt"


def strict_gate(M, q, truth):
    if M is None:
        return False
    m = matched(M, q)
    return (m / len(q) >= GATE_FRAC) and (m >= GATE_MIN) and same_lattice(M, truth)


def push_blind_q(driver, qq):
    """Replica of StreamDriver._push_blind, minus the raw-pixel peak-finding step (we already have q).
    Mirrors the real function's warmup-buffer append too, so warmup_rescue is exercised correctly."""
    driver.n_pushed += 1
    if len(qq) >= driver.min_peaks:
        nb = driver._blind_index(qq, driver.warmup_nbest)
        driver._rc.add_frame([c for c, _ in nb])
        driver.n_warmup += 1
        if driver._warmup_buf is not None:
            driver._warmup_buf.append(qq)
    Mc, sup, _ = driver._rc.verdict()
    if Mc is not None:
        driver._lock(Mc, sup, standardize=True)


def run_stream(frames, adaptive_relock, warmup_rescue, tag):
    N = 400; pix_mm = 0.1; dist_mm = 100.0; wave_A = 1.0
    panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
                   res=1.0 / (pix_mm / 1000.0), cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5),
                   coffset=0.0, min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
    clen = dist_mm / 1000.0
    kw = dict(B=20, dmin=2.0, tol=0.002, warmup_nbest=3, warmup_rescue=warmup_rescue)
    if adaptive_relock:
        kw.update(adaptive_relock=True, min_inliers=GATE_MIN)
    drv = sd.StreamDriver(None, panels, clen, wave_A, (N, N), dtype=np.uint16, **kw)

    # --- instrumentation: capture every M the driver actually settles on ---
    postlock_capture = {}          # global frame idx -> M (post-lock: direct or watchdog-rescued)
    slot_to_global = {}            # current batch's slot -> global frame idx

    orig_integrate_one = sd.StreamDriver._integrate_one
    def patched_integrate_one(self, i, M, grid, acc):
        gi = slot_to_global.get(i)
        if gi is not None:
            postlock_capture[gi] = np.asarray(M, float).copy()
        return orig_integrate_one(self, i, M, grid, acc)
    sd.StreamDriver._integrate_one = patched_integrate_one

    warmup_capture = []            # list of (q, M) attempted during a warm-up-buffer drain
    real_known_index = drv._known_index
    def known_index_wrapper(qs, Mc, B=1):
        Ms = real_known_index(qs, Mc, B=B)
        warmup_capture.extend(zip(qs, Ms))
        return Ms
    drv._known_index = known_index_wrapper

    warmup_idx = []
    for i, qq in enumerate(frames):
        qq = np.asarray(qq, float)
        if drv._blind:
            warmup_idx.append(i)
            push_blind_q(drv, qq)
        else:
            slot = drv._n
            slot_to_global[slot] = i
            drv._q[slot] = qq
            drv._n += 1; drv.n_pushed += 1
            if drv._n == drv.B:
                drv.flush()
                slot_to_global.clear()
    drv.flush()
    sd.StreamDriver._integrate_one = orig_integrate_one   # unpatch the class-level hook

    locked = not drv._blind
    s = drv.stats() if locked else {}
    print(f"\n--- {tag} ---")
    print(f"  locked={locked}  locked_after={drv.locked_after}  n_warmup={len(warmup_idx)}  "
          f"n_pushed={drv.n_pushed}  same_lattice(driver.Mc,LYSO)={same_lattice(drv.Mc, LYSO) if locked else 'N/A'}")
    if not locked:
        print("  NEVER LOCKED -- cannot score.")
        return
    print(f"  driver's own counters: n_indexed={s.get('indexed')}  "
          f"n_warmup_rescued={s.get('n_warmup_rescued', 'n/a')}  "
          f"n_watchdog_rescued={s.get('n_watchdog_rescued', 'n/a')}  n_relock={s.get('n_relock', 'n/a')}")

    n_postlock_ok = sum(1 for gi, M in postlock_capture.items() if strict_gate(M, frames[gi], drv.Mc))
    n_warmup_ok = sum(1 for q, M in warmup_capture if strict_gate(M, q, drv.Mc))
    total = len(frames)
    grand = n_postlock_ok + n_warmup_ok
    print(f"  STRICT-GATE rescore: post-lock {n_postlock_ok}/{len(postlock_capture)} accepted-by-driver  +  "
          f"warm-up-rescue {n_warmup_ok}/{len(warmup_capture)} attempted")
    print(f"  STREAMING TOTAL (strict gate, same as hybrid_index/glint1_knowncell.py): "
          f"{grand}/{total} = {100*grand/total:.1f}%")


def main():
    t0 = time.time()
    frames = list(gf.load(FRAMES_PATH))
    n = len(frames)
    print(f"loaded {n} real cxidb frames from {FRAMES_PATH}")

    from glint.hybrid_stream import hybrid_index
    res, stats = hybrid_index(frames, Mc_known=LYSO, warmup=True)
    hh = [strict_gate(r["M"], q, LYSO) for r, q in zip(res, frames)]
    print(f"REFERENCE offline hybrid_index (same as glint1_knowncell.py): {sum(hh)}/{n} = "
          f"{100*sum(hh)/n:.1f}%  (n_idx={stats['n_idx']}, n_resc={stats['n_resc']}, "
          f"n_nbest={stats.get('n_nbest')})  [{time.time()-t0:.0f}s]")

    run_stream(frames, adaptive_relock=False, warmup_rescue=False, tag="baseline: both OFF")
    run_stream(frames, adaptive_relock=False, warmup_rescue=True, tag="warmup_rescue ONLY")
    run_stream(frames, adaptive_relock=True, warmup_rescue=False, tag="adaptive_relock ONLY (watchdog rescue)")
    run_stream(frames, adaptive_relock=True, warmup_rescue=True, tag="BOTH new mechanisms")

    print(f"\nDONE elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
