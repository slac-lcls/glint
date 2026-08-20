"""CPU tests for the opt-in lock-quality probe in glint.stream_driver (lock_probe / lock_min_z).

No GPU/torch: the per-frame known-cell indexer (`_known_index`) and blind fan-out are injected as fakes
through the same seams as test_missbuf_rescue. The probe scores the new lock's best supporting frame
against the random-orientation null (glint.spurious_meter.null_margin) and records a z; lock_min_z refuses
a lock whose z is below the bar. Run: `python experiments/test_lock_probe.py` or `pytest`.
"""
import numpy as np
from glint.stream_driver import StreamDriver
from glint.multishot import same_lattice
from glint.alias_gate import AliasGate


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


M_A = _rot(0.5) @ np.diag([1 / 79.0, 1 / 79.0, 1 / 38.0])
M_B = _rot(0.9) @ np.diag([1 / 95.0, 1 / 95.0, 1 / 45.0])
Mi_B = np.linalg.inv(M_B.T)                                    # q -> hkl index matrix (q @ Mi_B = hkl)
N = 32
PANELS = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=1e4,
              cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5), coffset=0.0,
              min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]


def _lattice_q(rng, n=22):
    """q exactly on cell B's lattice -> _inliers(q, Mi_B) == len(q)."""
    hkl = rng.integers(-6, 7, size=(n, 3)).astype(float)
    hkl = hkl[np.abs(hkl).sum(1) > 0]
    return hkl @ M_B.T


def _offlattice_q(rng, n=22):
    """half-integer shifted -> ZERO inliers under Mi_B."""
    hkl = rng.integers(-6, 7, size=(n, 3)).astype(float)
    hkl = hkl[np.abs(hkl).sum(1) > 0]
    return (hkl + 0.5) @ M_B.T


def _fake_known(qs, Mn, B):
    """index_fused stand-in: returns cell B's index matrix for any frame with >=6 peaks, else None."""
    return [Mi_B.copy() if (q is not None and len(q) >= 6) else None for q in qs]


def _driver(lock_probe=False, lock_min_z=None, probe_null=64):
    drv = StreamDriver(M_A, PANELS, 0.1, 1.3, (N, N), dtype=np.uint16, B=16,
                       dmin=3.0, use_gpu=False, adaptive_relock=True, min_inliers=6,
                       lock_probe=lock_probe, lock_min_z=lock_min_z, probe_null=probe_null)
    drv._known_index = _fake_known
    drv._fanout = lambda Q, k: [[(M_B.copy(), 1.0)] for _ in Q]     # every missed frame votes B -> locks
    return drv


def _fill(drv, qfn, rng, k=5):
    missed = list(range(k))
    for i in missed:
        drv._q[i] = qfn(rng)
    return missed


# ------------------------------------------------------------------------------------------- tests
def test_probe_records_high_z_on_real_lock():
    rng = np.random.default_rng(1)
    drv = _driver(lock_probe=True)
    drv._watchdog(_fill(drv, _lattice_q, rng))
    assert drv.n_relock == 1 and len(drv.extra) == 1
    z = drv.extra[0]["lock_z"]
    assert z is not None and z > 4.0, z                         # true fit towers over the random-orientation null
    assert drv.stats()["lock_z"] == z


def test_lock_min_z_refuses_weak_lock():
    """Frames that don't actually fit the voted cell -> lock_z ~ 0 -> lock_min_z refuses; watch not reset."""
    rng = np.random.default_rng(2)
    drv = _driver(lock_probe=True, lock_min_z=3.0)
    drv._watchdog(_fill(drv, _offlattice_q, rng, k=5))
    assert drv.n_relock == 0 and len(drv.extra) == 0, "weak lock should have been refused"
    assert drv.lock_z == 0.0
    # the refused lock must NOT reset the histogram: this batch's 5 votes are retained so a later,
    # stronger batch can still lock (an empty/reset watch would report nframes == 0)
    assert drv._watch is not None and drv._watch.nframes == 5, drv._watch
    assert drv._watch.leaders()[1] >= 3, "accumulated support preserved"


def test_probe_annotate_only_does_not_block():
    """lock_probe on, lock_min_z None: z is recorded but never blocks the lock."""
    rng = np.random.default_rng(3)
    drv = _driver(lock_probe=True, lock_min_z=None)
    drv._watchdog(_fill(drv, _offlattice_q, rng))              # even a weak frame still locks (annotate only)
    assert drv.n_relock == 1 and drv.extra[0]["lock_z"] == 0.0


def test_default_off_no_probe():
    rng = np.random.default_rng(4)
    drv = _driver(lock_probe=False)
    drv._watchdog(_fill(drv, _lattice_q, rng))
    assert drv.n_relock == 1 and drv.extra[0]["lock_z"] is None
    assert drv.lock_z is None
    assert "lock_z" not in drv.stats()                        # not surfaced when the probe is off


# ---------------------------------------------------------------- driver-level alias-gate wiring
# q@M=hkl convention (M columns = real cell vectors): tetragonal so _conventional_tetragonal is stable.
M_REAL = np.diag([60.0, 60.0, 80.0])
M_ALIAS = M_REAL @ np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 2.0]])   # index-2 super-cell (c doubled)
_MINV = np.linalg.inv(M_REAL)


def _real_frame_q(rng, n=40):
    """q of a real M_REAL frame (one orientation): q @ M_REAL = integer hkl, q @ M_ALIAS = hkl@H integer
    too (super-cell keeps coverage) -- the exact case the gate must resolve to the tighter true cell."""
    hkl = rng.integers(-5, 6, size=(n, 3)).astype(float)
    hkl = hkl[np.abs(hkl).sum(1) > 0]
    return hkl @ _MINV


def _alias_driver(gate):
    drv = StreamDriver(M_A, PANELS, 0.1, 1.3, (N, N), dtype=np.uint16, B=16,
                       dmin=3.0, use_gpu=False, adaptive_relock=True, min_inliers=6, alias_gate=gate)
    drv._fanout = lambda Q, k: [[(M_ALIAS.copy(), 1.0)] * 3 for _ in Q]   # consensus only ever sees the alias
    return drv


def test_watchdog_rescue_checks_active_cells_not_the_previous_frame():
    """The watchdog's rescue check must compare each missed frame against the ACTIVE cells, not against
    whatever the previous frame blind-indexed to. Runs with NO alias gate: this is the shipped default.

    Regression for #112, which rebound the loop's `cells` variable -- the active-cell list the rescue
    checks against -- to the current frame's own N-best candidates. From the second unrescued frame on,
    every frame was then matched against the PREVIOUS frame's candidates, which for a recurring new cell
    is that same cell: a guaranteed self-match. Each such frame was "rescued" into cells[0]'s grid and
    accumulator -- the locked cell, the wrong one entirely -- and dropped from the consensus that
    existed to detect it. Here no active cell explains these frames (driver locked on A, frames are on
    M_REAL), so a correct watchdog rescues none of them and all five reach the vote.

    #112 claimed `alias_gate=None` left the shipped path bit-identical; this is the assertion that says
    so. It also guards the gate tests below, which cannot see enough frames to reach `min_frames` when
    the rescue eats them."""
    rng = np.random.default_rng(7)
    drv = _alias_driver(None)                                  # default: gate off
    missed = _fill(drv, _real_frame_q, rng, k=5)
    routed = []
    drv._integrate_one = lambda i, c, g, a, cell_id=0, **kw: routed.append((i, cell_id))
    drv._watchdog(missed)
    assert drv.n_watchdog_rescued == 0, \
        f"no active cell explains these frames, yet {drv.n_watchdog_rescued}/5 were 'rescued'"
    assert routed == [], f"frames misrouted into an existing cell's accumulator: {routed}"


def test_alias_gate_off_locks_alias():
    """No gate: the injected super-cell alias wins consensus and the driver locks it (the failure the gate
    exists to prevent)."""
    rng = np.random.default_rng(7)
    drv = _alias_driver(None)
    missed = _fill(drv, _real_frame_q, rng, k=5)
    drv._watchdog(missed)
    assert drv.n_relock == 1 and same_lattice(drv.extra[0]["Mc"], M_ALIAS), \
        [np.linalg.norm(drv.extra[0]["Mc"], axis=0)]


def test_alias_gate_adopt_recovers_true():
    """adopt gate: even though consensus only ever saw the alias, the gate finds the true cell is the
    tightest of the alias's derivatives on the best-fitting frame and adopts it -> locks TRUE, not alias."""
    rng = np.random.default_rng(7)
    drv = _alias_driver(AliasGate(adopt=True))
    missed = _fill(drv, _real_frame_q, rng, k=5)
    drv._watchdog(missed)
    assert drv.n_relock == 1, drv.n_relock
    assert same_lattice(drv.extra[0]["Mc"], M_REAL), np.linalg.norm(drv.extra[0]["Mc"], axis=0)
    assert not same_lattice(drv.extra[0]["Mc"], M_ALIAS), "should have repaired away from the alias"


if __name__ == "__main__":
    tests = (test_probe_records_high_z_on_real_lock, test_lock_min_z_refuses_weak_lock,
             test_probe_annotate_only_does_not_block, test_default_off_no_probe,
             test_watchdog_rescue_checks_active_cells_not_the_previous_frame,
             test_alias_gate_off_locks_alias, test_alias_gate_adopt_recovers_true)
    ok = 0
    for t in tests:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            import traceback
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}"); traceback.print_exc()
    print(f"{ok}/{len(tests)} passed")
    raise SystemExit(0 if ok == len(tests) else 1)
