"""glint_cli --gate floor: the strict bar plus a per-peak-count chance floor fitted for the CLI's own search.

WHY THIS EXISTS. --gate strict (glint#216) writes a frame as a crystal when >= GATE_MIN peaks and >= GATE_FRAC of
its peaks are matched. With --cell, the registration it judges is the best of the CLI route's searches (blind
N-best, then the per-frame index_known_gpu_cell rescue), and on a SPARSE frame with no lattice at all that best
count clears 25 % of a few dozen peaks (23 % of scrambled cxidb-17 copies below 60 peaks). --gate floor adds matched >= a*n + b + c*sqrt(n), the
99th percentile of that route's count on azimuth-scrambled frames (experiments/cli_gate_null.py,
RESULTS_cli_gate_null.md); --floor names a calibration (cxidb17) or gives the coefficients.

WHAT IT CHECKS.
  1. resolve_floor / gate_results on synthetic frames with exact matched counts: the named constants resolve;
     bad floors raise (a set, NaN, inf, booleans, wrong length, an unknown name); "floor" without floor= and
     floor= with another gate raise. A sparse frame at the measured chance level passes strict and is withdrawn
     by floor; the controls -- a sparse frame at the level of the real frames floor keeps, and a dense one --
     are kept by both; a dense frame under 25 % is withdrawn by both (floor includes strict).
  2. Real cxidb-17 frames through the CLI route (hybrid_index, Mc_known = LYSO, CPU torch): on scrambled copies
     of the sparse frames strict accepts many (the defect) and floor almost none; floor keeps nearly every real
     frame strict keeps.
  3. The CLI end to end (--qframes, --cell, CPU): scrambled frames that --gate strict WRITES are not written by
     --gate floor --floor cxidb17, the real controls are; the coefficients spelled out give the same stream;
     --gate floor without --floor, --floor without --gate floor, and bad --floor values are refused; the default
     (no --gate) stream is byte-identical to --gate none.

SKIPS, exit 0, without torch (glint.glint_fast imports it; the CPU CI job installs none) or below pyproject's
torch floor (>= 1.12), like test_cli_gate.py.

  PYTHONPATH=. python experiments/test_cli_gate_floor.py
"""
import copy
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:
    import torch
except ImportError:
    print(f"SKIP {os.path.basename(__file__)} -- no torch: glint.glint_fast cannot import here")
    sys.exit(0)
_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if _ver < (1, 12):
    print(f"SKIP {os.path.basename(__file__)} -- torch {torch.__version__} is below pyproject's "
          f"floor (>= 1.12); the indexer is not supported there")
    sys.exit(0)

import subprocess                                                          # noqa: E402
import tempfile                                                            # noqa: E402

import numpy as np                                                         # noqa: E402
from glint.glint_fast import GATE_FRAC, GATE_MIN, LYSO, load, matched_strict   # noqa: E402
from glint.hybrid_stream import (CLI_NULL_FLOORS, GATES, floor_value, gate_results,   # noqa: E402
                                 hybrid_index, resolve_floor)
from glint.multilattice import scramble_azimuth                            # noqa: E402

FRAMES = os.path.join(HERE, "frames_cxidb_clean.txt")
FLOOR = CLI_NULL_FLOORS["cxidb17"]
SEED = 20260930
fails = []


def check(name, cond, msg=""):
    print(f"  {name:78s}: {'PASS' if cond else 'FAIL'}   {msg}")
    if not cond:
        fails.append(name)


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception:                                                   # noqa: BLE001
        return False
    return False


def strict_ok(m, n):
    return m >= GATE_MIN and m >= GATE_FRAC * n


# ------------------------------------------------------------------------------------------ part 1 ----
def _frame(M, n_on, n_off, rng):
    """n_on peaks on lattice M and n_off whose hkl sit at half-integers (residual 0.5 > GATE_TOL on every
    component), so matched_strict under M is exactly n_on."""
    hkl = rng.integers(-7, 8, size=(4 * (n_on + n_off), 3)).astype(float)
    hkl = np.unique(hkl[np.abs(hkl).sum(1) > 0], axis=0)
    hkl = hkl[rng.permutation(len(hkl))]
    q = np.vstack([hkl[:n_on], hkl[n_on:n_on + n_off] + 0.5]) @ np.linalg.inv(M)
    assert matched_strict(M, q) == n_on and len(q) == n_on + n_off, "fixture: counts are not exact"
    return q


# (n_peaks, n_matched)
CHANCE = (44, 12)         # a scrambled sparse frame strict accepts: the measured null at 40-48 peaks is 11-15 matched
SPARSE_REAL = (55, 24)    # the sparse real frames floor keeps sit at 17-29 matched
DENSE_REAL = (300, 90)
DENSE_LOW = (400, 60)     # clears the floor, not 25 %: floor includes strict


def part1_synthetic():
    check("GATES offers floor", GATES == ("none", "strict", "floor"), GATES)
    check("the cxidb17 floor resolves to its (a, b, c)", resolve_floor("cxidb17") == tuple(FLOOR), resolve_floor("cxidb17"))
    check("(a, b), (a, b, c), a list and a 1-D array are accepted",
          resolve_floor((0.05, 10)) == (0.05, 10.0) and resolve_floor([0.01, 5, 1.2]) == (0.01, 5.0, 1.2)
          and resolve_floor(np.array([0.01, 5.0, 1.2])) == (0.01, 5.0, 1.2))
    bad = [{0.02, 5.0}, frozenset((0.02, 5.0, 1.2)), (float("nan"), 1.0), (0.1, float("inf")), (True, 1.0),
           (0.1,), (0.1, 1.0, 2.0, 3.0), "bogus", "0.1,5", 3.0, None, {"a": 1, "b": 2}, (0.1, "x"),
           (v for v in (0.1, 5.0)), np.array([[0.1, 5.0]])]
    ok = [b for b in bad if not raises(lambda b=b: resolve_floor(b))]
    check("bad floors raise ValueError (set, NaN, inf, bool, length, unknown name, ...)", not ok, ok)
    check("floor_value is a*n + b + c*sqrt(n)",
          abs(floor_value(49, (0.1, 2.0, 0.5)) - (4.9 + 2.0 + 3.5)) < 1e-12 and abs(floor_value(49, (0.1, 2.0)) - 6.9) < 1e-12)

    # the fixtures are what they claim under the SHIPPED constants, checked before gating anything
    f = lambda nm: floor_value(nm[0], FLOOR)
    check("fixture: CHANCE passes strict and is under the floor", strict_ok(CHANCE[1], CHANCE[0]) and CHANCE[1] < f(CHANCE),
          f"floor({CHANCE[0]}) = {f(CHANCE):.1f}")
    check("fixture: SPARSE_REAL and DENSE_REAL pass strict and the floor",
          all(strict_ok(m, n) and m >= floor_value(n, FLOOR) for n, m in (SPARSE_REAL, DENSE_REAL)),
          f"floor({SPARSE_REAL[0]}) = {f(SPARSE_REAL):.1f}, floor({DENSE_REAL[0]}) = {f(DENSE_REAL):.1f}")
    check("fixture: DENSE_LOW clears the floor but not 25 %",
          DENSE_LOW[1] >= f(DENSE_LOW) and not strict_ok(DENSE_LOW[1], DENSE_LOW[0]), f"floor = {f(DENSE_LOW):.1f}")

    rng = np.random.default_rng(SEED)
    M = LYSO
    cases = [CHANCE, SPARSE_REAL, DENSE_REAL, DENSE_LOW]
    frames = [_frame(M, m, n - m, rng) for n, m in cases] + [_frame(M, 30, 10, rng)]
    fresh = lambda: [dict(image="x", event=i, M=(None if i == 4 else M.copy()), q=q[:5].copy(), hkl=np.zeros((5, 3)))
                     for i, q in enumerate(frames)]
    res = fresh()
    ns = gate_results(res, frames, "strict")
    check("strict keeps the CHANCE frame (the defect) and withdraws only DENSE_LOW",
          ns == 1 and [r["M"] is not None for r in res[:4]] == [True, True, True, False], ns)
    res = fresh()
    nf = gate_results(res, frames, "floor", floor="cxidb17")
    check("floor withdraws CHANCE and DENSE_LOW, keeps SPARSE_REAL and DENSE_REAL",
          nf == 2 and [r["M"] is not None for r in res[:4]] == [False, True, True, False], nf)
    check("a withdrawn frame is written as unindexed (M, hkl None; q = its peaks)",
          res[0]["M"] is None and res[0]["hkl"] is None and np.array_equal(res[0]["q"], frames[0]))
    check("an unindexed frame is left alone", res[4]["M"] is None and res[4]["q"].shape == (5, 3))
    res2 = fresh()
    gate_results(res2, frames, "floor", floor=list(FLOOR))
    check("the coefficients spelled out decide exactly as the name", [r["M"] is None for r in res2] == [r["M"] is None for r in res])
    # the boundary: a count equal to the floor is kept, one under it is not (CHANCE has exactly 12 matched)
    edge = []
    for fl, kept in (((0.0, 12.0), True), ((0.25, 1.0), True), ((0.0, 12.001), False), ((0.25, 1.001), False)):
        r = fresh()
        gate_results(r, frames, "floor", floor=fl)
        edge.append((r[0]["M"] is not None) == kept and r[1]["M"] is not None)
    check("matched == floor is kept, matched < floor is withdrawn (>=, not >)", all(edge), edge)
    check('"floor" without floor= raises', raises(lambda: gate_results(fresh(), frames, "floor")))
    check("floor= with gate strict raises (it would go unused)", raises(lambda: gate_results(fresh(), frames, "strict", floor="cxidb17")))
    check("floor= with gate none raises", raises(lambda: gate_results(fresh(), frames, "none", floor=FLOOR)))
    check("gate none still withdraws nothing", gate_results(fresh(), frames, "none") == 0)


# ------------------------------------------------------------------------------------------ part 2 ----
SPARSE = 60


def _real():
    return [np.asarray(q, float) for q in load(FRAMES) if len(q) >= 6]


def _scrambled(fr, i, k):
    """Fit copy k of frame i, the stream cli_gate_null.py fits on: rng default_rng([1, i, k])."""
    return scramble_azimuth(fr[i], np.random.default_rng([1, i, k]))


def _kept(res, frames, gate, **kw):
    r = copy.deepcopy(res)
    gate_results(r, frames, gate, **kw)
    return np.array([x["M"] is not None for x in r])


def part2_route():
    """The CLI's --cell route (hybrid_index, Mc_known = LYSO, nbest = 3) on the committed 120's sparse frames
    (< 60 peaks) and two scrambled copies of each; strict and floor then judge the same registrations."""
    fr = _real()
    sp = [i for i, q in enumerate(fr) if len(q) < SPARSE]
    real = [fr[i] for i in sp]
    scr = [_scrambled(fr, i, k) for i in sp for k in (0, 1)]
    frames = real + scr
    res, _ = hybrid_index(frames, Mc_known=LYSO, nbest=3)
    R = len(real)
    reg = np.array([r["M"] is not None for r in res])
    ks, kf = _kept(res, frames, "strict"), _kept(res, frames, "floor", floor="cxidb17")
    check("the route registers nearly every scrambled copy (measured 37/44)",
          reg[R:].mean() >= 0.6, f"{int(reg[R:].sum())}/{len(scr)}")
    check(f"strict writes many scrambled sparse copies: the defect (measured 16/{len(scr)})",
          ks[R:].sum() >= 8, f"{int(ks[R:].sum())}/{len(scr)}")
    check(f"floor writes almost none of them (measured 1/{len(scr)})", kf[R:].sum() <= 3, f"{int(kf[R:].sum())}/{len(scr)}")
    check("floor never writes a frame strict refuses", not (kf & ~ks).any())
    check("floor keeps the sparse real frames strict keeps (measured 10 of 11)",
          ks[:R].sum() >= 8 and kf[:R].sum() >= ks[:R].sum() - 2, f"{int(kf[:R].sum())} of {int(ks[:R].sum())} (of {R})")


# ------------------------------------------------------------------------------------------ part 3 ----
# Sparse real frames floor keeps (54-57 peaks, 24-29 matched on this route) and scrambled copies [1, i, k] that
# --gate strict WRITES (40-44 peaks, 12 matched: 2 above strict's minimum there and 2 under the floor).
REAL_CTRL = (0, 14, 54, 76, 90, 97)
SCR_CHANCE = ((11, 1), (33, 2), (65, 0), (75, 25), (99, 1), (102, 18))


def _crystal_events(path):
    out = []
    for ch in open(path).read().split("----- Begin chunk -----")[1:]:
        ev = next((l.split("//")[-1].strip() for l in ch.splitlines() if l.startswith("Event:")), None)
        if "--- Begin crystal" in ch:
            out.append(int(ev))
    return out


def part3_cli():
    fr = _real()
    frames = [fr[i] for i in REAL_CTRL] + [_scrambled(fr, i, k) for i, k in SCR_CHANCE]
    nr = len(REAL_CTRL)
    with tempfile.TemporaryDirectory() as td:
        qpath = os.path.join(td, "q.txt")
        with open(qpath, "w") as f:
            for i, q in enumerate(frames):
                f.write(f"FRAME {i} {len(q)}\n" + "".join(f"{a:.6f} {b:.6f} {c:.6f}\n" for a, b, c in q))
        env = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1")
        base = [sys.executable, "-m", "glint.glint_cli", "--qframes", qpath, "--cell", "79.02 79.02 37.98 90 90 90",
                "--device", "cpu", "--mode", "sparse"]

        def run(extra, name):
            out = os.path.join(td, f"{name}.stream")
            r = subprocess.run(base + extra + ["-o", out], capture_output=True, text=True, env=env)
            return r, out

        streams, ev = {}, {}
        coeffs = ",".join(repr(float(v)) for v in FLOOR)
        for name, extra in (("default", []), ("none", ["--gate", "none"]), ("strict", ["--gate", "strict"]),
                            ("floor", ["--gate", "floor", "--floor", "cxidb17"]),
                            ("coeffs", ["--gate", "floor", "--floor", coeffs])):
            r, out = run(extra, name)
            if r.returncode != 0:
                check(f"CLI {' '.join(extra) or '(no --gate)'} runs", False, r.stderr.strip().splitlines()[-3:])
                return
            streams[name] = open(out, "rb").read()
            ev[name] = _crystal_events(out)
            if name == "floor":
                check("the report names the floor gate and its coefficients",
                      "gate (floor)" in r.stdout and "sqrt(n)" in r.stdout, [l for l in r.stdout.splitlines() if "gate" in l])
        scr_ev = set(range(nr, len(frames)))
        check("the default stream is byte-identical to --gate none", streams["default"] == streams["none"])
        check("--gate none writes all 12 frames as crystals", sorted(ev["none"]) == list(range(len(frames))), ev["none"])
        check("--gate strict writes the scrambled chance frames (the defect; measured 6/6)",
              len(scr_ev & set(ev["strict"])) >= 5, sorted(scr_ev & set(ev["strict"])))
        check("--gate floor writes none of the scrambled frames", not (scr_ev & set(ev["floor"])), ev["floor"])
        check("--gate floor writes every real control", sorted(ev["floor"]) == list(range(nr)), ev["floor"])
        check("--floor a,b,c gives the same stream as --floor cxidb17", streams["coeffs"] == streams["floor"], coeffs)
        for extra, why in ((["--gate", "floor"], "--gate floor without --floor"),
                           (["--floor", "cxidb17"], "--floor without --gate floor"),
                           (["--gate", "strict", "--floor", "cxidb17"], "--floor with --gate strict"),
                           (["--gate", "floor", "--floor", "bogus"], "an unknown --floor name"),
                           (["--gate", "floor", "--floor", "1,2,3,4"], "four coefficients"),
                           (["--gate", "floor", "--floor", "nan,1"], "a NaN coefficient")):
            r, _ = run(extra, "bad")
            check(f"refused by argparse: {why}", r.returncode == 2 and "--floor" in r.stderr,
                  (r.returncode, r.stderr.strip().splitlines()[-1:]))


def main():
    part1_synthetic()
    part2_route()
    part3_cli()
    print(f"{'FAILURES: ' + ', '.join(fails) if fails else 'ALL PASS'}  (torch {torch.__version__})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
