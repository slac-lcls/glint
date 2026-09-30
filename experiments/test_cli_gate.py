"""glint_cli --gate: what a frame must satisfy to be WRITTEN as a crystal.

WHY THIS EXISTS. The CLI wrote every registration the indexer returned. With a known cell (--cell, the way
LUTE's IndexGLINT runs) that is nearly every frame: the known-cell rescue's only test is
same_lattice(M, Mc), which a known-cell search passes by construction because it returns the cell it was
asked for. On LUTE's SFX test runs this wrote 28 % of mfx100848724 r51 as crystals where CrystFEL and cctbx
index about 1 %, and 98 % of mfxl1038923 r58. --gate strict applies the paper's scoring bar (>= GATE_MIN
matched and >= GATE_FRAC of the frame's peaks) to what gets written; --gate none (the default) keeps the
historical output.

WHAT IT CHECKS.
  1. gate_results on synthetic frames: "none" changes nothing; "strict" keeps an on-lattice frame, withdraws
     a scrambled one and a sparse one (fewer than GATE_MIN peaks), leaves an unindexed frame alone, and
     writes a withdrawn frame exactly as an unindexed one (M, hkl None, q its peaks). Bad arguments raise.
  2. On 30 real cxidb-17 frames and their 30 azimuth-scrambled copies (every peak rotated about the beam, so
     |q| and q_z are kept and the lattice is gone), the known-cell engine registers every frame and
     same_lattice passes every registration, the scrambled ones included; the strict gate keeps most real
     frames and almost no scrambled ones.
  3. The CLI end to end (--qframes, --cell, CPU): --gate none writes all 12 of 6 real + 6 scrambled frames as
     crystals; --gate strict writes none of the scrambled ones; an unknown gate is refused by argparse.

SKIPS, exit 0, without torch (glint.glint_fast imports it; the CPU CI job installs none) or below
pyproject's torch floor (>= 1.12), like test_known_depth_args.py.

  PYTHONPATH=. python experiments/test_cli_gate.py
"""
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
from glint.glint_fast import GATE_MIN, LYSO, load                          # noqa: E402
from glint.hybrid_stream import GATES, gate_results                        # noqa: E402
from glint.multilattice import scramble_azimuth                            # noqa: E402
from glint.multishot import same_lattice                                   # noqa: E402
from glint.replica_gpu import index_known_gpu_cell                         # noqa: E402

FRAMES = os.path.join(HERE, "frames_cxidb_clean.txt")
fails = []


def check(name, cond, msg=""):
    print(f"  {name:72s}: {'PASS' if cond else 'FAIL'}   {msg}")
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


def on_lattice(M, rng, n):
    hkl = rng.integers(-6, 7, size=(4 * n, 3)).astype(float)
    hkl = hkl[np.abs(hkl).sum(1) > 0][:n]
    return hkl @ np.linalg.inv(M)


def part1_synthetic():
    rng = np.random.default_rng(20260928)
    M = LYSO
    good = on_lattice(M, rng, 40)
    scr = scramble_azimuth(good, rng)
    sparse = on_lattice(M, rng, GATE_MIN - 1)
    miss = on_lattice(M, rng, 30)
    frames = [good, scr, sparse, miss]
    fresh = lambda: [dict(image="x", event=i, M=(None if i == 3 else M.copy()), q=f[:5].copy(), hkl=np.zeros((5, 3)))
                     for i, f in enumerate(frames)]
    res = fresh(); before = [dict(r) for r in res]
    check("gate 'none' withdraws nothing", gate_results(res, frames, "none") == 0)
    check("gate 'none' leaves every result as it was", all(r["M"] is b["M"] and r["q"] is b["q"] for r, b in zip(res, before)))
    res = fresh()
    n = gate_results(res, frames, "strict")
    check("gate 'strict' withdraws the scrambled and the sparse frame", n == 2, n)
    check("an on-lattice frame keeps its registration", res[0]["M"] is not None)
    check("a withdrawn frame is written as unindexed (M, hkl None; q = its peaks)",
          res[1]["M"] is None and res[1]["hkl"] is None and np.array_equal(res[1]["q"], scr))
    check("a frame with fewer than GATE_MIN peaks is withdrawn even when they all match", res[2]["M"] is None)
    check("an unindexed frame is left alone", res[3]["M"] is None and res[3]["q"].shape == (5, 3))
    check("GATES lists the choices the CLI offers", GATES == ("none", "strict"), GATES)
    check("an unknown gate raises ValueError", raises(lambda: gate_results(fresh(), frames, "floor")))
    check("misaligned results and frames raise ValueError", raises(lambda: gate_results(fresh(), frames[:3], "strict")))


def part2_real_vs_scrambled():
    fr = [np.asarray(q, float) for q in load(FRAMES) if len(q) >= 6][:30]
    rng = np.random.default_rng(20260928)
    scr = [scramble_azimuth(q, rng) for q in fr]
    frames = fr + scr
    res = []
    for i, q in enumerate(frames):
        M = index_known_gpu_cell(q, LYSO)
        res.append(dict(image="x", event=i, M=M, q=q, hkl=None))
    reg = [r["M"] is not None for r in res]
    check("the known-cell engine registers every real AND every scrambled frame", all(reg), f"{sum(reg)}/60")
    taut = sum(same_lattice(r["M"], LYSO) for r in res[30:] if r["M"] is not None)
    check("same_lattice passes every scrambled registration (it cannot fail here)", taut == 30, f"{taut}/30")
    gate_results(res, frames, "strict")
    kr = sum(r["M"] is not None for r in res[:30]); ks = sum(r["M"] is not None for r in res[30:])
    check("strict keeps most real frames (measured 20/30)", kr >= 15, f"{kr}/30")
    check("strict keeps almost no scrambled frames (measured 0/30)", ks <= 3, f"{ks}/30")


def _crystal_events(path):
    out = []
    for ch in open(path).read().split("----- Begin chunk -----")[1:]:
        ev = next((l.split("//")[-1].strip() for l in ch.splitlines() if l.startswith("Event:")), None)
        if "--- Begin crystal" in ch:
            out.append(int(ev))
    return out


def part3_cli():
    fr = [np.asarray(q, float) for q in load(FRAMES) if len(q) >= 6][:6]
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as td:
        qpath = os.path.join(td, "q.txt")
        with open(qpath, "w") as f:
            for i, q in enumerate(fr + [scramble_azimuth(q, rng) for q in fr]):
                f.write(f"FRAME {i} {len(q)}\n" + "".join(f"{a:.6f} {b:.6f} {c:.6f}\n" for a, b, c in q))
        env = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1")
        base = [sys.executable, "-m", "glint.glint_cli", "--qframes", qpath, "--cell", "79.1 79.1 37.9 90 90 90",
                "--device", "cpu", "--mode", "sparse"]
        ev = {}
        for g in ("none", "strict"):
            out = os.path.join(td, f"{g}.stream")
            r = subprocess.run(base + ["--gate", g, "-o", out], capture_output=True, text=True, env=env)
            if r.returncode != 0:
                check(f"CLI --gate {g} runs", False, r.stderr.strip().splitlines()[-3:])
                return
            ev[g] = _crystal_events(out)
            if g == "strict":
                check("the report says how many registrations the gate withdrew", "gate (strict)" in r.stdout,
                      [l for l in r.stdout.splitlines() if "gate" in l])
        check("--gate none writes all 12 frames (6 real + 6 scrambled) as crystals", sorted(ev["none"]) == list(range(12)), ev["none"])
        check("--gate strict writes no scrambled frame (events 6-11)", all(e < 6 for e in ev["strict"]), ev["strict"])
        check("--gate strict keeps some real frames", len(ev["strict"]) >= 3, ev["strict"])
        r = subprocess.run(base + ["--gate", "bogus", "-o", os.path.join(td, "x.stream")], capture_output=True, text=True, env=env)
        check("an unknown --gate is refused by argparse", r.returncode == 2 and "invalid choice" in r.stderr, r.returncode)


def main():
    part1_synthetic()
    part2_real_vs_scrambled()
    part3_cli()
    print(f"{'FAILURES: ' + ', '.join(fails) if fails else 'ALL PASS'}  (torch {torch.__version__})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
