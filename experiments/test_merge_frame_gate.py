"""The live merge refuses frames whose mean intensity is not measured, instead of scaling them by an unbounded 1/mean(I).

WHY THIS EXISTS (review r2, findings s1-01 and s1-04). MergeAccumulator.add_frame scaled every frame to mean 1
(v = I/mean(I)) at weight 1/sigma^2. A frame with mean(I) near 0 -- a lattice-free or wrong-orientation accept of
the live gate, or a very weak real crystal -- got a huge scale and swamped the merge; a frame with mean(I) <= 0 or
<= 5 measurements kept scale 1, i.e. raw detector units among frames normalised to 1. On 793 real cxidb-17
lysozyme crystals (CrystFEL-integrated GLINT solutions) the live CC1/2 was 0.015; with the gate it is 0.33.
glint.merge_scale.frame_scale is now the one rule: merge only if n > 5 and mean(I) > 3 sem(I); refused frames
are counted in stats()["refused_frames"]. Frames that pass are merged exactly as before (same v and w).

  PYTHONPATH=. python experiments/test_merge_frame_gate.py      # exit 0 = all pass  (numpy only, ~5 s)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glint.stream_driver import MergeAccumulator, laue_ops_4mmm  # noqa: E402

FAILS = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + str(detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


rng = np.random.default_rng(7)
NU = 1500
# unique reflections in the 4/mmm asymmetric unit (h >= k >= 0, l >= 0), true intensities ~ Exp(500) ADU
U = []
while len(U) < NU:
    h, k, l = (int(x) for x in rng.integers(0, 25, 3))
    if h >= k and (h, k, l) != (0, 0, 0) and (h, k, l) not in U:
        U.append((h, k, l))
U = np.array(U); T = rng.exponential(500.0, NU); BG = 200.0


def real_frame(G, n=200):
    j = rng.choice(NU, n, replace=False)
    mu = G * T[j]; sig = np.sqrt(mu + 2 * BG)
    return U[j], mu + rng.normal(0, 1, n) * sig, sig


def noise_frame(n=200):
    """What integrating a wrong lattice at predicted positions gives: I ~ N(0, sigma)."""
    j = rng.choice(NU, n, replace=False); sig = np.sqrt(2 * BG) * np.ones(n)
    return U[j], rng.normal(0, 1, n) * sig, sig


def merge(frames):
    acc = MergeAccumulator(ops=laue_ops_4mmm())
    for i, (hkl, I, s) in enumerate(frames):
        acc.add_frame(hkl, I, s, i)
    return acc


good = [real_frame(rng.lognormal(0, 0.5)) for _ in range(400)]
clean = merge(good)
cc0 = clean.stats()["cc_half"]
print(f"clean: 400 real frames, CC1/2 {cc0:.3f}")

print("\n1. a frame with mean(I) <= 0 is not merged (it used to enter at scale 1, in raw ADU)")
while True:
    nf = noise_frame()
    if nf[1].mean() <= 0:
        break
acc = merge(good + [nf])
a, b = clean.merged_by_key(), acc.merged_by_key()
dev = max(abs(b[k] / a[k] - 1) for k in a)
check("merged intensities unchanged by the frame", dev == 0.0, f"max rel change {dev:.3g}")
check("stats() reports it as refused", acc.stats().get("refused_frames") == 1, acc.stats().get("refused_frames"))

print("\n2. a frame with <= 5 measurements is not merged")
hkl, I, s = real_frame(1.0, n=5)
acc = merge(good + [(hkl, I, s)])
b = acc.merged_by_key()
check("merged intensities unchanged by the frame",
      max(abs(b[k] / a[k] - 1) for k in a) == 0.0)

print("\n3. 5% lattice-free frames (mean(I) > 0 but not significant) do not collapse the live CC1/2")
noisy = []
for f in good:
    noisy.append(f)
    if rng.random() < 0.05:
        nf = noise_frame()
        while not (0 < nf[1].mean() < 3 * nf[1].std() / np.sqrt(len(nf[1]))):
            nf = noise_frame()
        noisy.append(nf)
acc = merge(noisy); st = acc.stats()
check("CC1/2 within 0.02 of the clean merge", st["cc_half"] > cc0 - 0.02,
      f"{st['cc_half']:.3f} vs clean {cc0:.3f}, {len(noisy) - len(good)} noise frames")
check("every noise frame is refused", st.get("refused_frames") == len(noisy) - len(good),
      f"{st.get('refused_frames')} refused")

print("\n4. frames that pass are merged exactly as before: v = I/mean(I), w = 1/sigma^2")
ref = MergeAccumulator(ops=laue_ops_4mmm())
for i, (hkl, I, s) in enumerate(good):
    ref.add_frame(hkl, I, s, i, values=I * (1.0 / I.mean()), weights=1.0 / np.maximum(s, 1e-3) ** 2)
same = all(np.array_equal(getattr(ref, x)[:ref.n_rows], getattr(clean, x)[:clean.n_rows])
           for x in ("sw", "swv", "cnt"))
check("running sums bit-identical to the explicit v, w", same and clean.stats().get("refused_frames") == 0)

print("\n5. the offline merge (experiments/merge_stats.py) uses the same rule")
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "g.stream")
    frames = good[:60] + [nf]                       # nf: the last noise frame, refused by the gate
    with open(path, "w") as fh:
        for hkl, I, s in frames:
            fh.write("----- Begin chunk -----\n--- Begin crystal\nReflections measured after indexing\n"
                     "   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel\n")
            for (h, k, l), ii, ss in zip(hkl, I, s):
                fh.write(f"{h:4d} {k:4d} {l:4d} {ii:10.2f} {ss:10.2f} 0 0 0 0 p0\n")
            fh.write("End of reflections\n--- End crystal\n----- End chunk -----\n")
    out = subprocess.run([sys.executable, os.path.join(HERE, "merge_stats.py"), path],
                         capture_output=True, text=True)
    line = [ln for ln in out.stdout.splitlines() if "not merged" in ln]
    check("merge_stats.py drops the unmeasured frame", out.returncode == 0 and line and line[0].startswith("1/61"),
          line[0] if line else (out.stdout + out.stderr)[-300:])

print(f"\n{'FAILURES: ' + ', '.join(FAILS) if FAILS else 'ALL PASS'}")
sys.exit(1 if FAILS else 0)
