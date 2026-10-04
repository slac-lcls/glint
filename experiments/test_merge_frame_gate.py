"""The live merge refuses frames whose mean intensity is not measured, instead of scaling them by an unbounded 1/mean(I).

WHY THIS EXISTS (review r2, findings s1-01 and s1-04). MergeAccumulator.add_frame scaled every frame to mean 1
(v = I/mean(I)) at weight 1/sigma^2. A frame with mean(I) near 0 -- a lattice-free or wrong-orientation accept of
the live gate, or a very weak real crystal -- got a huge scale and swamped the merge; a frame with mean(I) <= 0 or
<= 5 measurements kept scale 1, i.e. raw detector units among frames normalised to 1. On 793 real cxidb-17
lysozyme crystals (CrystFEL-integrated GLINT solutions) the live CC1/2 was 0.015; with the gate it is 0.33.
glint.merge_scale.frame_scale is now the one rule: merge only if n > 5 and mean(I) > 3 sem(I); refused frames
are counted in stats()["refused_frames"]. Frames that pass are merged exactly as before (same v and w).
Finding s1-05, same lines: the default I/sigma bins started at 0 with a strict >, so every I <= 0 measurement was
dropped -- a selection on the sign of I that biases weak reflections up and flatters Rsplit. The bins now start
at -inf. On the same merged frames stats(thr=0.0) gives the old I > 0 CC1/2, CC*, Rsplit (to a few ulp), unique,
redundancy and completeness; stats()["measurements"] is not thresholded (every merged row, I <= 0 included),
and the old I > 0 count is unique x redundancy at thr=0.0. Check 6 pins both.

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

print("\n2a. empty default-merge frames are refused, explicit partiality frames are unchanged")
acc = MergeAccumulator(ops=laue_ops_4mmm())
acc.add_frame(np.ones((6, 3), int), np.full(6, np.nan), np.ones(6), 0)
acc.add_frame(np.empty((0, 3), int), [], [], 1)
check("empty and all-nonfinite default frames are refused",
      acc.stats()["frames"] == 2 and acc.stats()["refused_frames"] == 2)
acc = MergeAccumulator(ops=laue_ops_4mmm())
acc.add_frame(np.empty((0, 3), int), [], [], 0, values=[], weights=[])
check("empty explicit-values frame keeps existing accounting",
      acc.stats()["frames"] == 0 and acc.stats()["refused_frames"] == 0)

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
    hkl0, I0, s0 = frames[0]
    frames[0] = (np.vstack((hkl0, hkl0[:2])), np.r_[I0, np.nan, I0[1]],
                 np.r_[s0, s0[0], np.nan])
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

print("\n6. no measurement is dropped on the sign of I (review r2 s1-05): weak data, <T>/sigma = 0.5")
acc_w = MergeAccumulator(ops=laue_ops_4mmm()); n_all = 0; n_pos = 0; Tw = rng.exponential(5.0, NU)
for i in range(400):
    j = rng.choice(NU, 250, replace=False); sig = np.full(250, 10.0)
    I = Tw[j] + rng.normal(0, 1, 250) * sig
    I[np.abs(I) < 0.05] = 0.0                       # some rows exactly on the I/sigma = 0 edge: I > 0 is strict
    acc_w.add_frame(U[j], I, sig, i, values=I, weights=1 / sig ** 2)   # explicit v, w: scaling out of the picture
    n_all += 250                                    # every row here is finite
    n_pos += int((I > 0).sum())
st = acc_w.stats()
check("default stats() counts every measurement",
      st["measurements"] == n_all and round(st["unique"] * st["redundancy"]) == n_all,
      f"measurements {st['measurements']}, unique x redundancy {st['unique'] * st['redundancy']:.0f}, rows {n_all}")
mk = acc_w.merged_by_key(); keys = np.array(list(mk)); est = np.array([mk[k] for k in keys])
from glint.stream_driver import _asu_key  # noqa: E402
truth = dict(zip(_asu_key(U, laue_ops_4mmm()).tolist(), Tw))
bias = float(np.mean(est - np.array([truth[k] for k in keys.tolist()])))
check("merged intensity unbiased (|bias| < 0.5 on <T> = 5)", abs(bias) < 0.5, f"bias {bias:+.2f}")
s0 = acc_w.stats(thr=0.0)
# unique x redundancy is the number of rows in the thresholded sums, so this pins the selection to I > 0
# exactly: an I/sigma > 1 floor or side="right" in stats() gives fewer rows, side="right" in add_frame
# puts the I = 0 rows above the 0 floor
check("stats(thr=0.0) is the I > 0 selection", round(s0["unique"] * s0["redundancy"]) == n_pos,
      f"unique x redundancy {s0['unique'] * s0['redundancy']:.0f}, I > 0 rows fed in {n_pos}; "
      f"redundancy {s0['redundancy']:.1f} (I>0) vs {st['redundancy']:.1f} (all)")
# "measurements" is n_meas, counted at add time and not thresholded: every finite row merged, I <= 0
# included, whatever thr is. It is NOT the old I > 0 count; that is unique x redundancy above.
check("stats(thr=0.0)['measurements'] is every finite row, not thresholded", s0["measurements"] == n_all,
      f"{s0['measurements']}/{n_all}")

print(f"\n{'FAILURES: ' + ', '.join(FAILS) if FAILS else 'ALL PASS'}")
sys.exit(1 if FAILS else 0)
