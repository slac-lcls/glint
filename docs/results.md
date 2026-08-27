# Settled questions

Directions that are **closed** — the lever shipped, or it was tried and did not pay. They lived in
`ROADMAP.md` until that document grew long enough that its opening priorities went stale unnoticed;
the roadmap now maps what is *open*, and the reasoning lands here.

Most of this is negative results. That is deliberate: the cheapest thing this repo can give the next
person is an accurate map of which plausible ideas have already been paid for.

Measured numbers are in `experiments/check_numbers.py` (`--facts`), which self-checks its own
arithmetic. Quote from there rather than from prose.

---

## M3 refiner — six optimizers agree that momentum-GD, STEPS=8, is the optimum

**CG** (`cg_test.py`): momentum-GD Pareto-dominates it at every step count and at equal wall time
(grad-4 = 76 @ 15.3 ms vs cg-4 = 67 @ 15.7; grad-8 = 84 @ 19.7).

**Exact fused-phase 1-D Newton line search** (`refine_vec_ls`, `ls_test.py` / `ls_cap_test.py`) — the
elegant one. Along a direction the cos objective is closed-form `S(a) = Σ wm cos(φ + aψ)`, so f/f'/f''
are matmul-free once the phases are computed. Built and swept over the cap and momentum knobs. Verdict
**negative**: ls-8 = 72; relaxing the anti-jump cap gives nothing (72); momentum recovers only part
(77) — all below GD's 84, at ~2× the cost, because the direction's `ψ = d·q` is a *second* matmul, so
it was never gradient-cost.

**Root cause, and it is the same across grad / cg / bb / lm / newton / ls:** greedy per-step optimality
lands more starts in spurious maxima on the multimodal comb. Momentum-GD's gentle, non-greedy,
schedule-annealed ascent *is* the mechanism, not an approximation to something better.

Newton specifically (`REFINER=newton`) basin-jumps from imperfect seeds — wedge5 100 → 50% — for a
10–20% speedup. Flag-gated, default off.

## Indexing as phase retrieval — RAAR helps, but only in the sparse under-determined regime

Casting M3 as phase retrieval (P_data = round inlier projections to integer hkl; P_support =
least-squares refit onto range(Q)) and applying **RAAR** feedback lifts the **sparse** blind rate
**84 → 88/120** (β = 0.7, ~16 iters; reproduced) — right at the oracle-reachable ceiling (73–76%). It
is the *first* single-frame lever to beat momentum-GD, it recovers exactly the selection-miss frames GD
loses to spurious basins, and it **grows on hard data** (mosaic blind 11 → 17 at σ = .0015, +55% rel).

**Two corroborations bound it hard:**

1. **Hybrid-neutral in every regime** — full 120, few-frame down to N=5, mosaic to σ = .0015: all
   114–118/120. Consensus + rescue already cover those frames. (An early N=8 "+8" was noise; K=40 ties.)
2. **It regresses on RICH stills** — dials60 blind GD 56/60 vs RAAR 45/60 — and **no β/step setting
   recovers it** (all ~75–78%). On well-determined frames the hard round-to-integer projection locks
   onto wrong self-consistent *alias* assignments that GD's soft cosine ascent avoids.

⇒ **Do not feature as a rate headline or table row.** What is worth featuring is the *concept* —
indexing as phase retrieval, singular-vectors-as-FFT — as framing, plus an honest remark in the
accuracy-ceiling discussion: phase-retrieval feedback confirms the single-frame ceiling is an
under-determination / spurious phenomenon, since it helps exactly in the sparse under-determined regime
and nowhere else. `REFINER=raar` stays opt-in for blind-only / sparse.

Rest of the family: **SO2D** (adaptive-β saddle) cold = negative — it needs a stable iterate and 70k
blind seeds are the opposite — but a **RAAR warm-up rescues it to 88** (ties RAAR at 2× cost),
confirming the saddle-point literature. **HIO** (β = 1) underperforms at a short budget (83; wilder,
needs a long run plus polish). **ADMM** +1 over GD but below RAAR. **ER-polish** neutral.
RAAR β ≈ 0.7 is the pick of the whole family.

## M1 start grid — the cut is NOT free against the gate that ships

An earlier version of the roadmap concluded that the blind start grid could be thinned 2–3× at "no
end-to-end accuracy cost", because the *hybrid* rate held at 115–118/120 down to n_dir 700.

**Retracted by PR #29.** Against the strict Table-1 gate that cost **eight points, 93 → 85**. Those are
different metrics rather than contradictory data — but it was the third time in this repo a conclusion
proved scoped to a looser number than the one that ships.

- **n_dir 1100** (−4 pts for 1.24×) is defensible. **n_dir 700 is not.**
- The speedup is also smaller than the stage breakdown implies: blind gains 1.83× but hybrid only
  **1.36×**, because a thinner grid *moves* work onto the rescue rather than removing it (`n_resc`
  24 → 62).

The underlying accuracy measurement stands: blind degrades monotonically as the grid thins (n_dir
2200/1600/1100/700 → 84/80/76/65 of 120). The 70,400 starts are **load-bearing for generation**,
consistent with the generation-limited diagnosis, not redundant.

## Known-cell rescue — the gap to ffbidx is closed, and reversed

Listed as open work ("GPU-batch the rescue's candidate search to close the last ~4× gap to ffbidx")
until issue #5 closed via PRs #14 / #15 / #16:

| stage | ms/hit |
|---|---|
| per-frame numpy → GPU (`replica_gpu`) | 16.5 |
| \+ CUDA graph, bit-identical (#14) | 1.46 |
| \+ fused CUDA kernels, B=32 (#16) | 0.33 |
| \+ saturated batch B=120 | 0.17 |

Against pipelined ffbidx (3.1 ms) that is ~**18× faster**, not 4× slower. `KC_FP=32` (#15) gives
0.14 ms at B=120, rate-neutral.

The last two rows moved on 2026-08-27 (#165): `obj_fused` runs at K=4096/5760 against a 128-thread
block, so one block per frame made each thread walk 32-45 candidates serially. Splitting the
candidate axis over `blockIdx.y` is **bit-exact** — `inl`/`sub` identical, `same_lattice` 120/120,
rate (80,115) unchanged — and 1.54× faster at B=120 fp64. fp32 gains far less (1.14×), so the
speedup is an fp64 result and should not be read as a deployment-precision claim.

## M5 anneal — the CUDA-graph lever was overtaken

The roadmap carried "CUDA-graph the M5 anneal (~29% of the sparse frame; launch-bound at a 6.1 ms floor,
15 sequential 3×3 solves)". Both numbers are now dead:

- `ANNEAL_ITERS` default went **15 → 3** (commit 9310fa3): blind holds (85 vs 84 same_lattice, 78
  gated) and consensus holds (117/120), while the anneal stage drops 7.8 → 2.5 ms (3.1×). The 15-iter
  default was over-provisioned.
- PR #29's profile then measured the anneal at **1.49 ms = 11.8%** of the blind frame, not 29%. M3
  refine was the bottleneck at 65.8% — which is what motivated the fused-M3 work (PRs #30 / #39).

What survives is the note in `glint/fused_m3.py`: the anneal has its own ~2 ms launch floor, so the
rescue's batch → graph → fuse playbook does not transfer to it unchanged.

## Also closed

- **fp32 M5 anneal** — negative, but only because M5 was launch-bound at the time; the two were
  coupled. Committed as a default-off knob (`ANNEAL_FP32`, 05d70ba).
- **Naive CUDA streams for concurrent frames** — zero gain; data-dependent syncs serialize the issue.
  Multi-threading gave ~10–15%, within node noise.
- **Reverse / Chamfer selection cost** — negative on thin-slice data, but the test was arguably
  premature: it needs the 3-D information a fat slice provides. Re-test belongs with Laue/pink-beam
  (roadmap §5, issue #8), not here.
