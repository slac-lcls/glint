# GLINT M3 basin characterization (synthetic) — basin_char.py

Per-vector `refine_vec` (GD steps=40) recovering a true lattice vector v* from a θ-perturbed seed; still-frame
Ewald clouds; return-to-v* = |T_out ∓ v*|/|v*| < 5%. 12 orientations × 120 trials/point.

## Capture radius (recovery vs θ, short vector)
| cell | 2° | 4° | 6° | 8° | 10° | 14° |
|---|---|---|---|---|---|---|
| lyso 79/79/38 | 100 | 100 | 66 | 53 | 53 | 33 |
| cubic 100 | 100 | 85 | 53 | 30 | 19 | 8 |
| ortho 50/65/80 | 100 | 82 | 63 | 54 | 45 | 25 |
→ basin ~4° at 100%; **narrower for larger cells** (cubic-100 tightest). Fixed point stable (θ=0 → 100%).

## Robustness (lyso short, θ=4°) — the SFX-relevant axes
- **Sparsity/partiality: 100% even dropping 80% of peaks.** Per-vector refine barely needs the full cloud.
- **Position noise:** 100% up to noise=0.01·qmax, 58% at 0.02.
- **Mosaic:** 100% up to 1.0° (very robust).
⇒ the per-vector M3 refine is NOT the SFX bottleneck — sparsity/mosaic/noise hardly dent it. The difficulty
is SELECTION/ASSEMBLY among many candidate vectors (the known ~71% spurious-limited wall), not convergence.

## Optimizers at the basin edge (θ=8°)
GD 54% · **CG 87%** · BB 65% — CG has the widest basin (conjugate directions climb the multimodal comb).

## Restart-ensemble scaling (θ=8°, keep-best-objective)
K=1 → 52%, 3 → 80%, 5 → 89%, 9 → 96%, 15 → 99%, 25 → 100%. Clean monotone; **the Fibonacci seed grid IS this
ensemble** — K≈9–15 restarts ≈ full recovery. Ensemble/voting >> single-instance feedback (RAAR/SO2D, tested
elsewhere, HURT far from basin). Confirms [[phase-retrieval-stagnation-lessons]].

# Extended sweeps — basin_char2.py

## θ@50% basin edge per cell × {short, long} vector (GD steps=40; fine grid + linear interp)
| cell | short v* | long v* |
|---|---|---|
| lyso 79/79/38 | 9.9° | 7.1° |
| cubic 60 | 7.3° | 7.3° |
| cubic 100 | 6.3° | 6.5° |
| ortho 50/65/80 | 8.9° | 6.4° |
→ **long axes have narrower basins** (lyso 9.9 vs 7.1; ortho 8.9 vs 6.4); isotropic cubic is equal both ways;
**larger cell = tighter basin** (cubic-100 ~6.3° is smallest). θ@50% ≈ 6–10° (vs θ@100% ~4° in basin_char.py).

## Short vs long vector recovery vs θ (lyso 38 Å short / 79 Å long)
| θ | 0 | 2 | 4 | 6 | 8 | 10 | 14 | 20 |
|---|---|---|---|---|---|---|---|---|
| short | 100 | 100 | 100 | 68 | 53 | 49 | 30 | 15 |
| long | 100 | 100 | 86 | 64 | 41 | 26 | 11 | 5 |
→ the **long 79 Å axis is the harder target** — collapses ~1 step earlier (δ(q·v) ∝ σ|v|, so a longer real-space
vector is more sensitive to the same angular seed error). Both stable at θ=0 (fixed point).

## Optimizer × restart-ensemble interaction (lyso short, θ=8°, keep-best-objective)
| K | 1 | 3 | 5 | 9 | 15 | 25 |
|---|---|---|---|---|---|---|
| GD | 52 | 80 | 90 | 96 | 99 | 100 |
| CG | 87 | 98 | 100 | 100 | 100 | 100 |
→ **CGs wider basin makes the ensemble saturate ~5× faster: CG reaches 100% by K=5, GD needs K=25.** CG at
K=1 (87%) already beats GD at K=3 (80%). Actionable: the restart-ensemble (= Fibonacci seed grid) should refine
with **CG**, not GD — same recovery at a fraction of the restarts (compute-budget lever, [[glint-profile-optim]]).
