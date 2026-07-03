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
