# M2 objective tuning log

The blind candidate-generation objective (M2/M3) is
`f(v) = Σ_i w_i · c(q_i·v)` over observed peaks `q_i`, with a per-peak **weight** `w_i`,
a **proximity function** `c` (high when `q_i·v` is near integer), and an inlier **mask**.
All sweeps below: 120 sparse cxidb lysozyme frames, A100, `STEPS=8`, metric = blind
**correct-cell** rate (`same_lattice` with LYSO), M4–M6 assembly unchanged so the effect is
isolated to generation. Scripts: `sweep_qpow.py`, `sweep_prox.py`, `sweep_qapod.py`.

## 1. Resolution weight `w_i = |q_i|^(-p)` — `QPOW` (default 1.0)

| p | 0.0 | 0.5 | **1.0** | 1.25 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|
| rate | 65% | 67% | **70%** | 60% | 55% | 38% | 27% |

Sharp optimum at **p = 1.0** (GLINT's default; fine sweep `[0.7,1.2]` confirms). The xgandalf
paper's `1/|q|²` (p=2) collapses to 38% *in GLINT's pipeline* — the weight is **co-tuned with the
rest** (cos² coverage term + triplet assembly), so it doesn't port across pipelines. Real xgandalf
(its own assembly/normalization/50k starts) ties at 71%; the 38% is "p=2 inside GLINT", not xgandalf.

## 2. Proximity form / smooth window — `OBJFORM` (`""`=default), `OBJSIG`, `OBJKAP`

Replace the hard inlier mask `1[|q·v−round|<tol]` with smooth forms (all gradients
finite-difference validated ~1e-9):

| form | rate | |
|---|---|---|
| default `cos`/`cos²` + hard mask | 70% | baseline |
| wrapped Gaussian σ=.12 | **71%** | best (+2 frames) |
| cos × Gaussian window σ=.18 | 71% | ties |
| tent (xgandalf linear, 1..−1) | 70% | ties cos |
| Gaussian σ=.08 | 64% | too narrow |
| von Mises κ=6 / 12 / 25 | 64 / 54 / 53% | **sharper = worse** |

Smooth window is a *marginal* free edge (+2). **Width matters more than form**; the smooth optimum
σ≈.12 lines up with hard `tol`≈.18. Sharper combs (von Mises) manufacture spurious local maxima —
empirically confirming the xgandalf paper's smoothness argument.

## 3. erf² q-resolution apodization — `QHI`/`QLO`/`QAPSIG` (default off) — NEGATIVE

Smooth high-q / low-q(beamstop) taper on the weight, `(erf((q−q0)/σ))²` (square ⇒ C¹ at the edge):

| hi-q edge (·qmax) | none | .95 | .90 | .85 | .80 | .70 | lo-q .15 |
|---|---|---|---|---|---|---|---|
| rate | **70%** | 67 | 61 | 60 | 59 | 53 | 67 |

**Hurts** monotonically. Reason: M2 is a **gridless direct sum, not an FFT** → there is no
truncation/edge ringing to apodize, so high-q peaks are *signal* (high-res lattice constraints), not
artifact; tapering them off just deletes constraints. Apodization is the right tool for
transform-domain ringing (FFT/DPS, real-space windows); kept default-off for real data with genuine
high-q junk or a beamstop halo.

## 4. Inlier-window half-width `TOL` — hard mask vs apodized edge

The per-peak inlier indicator `1[|q·v−round| < TOL]` (xgandalf's ε). The xgandalf paper leaves ε
empirical ("the smaller ε, the more resistant to spurious peaks") and notes the hard indicator makes
the score *discontinuous* — the very thing the smooth `OBJFORM` window (§2) fixes. Sweep on 120 cxidb,
hard `TOL` vs smooth gauss `OBJSIG`, clean and mild mosaic:

| width | 0.10 | 0.15 | **0.18** | 0.25 | 0.35 |
|---|---|---|---|---|---|
| clean, HARD TOL | 59 | 63 | **70** | 63 | 62 |
| clean, GAUSS σ | 67 | 70 | 66 | 59 | — |
| mosaic, HARD TOL | 22 | 35 | **37** | 35 | 33 |
| mosaic, GAUSS σ | 36 | **39** | 37 | 33 | — |

`TOL = 0.18` is the **confirmed hard-mask optimum** (sharp-ish; 0.10/0.35 ~10 pts worse). The smooth
window ties it (clean) or edges it (mosaic +2). **Widening past the optimum hurts both hard and
smooth** — a wide window admits spurious peaks regardless of edge shape, so *apodization does not buy a
wide window*. The lever for catching long-axis inliers under broadening is `QDIST` (isotropic
reciprocal-distance), not widening the hkl-space window.

## Takeaway

The front-end is **well-tuned and robust** — no single-knob change cracks the ~70% single-frame
wall (spurious-limited). Multi-frame **consensus** remains the only rate lever (→ 96% hybrid). All
knobs are exposed (env vars, defaults = current behavior) for **per-dataset** tuning.

## Future / not yet tried

- **Low-order polynomial weight** `w_i = (1 + a·|q| + b·|q|²)` (or rational forms) — a more flexible
  resolution weight than a single power `|q|^(-p)`; fit `(a,b)`. **Needs more example datasets**
  (varied resolution / cell / peak density) before fine-tuning, or it just overfits cxidb.
- **Cross-dataset validation**: rerun the `QPOW` / `OBJFORM` / `QHI` sweeps on the rich DIALS-60 set
  (denser, more high-res) and on fat-Ewald sims — does the optimum move? Higher-res/denser data may
  reward smoothing or a different `p`.
- **Auto-tune** the knobs per frame from peak density / resolution, once the cross-dataset trend is
  known.
