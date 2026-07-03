# powder_index — ab-initio powder-pattern autoindexer

Ring positions (1/d^2) -> unit cell + Miller indices, all 7 crystal systems. Relocated from the public
`slac-lcls/drp-benchmarks` repo to keep it private (closed PR #4).

Method: `1/d^2(hkl)` is linear in the reciprocal-metric components, so high-symmetry + monoclinic use
**seed-and-verify** (assign trial hkl to the first lines, exact least-squares metric solve, verify by de
Wolff **M20**); triclinic uses **simulated annealing** on the M20 objective. Portable numpy/cupy (CPU tool
— GPU measured slower, latency-bound on the small metric solves).

- `powder_index.py` — the primitive (`index_powder`, `peaks_from_profile`, ...).
- `test_powder_index.py` — planted-cell recovery, all 7 systems + centering + cupy parity. Run: `pytest test_powder_index.py`.
- `benchmarks/powder_vs_gsas2.py` — head-to-head vs GSAS-II (import-or-skip; needs `gsas2pkg`).
- `powder_schematic.svg` — how-it-works figure.

`../powder_ml/` (the learned powder autoencoder) vendors this file via a symlink.

Landscape: method-family cousin of N-TREOR/ITO (seed-and-verify) + McMaille (SA for triclinic); coverage at
parity with DICVOL/GSAS-II/conograph, but NOT real-data-hardened (no zero-point handling; pseudo-symmetric
equal-edge cells still fail). Niche = small embeddable DRP primitive, not a DICVOL/TOPAS replacement.
