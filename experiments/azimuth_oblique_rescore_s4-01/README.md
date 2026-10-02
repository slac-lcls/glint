# Triclinic known-cell rescue rate, re-scored against ground truth (review r2 s4-01)

`experiments/azimuth_oblique.py` (24 random triclinic cells x 60 orientations, dense and still), CPU, 1 Oct 2026.
The per-frame engine (`replica_gpu.index_known_gpu_cell`) is fp64 throughout, so CPU and A100 run the same arithmetic.

| log | engine | gate |
|---|---|---|
| `main_oldgate.log` | main ef6d068 | the old `azimuth_coverage.gate`: `same_lattice` + >= 25 % + >= 10 (reproduces the docs/onboarding.md table) |
| `main_truthgate.log` | main ef6d068 | the same plus "M is a basis of the frame's true lattice" (T = inv(M_true) M integral, unimodular) |
| `fix_truthgate.log` | fix/kc-mirror-setting d0f00c3 (0.2 Å gate; every sampled random triclinic cell tilts far past both that gate and the final 1° one, so all take the two-handed path) | truth gate |

On the production grid ("full"), main indexes the true lattice on 52-60 % of frames; the old gate counted another
~40 % that were mirrored, non-lattice bases with the reference metric. The fix gives 99.4-100 %.
