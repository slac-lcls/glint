# Analyses behind the benchmark report and slide deck

Copies of the one-off probes run on Perlmutter, kept for reproducibility. Paths inside are the
author's (`/pscratch/sd/k/kmdalton/...`); the sbatch files are how each was run.

| script | what it computes |
|---|---|
| `gate.py`, `exc.py` | Ewald-gate choice: found-peak distance to the nearest node, and the share of integrated intensity kept at each excitation-error tolerance, on a pilot run integrated at tol 0.002 |
| `bias.py` | mean intensity and median sigma of the same reflections in the three arms, by shell and excitation error |
| `ccshells.py`, `ccshells2.py` | CC½ in resolution shells for the three arms, from partialator's half-datasets |
| `ccano_w.py` | inverse-variance weighted CCanom, w = 1/(σ²ΔI₁ + σ²ΔI₂) |
| `deck_stats.py` | per shell: CC½, Pearson/rank/model CCanom, R-work/R-free from refined.mtz |
| `subsample/subsample.py` | CC½ and CCanom against number of merged images, from random subsets (same draws for every arm) merged with partialator; `run33.sbatch`, `run45.sbatch` batch all merges of a dataset into one job |
| `subsample/diffs.py` | paired arm − CrystFEL differences per shell and size, with a t-statistic over the draws |

The CC and subsample scripts take the space group, point group and the reindex operator from the
environment (`SG`, `PG`, `REINDEX`; defaults `P 43 21 2`, `422`, `h,k,l`), so CXIDB 61
(`SG='P 21 21 2' PG=222 REINDEX=k,l,h`) runs through them unchanged.
