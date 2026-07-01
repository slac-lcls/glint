# GLINT roadmap

High-level directions. Concrete, assignable tasks live as **GitHub Issues** (labelled by track);
this file is the map. How we work: [`CONTRIBUTING.md`](CONTRIBUTING.md) + [`docs/onboarding.md`](docs/onboarding.md).

The engine is stable (blind serial + dense rotation, CrystFEL drop-in). The open work splits into
a **real-data / productization** side and an **algorithms** side.

## 1. Validation & real data
Run GLINT blind on real SFX datasets and judge output against established indexers.
- End-to-end on a real dataset: cell + merge (CC\*/Rsplit) vs cctbx.xfel / DIALS / CrystFEL.
- Extend the `--fromfile` → CrystFEL-refine → `partialator` merge beyond ProK / lysozyme (≥2 more proteins).
- Characterize where GLINT wins / ties / needs work across cell types and sparsity.

## 2. Productization (LCLS)
- Wire the LUTE `GLINTIndexer` task (`lute/`) into a real SFX DAG — it fills the gap that LUTE's
  CrystFEL builds lack (no FFBIDX).
- Packaging / deployment polish; documented recipes.

## 3. Pipeline & optimizer review
Audit the full pipeline — M1–M6 plus cross-frame **consensus** and the known-cell **rescue** — and the
optimizers behind them (M3 gradient ascent, M5 anneal, the refiner history incl. Newton/CG, the M1 start
grid). Goal: an independent read, plus proposals. Background reading: the paper's *Architecture*,
*Accuracy-ceiling* (catalog of single-frame levers that failed and why), and *Throughput* sections.

## 4. Throughput
- GPU-batch the known-cell rescue's candidate search to close the remaining gap to ffbidx.
- Trim the M1 start grid; re-profile M1–M6 occupancy.

## 5. CBXD — convergent-beam (exploratory)
Convergent-Beam X-ray Diffraction (Chapman group, arXiv:2602.14402) gives a cone of incident directions,
so each reflection is a Kossel-circle **streak** rather than a point. Known-cell indexing is solved; blind
orientation+cell is walled by per-streak precision. Directions: reproduce the streak simulation, and probe
the streak-shape / joint-fit path that could break the blind wall. (This is the figure recently pulled
from the paper — kept here as a live direction.)

---
*Directions are open and discussed in the Issues. Ownership is assigned there, not here.*
