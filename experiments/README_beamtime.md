# Beamtime data: why the experiment IDs are not in these scripts

Several scripts here read real LCLS beamtime data. The experiment IDs and the data paths are read
from the environment rather than committed, so the repository does not itself name which proprietary
runs we hold.

This is about the *identifiers*, not the data — the data was never in the repo. The point is that a
proposal ID in a committed file is a durable, greppable, externally-visible statement about what we
have access to, and beamtime data carries a proprietary period. Keeping it in the environment means
the scripts stay runnable by whoever has the access, without the repo asserting anything.

## What to set

| variable | used by | what it is |
|---|---|---|
| `GLINT_EXP` | `extract_pixels.py`, `fftindex_on_real.py`, `extract_strong_weak.py`, `psana_probe.py`, `peaknet/extract_geom_1038.py` | psana experiment id |
| `GLINT_RUN` | the same scripts | run number (each script defaults to the one it was written for) |
| `GLINT_PEAKNET_CXI` | `peaknet/run_peaknet_cxi.py` | directory of PeakNet CXI files |
| `GLINT_CXIDB17` | `build_cxidb_sw.py` | directory holding the cxidb-17 CrystFEL files |

Each script exits with a message naming its variable if it is unset, rather than failing somewhere
deeper with a confusing psana error.

```bash
GLINT_EXP=<exp> GLINT_RUN=51 python experiments/extract_strong_weak.py
```

## Two of these are not the same kind of thing

**`GLINT_CXIDB17`** points at **cxidb-17**, which is *public* deposited data (Coherent X-ray Imaging
Data Bank). Only the scratch path was ours, and only the path named a proposal. Nothing about this
dataset is restricted — it is parameterised here for tidiness and portability, not confidentiality.

**`GLINT_PEAKNET_CXI`** points into a **colleague's project area**, not ours. Whether that data can be
referenced or shared is that person's call, not a policy question — ask them.

## Before citing a run externally

Naming a run in a paper, a talk, or to an outside collaborator is a different act from having the ID in
a script. Confirm with the PI, and with LCLS if the proprietary period is in question. On-disk access
controls are a hint (the raw `xtc` directories are group-restricted, not world-readable) but they are
not the policy — the experiment record is.
