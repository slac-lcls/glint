# Measurement inputs committed here, and why

Three inputs behind published numbers turned out, on 2026-08-05, to exist on exactly one machine
each and not in git. That is invisible from the machine you work on — everything is simply there —
and shows up the moment anything runs elsewhere. All three were found the same way, by running
somewhere else:

| input | where it had been | how it surfaced |
|---|---|---|
| `nbest_120.npz` | untracked, one worktree | `consensus_barrier_ab.py` died with `FileNotFoundError` on S3DF |
| `prok_q.npz` | NERSC scratch only | the ProK re-run could not be moved off Perlmutter without a copy |
| DAQ reducer sources | untracked in `drp-benchmarks` | `DropletReducer.*` had already been **deleted** locally, leaving dangling symlinks |

The first two are committed here. The third belongs to `drp-benchmarks` and is handled there.

---

## `nbest_120.npz` — 33 kB, sha256 `d84701c9b7e84257`

The recorded N-best pool for the 120-frame sparse cxidb benchmark: **360 hypotheses**, 120 frames ×
top-3, frame-major (`fid` gives the frame index of each row).

| key | contents |
|---|---|
| `cells` | `(360,3,3)` the hypothesis matrices |
| `fid` | `(360,)` frame index per hypothesis |
| `score` | per-hypothesis score |
| `lyso` | the reference lysozyme cell, for the match gate |
| `nframes` | 120 |
| `index_s` | wall time of the indexing run that produced it |

**Used by** `consensus_barrier_ab.py`, and by any check of the consensus gate's acceptance
arithmetic — with this file the `min_frac`/`min_lead` decision at any K is reproducible on a laptop,
no GPU and no re-indexing. That is how the gate was shown to refuse at K=3 and K=4 and first accept
at K=5, which corrected the "saturates by ~3 frames" slide claim.

## `prok_q.npz` — 2.1 MB, sha256 `e4ac4f5486ba4400`

The 907 pre-extracted Proteinase K reciprocal-lattice-point frames (cxidb-45, SACLA MPCCD) behind
the head-to-head against DIALS at `glint.tex:836-848`. `q_<i>` for `i` in `0..n-1`, plus `n`.

**These are the same rlps DIALS was given**, which is the whole point — it makes the comparison a
test of indexing rather than of peak-finding.

**Each frame is exactly 170 peaks.** `dump_prok_q.py` (NERSC) uses
`peakfind(img, nmax=170, snr=8.0)` with `argsort(...)[::-1][:nmax]`, so it is a genuine top-170
brightness cap, not padding — there are no zero rows. The cap applies to both engines equally, but
the ProK numbers describe indexing on the 170 brightest peaks per frame, not on everything the
detector recorded. The paper does not currently say so.

Regenerating it needs NERSC: `/global/cfs/cdirs/lcls/dermen/cxidb45/data/run296940-{0,1,2}.h5` and
`/global/cfs/cdirs/lcls/mnasser/cxidb_62/mpccd-optimized.geom`, neither of which is ours to move.
Hence committing the extracted q, which is the only part the indexing claim depends on.

**The original harness no longer runs at all.** `hybrid_full_prok.py` lives in NERSC scratch and
imports `fftindex.hybrid_stream` — the pre-rename package, whose on-disk copy has since been renamed
`_fftindex_stale`. `prok_rerun.py` (session scratch, S3DF job 34241712) replaces it and runs against
the current package from this file.

---

## The general point

Each of these is small — 33 kB, 2 MB. Each is load-bearing for a figure in a paper. A published
number whose input exists in one scratch directory is one `rm -rf`, one purge policy, or one
laptop away from being unreproducible, and no amount of care on the working machine detects it.
If a number goes in a deliverable, the input it came from belongs in the repo.
