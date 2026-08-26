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

## Deposited-data terms — CXIDB entries 17, 45 and 62 (verified 2026-08-25)

The two committed derived inputs above trace back to public CXIDB depositions. Their terms were
verified against `cxidb.org` on **2026-08-25** (previously this repository only assumed them):

**Site-wide policy.** The CXIDB deposition page states, verbatim: "All deposited data and metadata
are made available under the CC0 waiver to promote maximum reuse"
(<https://www.cxidb.org/deposit.html>, linking CC0 1.0,
<https://creativecommons.org/publicdomain/zero/1.0/>).

**Per-entry statements.** Each entry page below shows the visible statement "Licensed under the CC0
Public Domain Dedication Waiver. Please give proper credit via citations according to established
scientific practice." and embeds machine-readable metadata
`"license": "http://creativecommons.org/about/cc0"`. (The pages also carry a site-template footer
reading "© CXI DB. All rights reserved." — recorded here for completeness. The deposition policy
and per-entry licence metadata above are the statements specific to the deposited data; no legal
reading of the footer's relationship to them is asserted here.)

| Entry | URL | Dataset DOI | Depositor(s) | Publication |
|---|---|---|---|---|
| 17 (lysozyme, LCLS-CXI) | <https://www.cxidb.org/id-17.html> | 10.11577/1096920 | S. Boutet | Boutet *et al.*, *Science* **337**, 362 (2012), doi:10.1126/science.1217737 |
| 45 (Proteinase K, SACLA BL3, MPCCD) | <https://www.cxidb.org/id-45.html> | 10.11577/1350027 | M. Sugahara, T. Nakane | Masuda *et al.*, *Sci. Rep.* **7**, 45604 (2017), doi:10.1038/srep45604 |
| 62 (ACG, SACLA, MPCCD) | <https://www.cxidb.org/id-62.html> | 10.11577/1365656 | F. Yumoto, K. Yamashita | Yamashita *et al.*, *IUCrJ* **4** (2017), doi:10.1107/S2052252517008557 |

**What this repository holds, and from where:**

- `experiments/frames_cxidb_clean.txt` (and duplicates under `experiments/xgandalf/`) — peak lists
  derived by GLINT's own extraction from **entry 17** images. No deposited images are
  redistributed.
- `experiments/nbest_120.npz` — derived from the **entry-17** peak lists above
  (`frames_cxidb_clean.txt`) by GLINT's blind N-best indexing (top-3 hypotheses per frame; the
  producing run's wall time is recorded in the file's own `index_s` key, see its section above).
  The exact producing run predates this file's commit and no in-tree generator reproduces it —
  the file is a recorded artifact, pinned by size and sha256 above. It contains only derived
  lattice hypotheses; no deposited data are redistributed.
- `experiments/prok_q.npz` — reciprocal-lattice points extracted from **entry 45** images
  (`dump_prok_q.py`, NERSC; see above). The extraction used a detector geometry file
  (`mpccd-optimized.geom`) that originates from the **entry 62** deposition — so entry 45's derived
  data here depends on entry-62 material as well. Both entries state the same CC0 terms.

**Reading of the terms.** CC0 1.0 is a public-domain dedication: it imposes no licence condition
on reuse or redistribution of the deposited data or quantities derived from it. The per-entry
citation request is established scientific practice, not a legal condition; the originating
publications are cited in the accompanying paper. This resolves the item previously marked
"unresolved" in `THIRD_PARTY_NOTICES.md` §4.

---

## The general point

Each of these is small — 33 kB, 2 MB. Each is load-bearing for a figure in a paper. A published
number whose input exists in one scratch directory is one `rm -rf`, one purge policy, or one
laptop away from being unreproducible, and no amount of care on the working machine detects it.
If a number goes in a deliverable, the input it came from belongs in the repo.
