# Contributing to GLINT

Full walkthrough: [`docs/onboarding.md`](docs/onboarding.md). The short version:

1. **GitHub is canonical** (`slac-lcls/glint`). Branch → pull request → CI green → review → merge to `main`;
   keep `main` green. Small fixes go the same way: the suites and the number guard run on the pull request,
   and that is where a pin or a figure gets checked.
2. **Validate before you push.** `python experiments/run_ci_locally.py` runs every CPU suite CI runs, in an
   environment as poor as CI's (torch, cupy and numba blocked); `--check-sync` verifies that its list matches
   `ci.yml`, and CI runs that check too, so a suite added in one place and not the other fails. A change to the
   front end, the consensus, the rescue or the streaming driver must also hold the measured rates on a GPU:
   the 120-frame cxidb set (blind 77%, 92/120, at the strict ≥25%-of-spots bar), the 10-cell dense sweep, and
   the 480-frame streaming arm (`record_stream_replay.py --expect 333/480 --expect-wresc 10 --expect-relock 1`).
   If a change is meant to be bit-identical, verify it is.
3. **Numbers are guarded.** `experiments/check_numbers.py` scans the README, the docs and the manuscript for
   retired or unqualified figures; quote measured values from its `--facts` table, and when a measurement really
   changes, edit FACTS and re-run every target rather than the sentence.
4. **Scope changes.** Shipped engine goes in `glint/`; throwaway analysis, benchmarks and results notes go in
   `experiments/`. A new `StreamDriver` option is appended to the constructor (it is not keyword-only), gets a
   CPU suite, and a row in [`docs/stream_driver_options.md`](docs/stream_driver_options.md), which CI checks
   against the signature.
5. **Package name:** `import glint` (the `fftindex` alias still works). Setup per environment is in the
   onboarding doc (S3DF / NERSC / laptop).

Questions or access (repo, compute, the paper): ask Stefano.
