# Contributing to GLINT

Full walkthrough: [`docs/onboarding.md`](docs/onboarding.md). The short version:

1. **GitHub is canonical** (`slac-lcls/glint`). Branch → PR → review → merge to `main`; keep `main` green.
   Tiny low-risk fixes may go direct to `main` — use judgement.
2. **Validate before you push.** `python experiments/test_cli_smoke.py` must pass (6/6), and no
   indexing-rate regression on the 120-frame cxidb set (blind ~71% gated) or the 10-cell dense sweep.
3. **Scope changes.** Shipped engine goes in `glint/`; throwaway analysis and benchmarks go in `experiments/`.
4. **Package name:** `import glint` (the `fftindex` alias still works). Setup per environment is in the
   onboarding doc (S3DF / NERSC / laptop).

Questions or access (repo, compute, the paper): ask Stefano.
