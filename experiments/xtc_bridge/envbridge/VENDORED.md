# envbridge (vendored)

Copied from `slac-lcls/drp-benchmarks`, directory `envbridge/envbridge/`, at commit 5cd09f8
(includes drp-benchmarks#34, the conda PYTHONPATH fix). Same author as GLINT; covered by GLINT's
`LICENSE.md`. Vendored so that `glint_xtc.py --psana 2` works without access to drp-benchmarks.

One change from upstream: `Worker(..., call_timeout=...)`. Upstream bounds every call by the
activation timeout (120 s), which ended any `--psana 2` run longer than two minutes, since the reader
reads the whole run in one call. `call_timeout` defaults to the activation timeout, so other callers
behave as before; `glint_xtc.py --bridge-timeout` sets it (default: no limit).

API and design notes: the upstream README. Tests: `../test_envbridge_local.py`.
