# envbridge (vendored)

Copied from `slac-lcls/drp-benchmarks`, directory `envbridge/envbridge/`, at commit 5cd09f8
(includes drp-benchmarks#34, the conda PYTHONPATH fix). Same author as GLINT; covered by GLINT's
`LICENSE.md`. Vendored so that `glint_xtc.py --psana 2` works without access to drp-benchmarks.

Changes from upstream:

**`Worker(..., call_timeout=...)`.** Upstream bounds every call by the
activation timeout (120 s), which ended any `--psana 2` run longer than two minutes, since the reader
reads the whole run in one call. `call_timeout` defaults to the activation timeout, so other callers
behave as before; `glint_xtc.py --bridge-timeout` sets it (default: no limit).

**Review fixes (glint#275), all with tests that fail on the upstream code:**
- literal PYTHONPATH entries and the socket path are shell-quoted, so a checkout path with a space
  no longer stops the worker from starting;
- a failed activation (timeout or bad handshake) kills the worker and closes the socket before
  raising, since `__exit__` never runs when `__enter__` raises;
- `close()` removes the socket file (an AF_UNIX socket leaves it behind);
- a 0-d array keeps shape `()` (it came back as `(1,)`), and object or structured dtypes are refused
  at encode (they came back as raw bytes or failed on the far side).

API and design notes: the upstream README. Tests: `../test_envbridge_local.py`.
