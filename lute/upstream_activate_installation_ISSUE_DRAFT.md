DRAFT for a GLINT maintainer to review and, if it still applies against current `slac-lcls/lute`,
open against that repo. Not opened by this PR or this repo. Source: glint#128.

---

**Title:** `install/bin/activate_installation` silently selects an unpopulated python tree instead
of failing loudly

**Body:**

`install/bin/activate_installation` derives its library path from the ambient interpreter:

```bash
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
LUTE_LIB_PATH=(${LUTE_INSTALL_PATH}lib/python${PY_VER}/site-packages)
export PYTHONPATH="${LUTE_LIB_PATH}:${PYTHONPATH}"
```

An install can carry several `lib/pythonX.Y` trees side by side (in our case 3.9, 3.11, 3.12), and
not all of them are necessarily populated with `launch_scripts`. When the ambient `python3` resolves
to an unpopulated one, this script reports nothing wrong: it prints a path and exports a
`PYTHONPATH` that resolves to nothing. Every downstream LUTE task then fails with
`ModuleNotFoundError: No module named 'launch_scripts'`, or -- when reached via
`submit_launch_slurm.sh` / a workflow manager -- a bare subprocess return code `127`, far from the
actual cause. `install/bin/launch_slurm` and `install/bin/submit_slurm` additionally bake in a
shebang pinned to a specific interpreter, so the python3 that matters may not even be the one
ambient in the caller's shell.

Two smaller issues observed in the same path, which may or may not still apply upstream:
- `launch_scripts/activate_installation` does not exist (only `install/bin/activate_installation`
  does), yet `launch_scripts/submit_launch_slurm.sh` sources it -- failing silently, so its `.py`
  launcher then runs as a shell script.
- `python -m launch_scripts.<module>` run from inside the LUTE source checkout picks up the
  source-tree `launch_scripts/` (via CWD preceding `PYTHONPATH`) rather than the installed one,
  which is confusing to debug since the same command works from any other directory.

**Suggested fix:** have `activate_installation` verify that the tree it selected actually contains
`launch_scripts`, and fail loudly (nonzero exit, message naming the ambient python3 version and
what to do about it) rather than exporting a `PYTHONPATH` into the void. Alternatively, populate all
shipped `lib/pythonX.Y` trees at install time, or generate the `install/bin/*` shebangs against an
interpreter whose tree is actually populated.

**Environment (as reported against our install):** `~/git/lute_new/lute` @ `c457afc`
(`file_indexing`, `v0.2.0-3`), S3DF. Note that checkout was also well behind `origin/dev` at the
time, so please check whether this is already fixed there before filing.
