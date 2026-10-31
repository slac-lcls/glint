# glint_xtc_mpi GPU placement on real hardware (glint#256)

S3DF job **40623083**, 2026-10-12, `sdfampere002` (2× A100-SXM4-40GB). It ran `val.sbatch` as
written, against branch `rework/pin-gpu-256` @ `7516b19`, in both GLINT conda1 environments:
`ana-4.0.58-py3-minipytorch` (torch 2.1.0, CuPy 12.3, mpi4py 3.1.5) and `ana-4.0.59-py3-minipytorch`
(torch 1.11.0, CuPy 13.0, mpi4py 3.1.5). Every launch is `srun --mpi=pmix`. `val_40623083.out` is
the verbatim log. Its PMIx `munge` and `openib` banners are site noise, not failures.

`probe_placement.py` imports mpi4py first, as `glint_xtc_mpi.main()` does. It runs the wrapper's own
`agree_on_placement` on every rank, and asks a child process which GPU its CuPy device 0 is, the way
the `--psana 2` reader is started.

| launch | expected | both envs |
|---|---|---|
| a. `-n 2 --gpus-per-task=1` | run, different GPUs | RUN: both ranks see the mask `'0'`, on PCI 41 and 01 |
| b. `-n 2 --gpus=1` | stop | STOP: both on PCI 41, with the launch fix |
| c. b plus `--share-gpu` | run | RUN |
| d. `-n 2 --gpus=2 --gpu-bind=none` | stop | STOP: both see `'0,1'`, and both resolve to PCI 01 |
| e. `GLINT_GPUS_PER_NODE=2` exported | stop | STOP, with "Unset it" |
| g. `test_mpi_smoke.py`, 2 ranks, 1 GPU | all pass | ALL PASS, 24/24 events |

In a–d the child's GPU matched its rank's on every rank, and the environment was unchanged after
the check.

- **Why the check uses identity, not the mask.** In launch a, S3DF gives each task its own GPU but
  shows every task the same mask string, `'0'`. A comparison of mask strings would refuse this
  correct launch; the identity check accepts it.
- **Mona's timing point, measured (launch f, `timing_probe.py`).** In both envs, importing mpi4py
  does not map `libcuda`, and hiding every GPU after the import still takes effect: CuPy then sees
  0 devices. So in GLINT's own conda1 envs, writing the mask after `import mpi4py` would still have
  moved the rank. lcls2#175 measured the opposite in the psana2-gpu env, where CUDA is already
  initialized after `import psana`. Not writing the mask removes the dependence on that either way.

To rerun: copy the three scripts to an S3DF home directory, set `GLINT_ROOT` in `val.sbatch` to a
checkout, and run `sbatch val.sbatch` from `sdfiana027`. Use an account with GPU time on `ampere`.
