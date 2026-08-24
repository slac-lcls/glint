# Third-party notices

GLINT is licensed under the terms in [`LICENSE.md`](LICENSE.md). This file records third-party
material and the notices that accompany it.

GLINT **vendors no third-party libraries**: there is no `vendor/`, `third_party/` or `external/`
directory, no git submodule, and no third-party source file in the tree.

---

## 1. Third-party code incorporated

### psana `Detector/UtilsCommonMode.py` — REMOVED 2026-08-24, no psana code remains

`experiments/xtc_bridge/test_gpu_calib_cm.py` previously contained a verbatim transcription of five
functions — `common_mode_rows`, `common_mode_cols`, `common_mode_2d`,
`common_mode_rows_hsplit_nbanks`, `common_mode_2d_hsplit_nbanks` — from psana's
`Detector/UtilsCommonMode.py` (release `ana-4.0.58-py3`), created 2018-01-31 by **Mikhail Dubrovin**
(SLAC/LCLS).

Upstream (`github.com/lcls-psana/Detector`) publishes no LICENSE, COPYING or NOTICE file, so no
grant attached to that code and it could not be covered by this repository's licence.

**The transcription has been deleted.** The test now imports those five names from psana at run
time and exits 0 with a `SKIPPED` message when psana is unavailable. No psana code remains in this
repository. Verified against psana `ana-4.0.59-py3-minipytorch`: all five modes and the `npix_min`
boundary agree to `maxdiff 0.000e+00`.

⚠️ Consequence for CI: the CPU-only CI job has no psana, so that step now skips and provides **no**
regression coverage for `GpuCalibrator._common_mode`. The check has to be run where psana exists.

### fast-feedback-indexer (ffbidx) — BSD-3-Clause

`glint/replica_gpu.py` and `glint/replica_gpu_batch.py` implement the fast-feedback cell-assembly
method in original PyTorch/CuPy, written with reference to the C++/CUDA source of
`github.com/paulscherrerinstitute/fast-feedback-indexer`. These modules **are** part of the
distributed package.

> Copyright 2022 Paul Scherrer Institute
>
> Redistribution and use in source and binary forms, with or without modification, are permitted
> provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice, this list of conditions
>    and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice, this list of
>    conditions and the following disclaimer in the documentation and/or other materials provided
>    with the distribution.
> 3. Neither the name of the copyright holder nor the names of its contributors may be used to
>    endorse or promote products derived from this software without specific prior written
>    permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
> IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
> FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
> CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
> DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
> IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
> OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

(The upstream `LICENSE.md` additionally records that its bundled `eigen` directory is MPL-2.0 and
that Python carries the PSF licence. GLINT bundles neither.)

### psana calibration

`experiments/xtc_bridge/gpu_calib.py` reproduces `det.calib` in CuPy, written with reference to
psana's `Detector/UtilsEpix10ka.py` and `UtilsCommonMode.py`. Not distributed. Terms as in section 1.

---

## 3. Published methods implemented in original code

The following are **not** third-party material: each is GLINT's own implementation of a published
algorithm, and no upstream source is present. They are listed because the names appear throughout the
code and documentation, and because the authors deserve citation. Per-module `Credit:` docstrings
carry the detail.

| GLINT module | Method and authors |
|---|---|
| `glint/peakfinder8.py` | peakfinder8 — A. Barty *et al.*, "Cheetah", *J. Appl. Cryst.* **47**, 1118–1131 (2014); also CrystFEL (T. A. White *et al.*) |
| `glint/peakfinder9.py` | peakfinder9 — Y. Gevorkov (MSc thesis, CFEL/DESY); CrystFEL integration by T. A. White; EuXFEL calNG port by D. Hammer |
| `glint/peakfinder_v4.py` | psalgos `peak_finder_v4r3` / `peaks_droplet` — M. Dubrovin (SLAC/LCLS) |
| `glint/radial.py` | sparse-LUT + pixel-split azimuthal integration — pyFAI, Kieffer *et al.*, *J. Appl. Cryst.* **48**, 510–519 (2015) |
| `glint/glint_index.py` | direct-sum objective — xgandalf, Gevorkov *et al.*, *Acta Cryst.* **A75**, 694–704 (2019); assembly — TORO, Gasparotto *et al.* (2024) |
| `glint/seed.py` | difference vectors — DirAx, Duisenberg (1992); projection-slice seeding — DPS/MOSFLM, Steller, Bolotovsky & Rossmann (1997) |
| `glint/multishot.py` | pair-angle matching — TakeTwo, Ginn *et al.* (2016) |
| `glint/partiality.py` | partiality/scaling model — `partialator`, CrystFEL |
| `experiments/powder_index/powder_index.py` | powder indexing — de Wolff M20, ITO, TREOR, DICVOL, McMaille |

---

## 4. Third-party-derived data

Numeric quantities derived by GLINT's own extraction scripts from public deposited datasets. **No
deposited images are redistributed.** See `experiments/DATA_PROVENANCE.md`.

| In-tree file | Derived from |
|---|---|
| `experiments/frames_cxidb_clean.txt` (and duplicates under `experiments/xgandalf/`) | CXIDB entry **17** — lysozyme, LCLS-CXI |
| `experiments/prok_q.npz` | CXIDB entry **45** — Proteinase K, SACLA MPCCD; geometry from CXIDB entry **62** |

**Status: unresolved.** Deposition terms have not been established against the `cxidb.org` entry
pages. The originating publications are cited in the accompanying paper.

---

## 5. Dependencies

Used as libraries or invoked as external programs; no source is copied into this repository.

**Distributed dependency closure:** numpy (BSD-3-Clause), scipy (BSD-3-Clause), torch
(BSD-3-Clause). Optional accelerators, imported under `try`/`except`: cupy (MIT), numba
(BSD-2-Clause).

**Research and validation paths only (`experiments/`, `lute/`; not distributed):** DIALS and
cctbx/scitbx/iotbx/rstbx/simtbx (BSD-3-Clause), pyFAI (MIT), pyopencl (MIT), GSAS-II, h5py, mpi4py,
scikit-learn, joblib (BSD-3-Clause), matplotlib (PSF-based), pydantic, pytest (MIT), requests
(Apache-2.0), psana / Detector / PSCalib.

**Linked at build time by two first-party benchmark drivers** (`experiments/xgandalf/xg_driver.cpp`,
`ffbidx_driver.cpp`): libxgandalf (**GPL-3.0-or-later**), Eigen (MPL-2.0), ffbidx C API
(BSD-3-Clause). No build script is committed and the compiled binaries are git-ignored, so no
GPL-linked binary is distributed. **Distributing a compiled `xg_driver` would place that binary under
GPL-3.0** — keep `experiments/` outside any distribution.

**Invoked as external programs** (separate processes, no linking): CrystFEL `process_hkl`,
`partialator`, `ambigator` (GPL-3.0-or-later); `pdftotext` (GPL-2.0, optional).

**CI:** `actions/checkout`, `actions/setup-python` (MIT).

Licence identifiers above are the standard published terms of each project; verify against the
version actually installed before relying on them in a legal filing.
