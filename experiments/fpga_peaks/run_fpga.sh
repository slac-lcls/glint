#!/bin/bash
PY=/sdf/group/lcls/ds/ana/sw/conda1/inst/envs/ana-4.0.58-py3-minipytorch/bin/python3
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$ROOT"
GLINT_ROOT=$PWD $PY experiments/fpga_peaks/fpga_selection.py --json experiments/fpga_peaks/fpga_selection.json
