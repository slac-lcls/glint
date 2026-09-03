#!/bin/bash
PY=/sdf/group/lcls/ds/ana/sw/conda1/inst/envs/ana-4.0.58-py3-minipytorch/bin/python3
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
GLINT_ROOT="$ROOT" "$PY" "$ROOT/experiments/fpga_peaks/fpga_selection.py" --json "$ROOT/experiments/fpga_peaks/fpga_selection.json"
