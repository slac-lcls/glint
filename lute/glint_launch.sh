#!/bin/bash
# GLINT launcher for the LUTE `IndexGLINT` ThirdPartyTask: activate the GLINT GPU (torch) env and run
# the productized glint CLI. LUTE builds the flags (--peaks/--geom/--cell/--mode/--fromfile/...) and
# invokes:  glint_launch.sh <flags>.  Point IndexGLINTParameters.executable at this script.
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh >/dev/null 2>&1
conda activate ana-4.0.58-py3-minipytorch >/dev/null 2>&1
cd "$(dirname "$0")/.." || exit 1                       # repo root (so `fftindex` imports)
exec python -m fftindex.glint_cli "$@"
