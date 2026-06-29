#!/bin/bash
# Stage A launcher: psana2 env + run extractor on a psana node (data mount lives there).
source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh >/dev/null 2>&1
cd /sdf/home/s/smarches/git/fftindex/experiments
exec python extract_strong_weak.py "$@"
