#!/bin/bash
#SBATCH -A m5149 -C cpu -q debug -N 1 -t 00:30:00 -o /pscratch/sd/k/kmdalton/cxidb_45/work/probe/deck_stats_%j.out
PY="env PYTHONNOUSERSITE=1 /pscratch/sd/k/kmdalton/glint/.mamba/envs/glint/bin/python -u /pscratch/sd/k/kmdalton/cxidb_45/work/probe/deck_stats.py"
echo "== CXIDB33"; $PY /pscratch/sd/k/kmdalton/cxidb_33/work/partialator 79.2,79.2,37.9,90,90,90 1.9 12 8
echo "== CXIDB45"; $PY /pscratch/sd/k/kmdalton/cxidb_45/work/partialator 68.5,68.5,108.6,90,90,90 1.06 14 10
