#!/bin/bash
#SBATCH -A m5149 -C cpu -q shared -c 8 --mem=30G -t 00:20:00 -o /pscratch/sd/k/kmdalton/cxidb_45/work/probe/ccshells2_%j.out
PY="env PYTHONNOUSERSITE=1 /pscratch/sd/k/kmdalton/glint/.mamba/envs/glint/bin/python -u /pscratch/sd/k/kmdalton/cxidb_45/work/probe/ccshells2.py"
echo "== CXIDB33"; $PY /pscratch/sd/k/kmdalton/cxidb_33/work/partialator 79.2,79.2,37.9,90,90,90 30,1.9,12 2.4,1.9,10
echo "== CXIDB45"; $PY /pscratch/sd/k/kmdalton/cxidb_45/work/partialator 68.5,68.5,108.6,90,90,90 30,1.06,12 1.6,1.06,12
