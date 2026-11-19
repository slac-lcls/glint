#!/bin/bash
#SBATCH -A m5149 -C cpu -q debug -N 1 -t 00:15:00 -o /pscratch/sd/k/kmdalton/cxidb_45/work/probe/subsample/smoke_%j.out
PYTHONNOUSERSITE=1 /pscratch/sd/k/kmdalton/glint/.mamba/envs/glint/bin/python -u /pscratch/sd/k/kmdalton/cxidb_45/work/probe/subsample/smoke.py
