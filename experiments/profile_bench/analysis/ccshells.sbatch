#!/bin/bash
#SBATCH -A m5149 -C cpu -q shared -c 8 --mem=30G -t 00:20:00 -o /pscratch/sd/k/kmdalton/cxidb_45/work/probe/ccshells_%j.out
PYTHONNOUSERSITE=1 /pscratch/sd/k/kmdalton/glint/.mamba/envs/glint/bin/python -u /pscratch/sd/k/kmdalton/cxidb_45/work/probe/ccshells.py /pscratch/sd/k/kmdalton/cxidb_45/work/partialator
