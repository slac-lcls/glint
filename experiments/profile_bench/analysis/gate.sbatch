#!/bin/bash
#SBATCH -A m5149 -C cpu -q shared -c 2 --mem=8G -t 00:40:00 -J gate45 -o /pscratch/sd/k/kmdalton/cxidb_45/work/probe/gate_%j.out
export PYTHONNOUSERSITE=1
/pscratch/sd/k/kmdalton/glint/.mamba/envs/glint/bin/python -u /pscratch/sd/k/kmdalton/cxidb_45/work/probe/gate.py
